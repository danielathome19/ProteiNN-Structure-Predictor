import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat
import sidechainnet as scn
from sidechainnet.structure.structure import inverse_trig_transform
from sidechainnet.structure.build_info import NUM_ANGLES
import py3Dmol
import torch
from torch import nn, einsum
import random
from inspect import isfunction
from tqdm import tqdm
import dill as pickle


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else (val,) * depth


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)


def init_loss_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters())
    batch_losses = []
    epoch_training_losses = []
    epoch_test_losses = []
    mse_loss = torch.nn.MSELoss()
    return optimizer, batch_losses, epoch_training_losses, epoch_test_losses, mse_loss


def build_visualizable_structures(model, data, device):
    # For one batch of data, build a structure using the model's predictions
    with torch.no_grad():
        for batch in data:
            model_input = batch.int_seqs.to(device)
            mask_ = batch.msks.to(device)
            # Make predictions for angles and construct 3D atomic coordinates
            predicted_angles_sincos = model(model_input, mask=mask_)
            # Use this function to recover the original angles because the model predicts sin/cos values
            predicted_angles = inverse_trig_transform(predicted_angles_sincos)
            # Use BatchedStructureBuilder to build an entire batch of structures
            sb_pred = scn.BatchedStructureBuilder(batch.int_seqs, predicted_angles.cpu())
            sb_true = scn.BatchedStructureBuilder(batch.int_seqs, batch.crds.cpu())
            break
    return sb_pred, sb_true


def plot_protein(exp1, exp2):
    p = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', viewergrid=(2, 1))
    p.addModel(open(exp1, 'r').read(), 'pdb', viewer=(0, 0))
    p.addModel(open(exp2, 'r').read(), 'pdb', viewer=(1, 0))
    p.setStyle({'cartoon': {'color': 'spectrum'}})
    p.zoomTo()
    p.show()


def encode_sequence(sequence):
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
    return [aa_to_int[aa] for aa in sequence]


def predict_train(model, dataloader, device):
    s_pred, s_true = build_visualizable_structures(model, dataloader["train"], device)
    z_idx = 2
    for idx in range(3):
        s_pred.to_pdb(idx, path='{}_{}_pred.pdb'.format(idx, z_idx))
        s_true.to_pdb(idx, path='{}_{}_true.pdb'.format(idx, z_idx))
        # plot_protein('{}_{}_pred.pdb'.format(idx, z_idx), '{}_{}_true.pdb'.format(idx, z_idx))  # For Jupyter


def predict_sequence(model, sequence, device):
    int_seq = torch.tensor(encode_sequence(sequence)).unsqueeze(0).to(device)
    mask = torch.ones(int_seq.shape).to(device)
    predicted_angles_sincos = model(int_seq, mask=mask)
    predicted_angles = inverse_trig_transform(predicted_angles_sincos)
    sb_pred = scn.BatchedStructureBuilder(int_seq, predicted_angles.cpu())
    sb_pred.to_pdb(0, path='input_pred.pdb')


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            seq_len=None,
            heads=8,
            dim_head=64,
            dropout=0.0,
            gating=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x, mask=None, attn_bias=None, context=None, context_mask=None, tie_dim=None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)
        context = default(context, x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        i, j = q.shape[-2], k.shape[-2]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # scale
        q = q * self.scale

        # query / key similarities
        if exists(tie_dim):
            # as in the paper, for the extra MSAs
            # they average the queries along the rows of the MSAs
            # this particular module is named MSAColumnGlobalAttention

            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r=tie_dim), (q, k))
            q = q.mean(dim=1)

            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)
        if exists(attn_bias):
            dots = dots + attn_bias

        # masking
        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device=device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda:
                                                                torch.ones(1, k.shape[-2], device=device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            try:
                mask = mask.to(torch.bool)
                dots = dots.masked_fill(~mask, mask_value)
            except:
                # dots = dots.masked_fill(mask, mask_value)
                try:  # TODO: remove, this is a hack for now
                    dots = dots.masked_fill(mask, mask_value)
                except:
                    mask = mask[:, :, :dots.shape[-2], :dots.shape[-1]]
                    dots = dots.masked_fill(mask, mask_value)

        # attention
        dots = dots - dots.max(dim=-1, keepdims=True).values
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        # gating
        gates = self.gating(x)
        out = out * gates.sigmoid()
        # combine to out
        out = self.to_out(out)
        return out


class ProteinNet(nn.Module):
    """A protein sequence-to-angle model that consumes integer-coded sequences."""

    def __init__(self,
                 d_hidden,
                 dim,
                 d_in=21,
                 d_embedding=32,
                 heads=8,
                 integer_sequence=True,
                 n_angles=scn.structure.build_info.NUM_ANGLES):

        super(ProteinNet, self).__init__()
        # Dimensionality of RNN hidden state
        self.d_hidden = d_hidden

        self.attn = Attention(dim=dim, heads=heads)
        # Output vector dimensionality (per amino acid)
        self.d_out = n_angles * 2
        # Output projection layer. (from RNN -> target tensor)
        self.hidden2out = nn.Sequential(
            nn.Linear(d_embedding, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, self.d_out)
        )
        self.out2attn = nn.Linear(self.d_out, dim)
        self.final = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, self.d_out))
        self.norm_0 = nn.LayerNorm([dim])
        self.norm_1 = nn.LayerNorm([dim])
        self.activation_0 = nn.GELU()
        self.activation_1 = nn.GELU()

        # Activation function for the output values (bounds values to [-1, 1])
        self.output_activation = torch.nn.Tanh()

        # Embed model's input differently depending on the type of input
        self.integer_sequence = integer_sequence
        if self.integer_sequence:
            self.input_embedding = torch.nn.Embedding(d_in, d_embedding, padding_idx=20)
        else:
            self.input_embedding = torch.nn.Linear(d_in, d_embedding)

    def get_lengths(self, sequence):
        """Compute the lengths of each sequence in the batch."""
        if self.integer_sequence:
            lengths = sequence.shape[-1] - (sequence == 20).sum(axis=1)
        else:
            lengths = sequence.shape[1] - (sequence == 0).all(axis=-1).sum(axis=1)
        return lengths.cpu()

    def forward(self, sequence, mask=None):
        """Run one forward step of the model."""
        # Compute sequence lengths
        lengths = self.get_lengths(sequence)

        # Embed input tensors for input to the RNN
        sequence = self.input_embedding(sequence)

        # Pass in data into the RNN via PyTorch's pack_padded_sequences
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True)
        
        # At this point, output has the same dimensionality as the RNN's hidden state: i.e. (batch, length, d_hidden)
        # Use a linear transformation to transform the output tensor into the correct dimensionality (batch, length, 24)
        output = self.hidden2out(output)
        output = self.out2attn(output)
        output = self.activation_0(output)
        output = self.norm_0(output)
        output = self.attn(output, mask=mask)
        output = self.activation_1(output)
        output = self.norm_1(output)
        output = self.final(output)

        # Bound the output values between [-1, 1]
        output = self.output_activation(output)

        # Reshape the output to be (batch, length, angle, (sin/cos val))
        output = output.view(output.shape[0], output.shape[1], 12, 2)

        return output


def main(mode="train", sequence=""):
    warnings.filterwarnings('ignore')
    seed = 0
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    batch_size = 4
    dataloader = scn.load(
        with_pytorch="dataloaders",
        batch_size=batch_size,
        dynamic_batching=False,
        num_workers=0)
    # print("Available Dataloaders:", list(dataloader.keys()))

    def validation(model, datasplit):
        # Evaluate a model (sequence->sin/cos represented angles [-1,1]) on MSE.
        total = 0.0
        n = 0
        print("Running validation...")
        with torch.no_grad():
            for batch in datasplit:
                # Prepare variables and create a mask of missing angles (padded with zeros)
                # The mask is repeated in the last dimension to match the sin/cos representation.
                seqs = batch.int_seqs.to(device).long()
                mask_ = batch.msks.to(device)
                true_angles_sincosine = scn.structure.trig_transform(batch.angs).to(device)
                mask = (batch.angs.ne(0)).unsqueeze(-1).repeat(1, 1, 1, 2)

                # Make predictions and optimize
                predicted_angles = model(seqs, mask=mask_)
                loss = mse_loss(predicted_angles[mask], true_angles_sincosine[mask])

                total += loss
                n += 1

        return torch.sqrt(total / n)

    def train(model, n_epoch):
        for epoch in range(n_epoch):
            print(f'Epoch {epoch}')
            progress_bar = tqdm(total=len(dataloader['train']), smoothing=0)
            for batch in dataloader['train']:
                seqs = batch.int_seqs.to(device).long()
                mask_ = batch.msks.to(device)
                true_angles_sincos = scn.structure.trig_transform(batch.angs).to(device)
                mask = (batch.angs.ne(0)).unsqueeze(-1).repeat(1, 1, 1, 2)

                predicted_angles = model(seqs, mask=mask_)
                loss = mse_loss(predicted_angles[mask], true_angles_sincos[mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()

                # Housekeeping
                batch_losses.append(float(loss))
                progress_bar.update(1)
                progress_bar.set_description(f"\rRMSE Loss = {np.sqrt(float(loss)):.4f}")
                
            # Evaluate the model's performance on train-eval, downsampled for efficiency
            epoch_training_losses.append(validation(model, dataloader['train-eval']))
            print(f"     Train-eval loss = {epoch_training_losses[-1]:.4f}")
            torch.save(model.state_dict(), 'model.pt')
            
        # Evaluate the model on the test set
        epoch_test_losses.append(validation(model, dataloader['test']))
        print(f"Test loss = {epoch_test_losses[-1]:.4f}")

    model = ProteinNet(d_hidden=512,
                       dim=256,
                       d_in=49,
                       d_embedding=32,
                       integer_sequence=True)
    model = model.to(device)
    optimizer, batch_losses, epoch_training_losses, epoch_test_losses, mse_loss = init_loss_optimizer(model)

    if mode == "train":
        train(model, 25)

        # Export the model to ONNX for visualization in Netron
        batch = next(iter(dataloader['train']))
        seqs = batch.int_seqs.to(device).long()
        mask_ = batch.msks.to(device)
        torch.onnx.export(model, (seqs, mask_), "model.onnx", opset_version=12)

        # Plot the loss of each batch over time
        plt.plot(np.sqrt(np.asarray(batch_losses)), label='batch loss')
        plt.ylabel("RMSE")
        plt.xlabel("Step")
        plt.title("Training Loss over Time")
        plt.show()

        # Plot the loss of each epoch over time
        plt.plot([x.cpu().detach().numpy() for x in epoch_training_losses], label='train-eval')
        plt.plot([x.cpu().detach().numpy() for x in epoch_test_losses], label='test')
        plt.ylabel("RMSE")
        plt.xlabel("Epoch")
        plt.title("Training and Validation Losses over Time")
        plt.legend()
        plt.show()

        predict_train(model, dataloader, device)

    if mode == "predict":
        model.load_state_dict(torch.load('model.pt'))
        predict_sequence(model, sequence, device)

    pass


if __name__ == '__main__':
    modeMain = input("Choose a mode from one of the following: train, predict: ")
    if modeMain not in ["train", "predict"]:
        raise ValueError(f"Invalid mode: {modeMain}")
    else:
        sequenceMain = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPR" \
                       "LASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSS" \
                       "TNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENP" \
                       "NSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
        if modeMain == "predict":
            s_in = input("Enter a protein sequence, or hit 'Enter' for the default: ")
            sequenceMain = s_in if s_in else sequenceMain
        main(modeMain, sequenceMain)
    print("\nDone!")
