import os
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from Bio.PDB import PDBIO, PDBParser, Model, Chain, Residue, Atom
from keras.utils import pad_sequences
from tensorflow import keras
from keras import layers
from keras import mixed_precision
from einops import rearrange, reduce, repeat
from einops.layers.tensorflow import Rearrange, Reduce
import tensorflow_addons as tfa
import sidechainnet as scn
from sidechainnet.examples import losses, models
from sidechainnet.structure.structure import inverse_trig_transform
from sidechainnet.structure.build_info import NUM_ANGLES
import py3Dmol
import keras.backend as K
warnings.filterwarnings('ignore')
TF_GPU_ALLOCATOR = 'cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


class Attention(tf.keras.layers.Layer):
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

        self.to_q = layers.Dense(inner_dim, use_bias=False)
        self.to_kv = layers.Dense(inner_dim * 2, use_bias=False)
        self.to_out = layers.Dense(dim)
        self.gating = layers.Dense(inner_dim)
        self.dropout = layers.Dropout(dropout)
        self.to_out.build((None, None, inner_dim))
        self.to_out.kernel.assign(tf.zeros_like(self.to_out.kernel))

    def call(self, x, mask=None, attn_bias=None, context=None, context_mask=None, tie_dim=None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, context is not None
        context = x if context is None else context
        q, k, v = self.to_q(x), *tf.split(self.to_kv(context), num_or_size_splits=2, axis=-1)
        i, j = q.shape[-2], k.shape[-2]
        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h=h) for t in (q, k, v))

        # scale
        q = q * self.scale

        # query / key similarities
        if tie_dim is not None:
            q, k = (rearrange(t, '(b r) ... -> b r ...', r=tie_dim) for t in (q, k))
            q = tf.reduce_mean(q, axis=1)

            dots = tf.einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = tf.einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)
        if attn_bias is not None:
            dots = dots + attn_bias

        # masking
        if mask is not None:
            mask = tf.ones((1, i), dtype=tf.bool) if mask is None else mask
            context_mask = mask if not has_context else (tf.ones((1, k.shape[-2]), dtype=tf.bool) if context_mask is None else context_mask)
            mask_value = -tf.float32.max
            mask = tf.expand_dims(mask, 1)[:, :, :, None] * tf.expand_dims(context_mask, 1)[:, None, None, :]
            mask = tf.cast(mask, tf.bool)
            dots = tf.where(mask, dots, mask_value)

            # attention
            dots = dots - tf.reduce_max(dots, axis=-1, keepdims=True)
            attn = tf.nn.softmax(dots, axis=-1)
            attn = self.dropout(attn)
            # aggregate
            out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
            # merge heads
            out = rearrange(out, 'b h n d -> b n (h d)')
            # gating
            gates = self.gating(x)
            out = out * tf.sigmoid(gates)
            # combine to out
            out = self.to_out(out)
            return out


class ProteinNet(tf.keras.Model):
    def __init__(self,
                 d_hidden,
                 dim,
                 d_in=21,
                 d_embedding=32,
                 heads=8,
                 integer_sequence=True,
                 n_angles=12):

        super(ProteinNet, self).__init__()
        self.d_hidden = d_hidden

        self.attn = Attention(dim=dim, heads=heads)
        self.d_out = n_angles * 2

        self.hidden2out = tf.keras.Sequential([
            layers.Dense(d_embedding, activation=None),
            layers.Activation(tf.keras.activations.gelu),
            layers.Dense(self.d_out, activation=None)
        ])

        self.out2attn = layers.Dense(dim, activation=None)
        self.final = tf.keras.Sequential([
            layers.Activation(tf.keras.activations.gelu),
            layers.Dense(self.d_out, activation=None)
        ])
        self.norm_0 = layers.LayerNormalization(axis=-1)
        self.norm_1 = layers.LayerNormalization(axis=-1)
        self.activation_0 = layers.Activation(tf.keras.activations.gelu)
        self.activation_1 = layers.Activation(tf.keras.activations.gelu)
        self.output_activation = layers.Activation('tanh')

        self.integer_sequence = integer_sequence
        if self.integer_sequence:
            self.input_embedding = layers.Embedding(d_in, d_embedding, mask_zero=True)
        else:
            self.input_embedding = layers.Dense(d_embedding, activation=None)

    def get_lengths(self, sequence):
        if self.integer_sequence:
            lengths = tf.reduce_sum(tf.cast(sequence != 20, tf.int32), axis=1)
        else:
            lengths = tf.reduce_sum(tf.cast(tf.reduce_all(sequence != 0, axis=-1), tf.int32), axis=1)
        return lengths

    def call(self, sequence, mask=None):
        lengths = self.get_lengths(sequence)
        sequence = self.input_embedding(sequence)

        sorted_lengths, sorted_indices = tf.math.top_k(lengths, k=tf.shape(lengths)[0], sorted=True)
        sorted_lengths = tf.cast(sorted_lengths, tf.int32)
        sorted_sequence = tf.gather(sequence, sorted_indices)

        # Compute the padding required
        max_length = tf.reduce_max(sorted_lengths)
        padding_required = max_length - sorted_lengths

        # Pad the sequences
        def pad_sequence(i):
            pad_shape = tf.stack([padding_required[i], tf.shape(sorted_sequence)[-2], tf.shape(sorted_sequence)[-1]])
            padding_tensor = tf.zeros(pad_shape, dtype=tf.float32)
            return tf.concat([sorted_sequence[i], padding_tensor], axis=0)

        padded_sequences = tf.map_fn(pad_sequence, tf.range(tf.shape(sorted_sequence)[0]), dtype=tf.float32)

        output = self.hidden2out(padded_sequences)
        output = self.out2attn(output)
        output = self.activation_0(output)
        output = self.norm_0(output)
        output = self.attn(output, mask=mask)
        output = self.activation_1(output)
        output = self.norm_1(output)
        output = self.final(output)
        output = self.output_activation(output)
        output = tf.reshape(output, (tf.shape(output)[0], tf.shape(output)[1], 12, 2))

        return output


def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    mse = K.square(y_pred - y_true) * mask
    return K.sum(mse) / K.sum(mask)


def generator_fn(loader):
    for batch in loader:
        sequences, angles = batch.seqs.numpy(), batch.angs.numpy()
        yield sequences, angles


def pytorch_loader_to_tf_dataset(loader, batch_size):
    def gen():
        for sequences, angles in loader:
            yield sequences.numpy(), angles.numpy()

    output_signature = (
        tf.TensorSpec(shape=(None, None, 21), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 12), dtype=tf.float32)
    )

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    ).batch(batch_size)


def main():
    # Load the data in the appropriate format for training.
    batch_size = 4
    dataloader = scn.load(
        with_pytorch="dataloaders",
        batch_size=batch_size,
        dynamic_batching=False,
        thinning=30,
        num_workers=0,
        seq_as_onehot=True)
    # print("Available Dataloaders:", list(dataloader.keys()))

    dataloaders = {}
    for key in dataloader.keys():
        if key in ['train', 'test', 'valid-20']:
            dataloaders[key] = pytorch_loader_to_tf_dataset(dataloader[key], batch_size)
    # print("Dataset:", dataloaders)

    # Create the model
    model = ProteinNet(d_hidden=512,
                       dim=256,
                       d_in=21,
                       d_embedding=32,
                       integer_sequence=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=masked_mse)

    history = model.fit(
        dataloaders['train'],
        epochs=20,
        validation_data=dataloaders['valid-20'],
        verbose=1
    )

    model.summary()

    # epochs = 20
    # train_losses = []
    # val_losses = []
    #
    # for epoch in range(epochs):
    #     print(f'Epoch {epoch + 1}/{epochs}')
    #
    #     # Training
    #     train_loss = 0
    #     train_steps = 0
    #     for sequences, angles in dataloaders['train']:
    #         with tf.GradientTape() as tape:
    #             # Forward pass
    #             predictions = model(sequences)
    #             # Compute loss
    #             loss = mse_loss(angles, predictions)
    #         # Compute gradients
    #         gradients = tape.gradient(loss, model.trainable_variables)
    #         # Update weights
    #         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #         train_loss += loss.numpy()
    #         train_steps += 1
    #
    #     train_loss /= train_steps
    #     train_losses.append(train_loss)
    #     print(f"Train Loss: {train_loss:.4f}")
    #
    #     # Validation
    #     val_loss = 0
    #     val_steps = 0
    #     for sequences, angles in dataloaders['valid-20']:
    #         predictions = model(sequences)
    #         loss = mse_loss(angles, predictions)
    #
    #         val_loss += loss.numpy()
    #         val_steps += 1
    #
    #     val_loss /= val_steps
    #     val_losses.append(val_loss)
    #     print(f"Validation Loss: {val_loss:.4f}")

    # Plot loss
    plt.plot(history.history["loss"], label='Train Loss')
    plt.plot(history.history["val_loss"], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    pass


if __name__ == '__main__':
    print("Hello world!")
    main()
    print("\nDone!")
