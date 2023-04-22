import os
import time
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


def pytorch_loader_to_tf_dataset(loader, batch_size):
    def gen():
        for batch in loader:
            sequences, angles = batch.seqs.numpy(), batch.angs.numpy()
            yield sequences, angles

    output_signature = (
        tf.TensorSpec(shape=(None, None, 20), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 12), dtype=tf.float32)
    )

    # Pad the sequences to the maximum length in the batch
    max_seq_len = 0
    max_angle_len = 0
    for batch in loader:
        max_seq_len = max(max_seq_len, batch.seqs.shape[1])
        max_angle_len = max(max_angle_len, batch.angs.shape[1])

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    ).padded_batch(
        batch_size,
        padded_shapes=([None, max_seq_len, 20], [None, max_angle_len, 12])
    )


def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output


class ProteinTransformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, output_dim, dropout_rate=0.1):
        super(ProteinTransformer, self).__init__()
        self.d_model = d_model

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dense_output = layers.Dense(output_dim)
        self.dropout_output = layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training, mask)

        x = self.dense_output(x)
        x = self.dropout_output(x, training=training)

        return x


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

    # Define the model's hyperparameters
    num_layers = 4
    d_model = 128
    num_heads = 8
    dff = 512
    input_vocab_size = 21
    output_dim = 12
    dropout_rate = 0.1

    model = ProteinTransformer(num_layers, d_model, num_heads, dff, input_vocab_size, output_dim, dropout_rate)

    # Define the optimizer, loss function, and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_object = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')

    @tf.function
    def train_step(inp, tar):
        with tf.GradientTape() as tape:
            predictions = model(inp, training=True)
            loss = loss_object(tar, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    # Define the validation step
    @tf.function
    def valid_step(inp, tar):
        predictions = model(inp, training=False)
        loss = loss_object(tar, predictions)

        valid_loss(loss)

    # Training and validation loop
    epochs = 20

    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        valid_loss.reset_states()

        for (batch, (inp, tar)) in enumerate(dataloaders['train']):
            train_step(inp, tar)

        for (batch, (inp, tar)) in enumerate(dataloaders['valid-20']):
            valid_step(inp, tar)

        print(f'Epoch {epoch + 1}, '
              f'Train Loss: {train_loss.result()}, '
              f'Validation Loss: {valid_loss.result()}, '
              f'Time taken for this epoch: {time.time() - start:.2f} secs')

    model.summary()
    tf.keras.utils.plot_model(model, to_file='protein_transformer.png', show_shapes=True)

    # Save the trained model
    model.save_weights('protein_transformer_weights.h5')

    # Test the model on a protein sequence
    test_sequence = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPR" \
                    "LASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSS" \
                    "TNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENP" \
                    "NSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
    input_sequence = np.array([scn.onehot(test_sequence, 'seq')], dtype=np.float32)

    predicted_angles = model(input_sequence, training=False)
    predicted_angles = np.squeeze(predicted_angles.numpy(), axis=0)
    # Save the predicted structure to a PDB file
    coords, _, _ = inverse_trig_transform(predicted_angles)
    scn.save_pdb(coords.numpy(), test_sequence, "predicted_structure.pdb")
    pass


if __name__ == '__main__':
    print("Hello world!")
    main()
    print("\nDone!")
