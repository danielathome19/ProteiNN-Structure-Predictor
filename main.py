import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import Bio as bp
import os
import sys
import time
import datetime
import math
import random
import pickle
import gc
import transformer as tfr
import warnings
import tensorflow_datasets as tfds
# import tensorflow_models as tfm
import data_utils as du
import keras_transformer as ktr
from tensorflow import keras as K
from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from seqgan import SeqGAN
warnings.filterwarnings('ignore')


def demo_transformer():
    sequences = []

    seq_len = 256  # change to the actual sequence length later on
    vocab_size = 21
    embed_dim = 64
    num_heads = 4
    ff_dim = 256
    num_blocks = 6
    output_dim = 2 * seq_len  # 2 torsion angles (phi, psi) for each residue

    padded_sequences = K.utils.pad_sequences(sequences, maxlen=seq_len, padding='post', truncating='post')

    # Create the model
    # Note that this is a simple example and may not achieve high accuracy.
    # To improve the model, maybe explore advanced techniques, such as incorporating evolutionary information from
    # multiple sequence alignments, using larger datasets, or employing more sophisticated architecture designs.
    # model = tfr.protein_structure_model(seq_len, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, output_dim)
    model = tfr.Transformer(vocab_size, embed_dim, num_heads, ff_dim, num_blocks, output_dim)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Generate dummy data
    x = np.random.randint(0, vocab_size, size=(1000, seq_len))
    y = np.random.random((1000, output_dim))

    # Train the model
    model.fit(x, y, batch_size=32, epochs=2)

    model.summary()
    pass


def processDataset():
    # Load the proteinNet dataset from tfds
    # https://www.tensorflow.org/datasets/catalog/protein_net
    (data, metadata) = tfds.load('protein_net', split=['train_100', 'validation', 'test'],
                                 as_supervised=True, with_info=True)
    # Supervised keys (See as_supervised doc): ('primary', 'tertiary')
    # [Split('validation'), Split('test'), 'train_30', 'train_50', 'train_70', 'train_90', 'train_95', 'train_100'].
    train_data, val_data, test_data = data
    # print(metadata)
    # sample = tfds.as_dataframe(train_data.take(1), metadata)
    # print(sample)
    # print("Primary (Sequence):")
    # print(sample['primary'].values[0])
    # print("Tertiary (Structure Coordinates):")
    # print(sample['tertiary'].values[0])
    # return

    # Turn the dataset into a pandas dataframe
    train_df = tfds.as_dataframe(train_data, metadata)
    val_df = tfds.as_dataframe(val_data, metadata)
    test_df = tfds.as_dataframe(test_data, metadata)
    print(train_df)
    batch_size = 32

    def process_dataframe(protein_df, batch_size=32):
        max_seq_len = max([len(seq) for seq in protein_df['primary']])
        padded_primary = K.utils.pad_sequences(protein_df['primary'], maxlen=max_seq_len, padding='post')
        max_tertiary_len = max([len(tertiary) for tertiary in protein_df['tertiary']])
        padded_tertiary = np.zeros((len(protein_df['tertiary']), max_tertiary_len, 3))
        for i, tertiary in enumerate(protein_df['tertiary']):
            padded_tertiary[i, :len(tertiary)] = tertiary
        input_data = padded_primary[:, :-1]
        output_data = padded_tertiary[:, 1:]
        # Calculate the number of classes (i.e. the number of unique amino acids)
        nc = len(set([aa for seq in protein_df['primary'] for aa in seq]))
        return tf.data.Dataset.from_tensor_slices((input_data, output_data)).batch(batch_size), max_seq_len, nc

    train_data, tr_max_len, tr_nc = process_dataframe(train_df, batch_size)
    val_data, vl_max_len, vl_nc = process_dataframe(val_df, batch_size)
    test_data, ts_max_len, ts_nc = process_dataframe(test_df, batch_size)

    max_seq_len = max(tr_max_len, vl_max_len, ts_max_len)
    num_classes = max(tr_nc, vl_nc, ts_nc)

    # Create a SeqGAN model
    model = SeqGAN(input_dim=100, embedding_dim=64, hidden_dim=256, max_length=max_seq_len, num_classes=num_classes)
    model.compile(K.optimizers.Adam(lr=0.0002, beta_1=0.5), K.optimizers.Adam(lr=0.0002, beta_1=0.5))
    model.summary()

    # Train the model
    history = model.fit(train_data, epochs=10, batch_size=batch_size)

    model.plot()
    plt.plot(history.history['discriminator_loss'], label='discriminator')
    plt.plot(history.history['generator_loss'], label='generator')
    plt.legend()
    plt.show()

    pass


def trainModel():
    pass


def predictStructure():
    sequence = input("Enter a protein sequence or hit Enter to use the default: ")
    if sequence == "":
        sequence = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPR" \
                   "LASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSS" \
                   "TNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENP" \
                   "NSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
    # plDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100.


if __name__ == "__main__":
    print("Hello world!")
    # demo_transformer()
    processDataset()
    # trainModel()
    # predictStructure()
    print("\nDone!")
