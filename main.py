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
from tensorflow import keras as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

    padded_sequences = pad_sequences(sequences, maxlen=seq_len, padding='post', truncating='post')

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


def tokenizeSequences(sequences):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    token_dict = {amino_acid: idx + 1 for idx, amino_acid in enumerate(amino_acids)}

    tokenized_sequences = []
    for seq in sequences:
        tokenized_seq = [token_dict.get(aa, 0) for aa in seq]
        tokenized_sequences.append(tokenized_seq)
    return tokenized_sequences


def preprocessSequences(data):
    sequences = []
    for example in data:
        sequence = example[0]['primary'].numpy().decode()
        sequences.append(sequence)
    return sequences


def preprocessStructures(data):
    # Add function to preprocess the structures (e.g., to distance maps)
    
    return data


def processDataset():
    # Load the proteinNet dataset from tfds
    # https://www.tensorflow.org/datasets/catalog/protein_net
    (data, metadata) = tfds.load('protein_net', split=['train_100', 'validation', 'test'],
                                 as_supervised=True, with_info=True)
    # [Split('validation'), Split('test'), 'train_30', 'train_50', 'train_70', 'train_90', 'train_95', 'train_100'].
    train_data, val_data, test_data = data  # 3 tuples of (data, ('primary', 'tertiary'))
    print(metadata)
    print(train_data)
    return

    # Preprocess sequences
    seq_len = 256
    train_sequences = preprocessSequences(train_data)
    val_sequences = preprocessSequences(val_data)
    test_sequences = preprocessSequences(test_data)

    tokenized_train_sequences = tokenizeSequences(train_sequences)
    tokenized_val_sequences = tokenizeSequences(val_sequences)
    tokenized_test_sequences = tokenizeSequences(test_sequences)

    padded_train_sequences = pad_sequences(tokenized_train_sequences, maxlen=seq_len, padding='post', truncating='post')
    padded_val_sequences = pad_sequences(tokenized_val_sequences, maxlen=seq_len, padding='post', truncating='post')
    padded_test_sequences = pad_sequences(tokenized_test_sequences, maxlen=seq_len, padding='post', truncating='post')

    # Preprocess structures
    max_structure_len = 256
    train_structures = preprocessStructures(train_data)
    val_structures = preprocessStructures(val_data)
    test_structures = preprocessStructures(test_data)
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
