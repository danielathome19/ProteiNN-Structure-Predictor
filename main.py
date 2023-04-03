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


# region DEPRECATED


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


# endregion DEPRECATED


def parse_structure_string(structure_string):
    structure = []
    lines = structure_string.strip().split("\n")
    for line in lines:
        x, y, z = map(float, line.split())
        structure.append((x, y, z))
    return structure


# def calculate_distance_map(structure, max_structure_len):
#     num_residues = len(structure)
#     distance_map = np.zeros((max_structure_len, max_structure_len), dtype=np.float32)
#     for i in range(num_residues):
#         for j in range(i + 1, num_residues):
#             distance = np.sqrt(np.sum((np.array(structure[i]) - np.array(structure[j]))**2))
#             distance_map[i, j] = distance
#             distance_map[j, i] = distance
#     return distance_map
#
#
# def preprocess_structures(data, max_structure_len):
#     # structure_string = data.numpy().decode()
#     if isinstance(data, np.ndarray):
#         structure_string = data.tobytes().decode()
#     else:
#         structure_string = data.numpy().tobytes().decode()
#     structure = parse_structure_string(structure_string)
#     distance_map = calculate_distance_map(structure, max_structure_len)
#     return distance_map


def calculate_distance_map(structure, max_structure_len):
    num_residues = len(structure)
    distance_map = np.zeros((max_structure_len, max_structure_len), dtype=np.float32)
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            distance = np.sqrt(np.sum((structure[i] - structure[j])**2))
            distance_map[i, j] = distance
            distance_map[j, i] = distance
    return distance_map


def preprocess_structures(data, max_structure_len):
    structure = data.numpy()
    distance_map = calculate_distance_map(structure, max_structure_len)
    return distance_map


# def tokenize_sequences(sequence):
#     amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
#     token_dict = {amino_acid: idx + 1 for idx, amino_acid in enumerate(amino_acids)}
#     tokenized_seq = [token_dict.get(aa, 0) for aa in sequence.numpy().decode()]
#     return tokenized_seq


def tokenize_sequences(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    token_dict = {amino_acid: idx + 1 for idx, amino_acid in enumerate(amino_acids)}
    seq_string = sequence.numpy().tobytes().decode(errors='replace')
    tokenized_seq = [token_dict.get(aa, 0) for aa in seq_string]
    return tokenized_seq


def tf_tokenize_sequences(sequence):
    tokenized_seq = tf.py_function(tokenize_sequences, [sequence], tf.int32)
    tokenized_seq.set_shape([None])
    return tokenized_seq


def preprocess_data(_, element, seq_len=256, max_structure_len=256):
    sequence = element[0]
    structure = element[1]
    tokenized_sequence = tf_tokenize_sequences(sequence)
    padding_length = seq_len - tf.shape(tokenized_sequence)[0]
    padding = tf.maximum(padding_length, 0)
    padded_sequence = tf.pad(tokenized_sequence, paddings=[[0, padding]], mode='CONSTANT', constant_values=0)[:seq_len]
    preprocessed_structure = tf.py_function(preprocess_structures, [structure, max_structure_len],
                                            tf.float32)
    return padded_sequence, preprocessed_structure


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

    # Preprocess sequences
    seq_len = 256
    max_structure_len = 256
    train_data = train_data.map(preprocess_data)
    val_data = val_data.map(preprocess_data)
    test_data = test_data.map(preprocess_data)

    # Parameters
    vocab_size = 21  # 20 amino acids + 1 for padding
    embed_dim = 64
    num_heads = 8
    ff_dim = 128
    num_blocks = 6
    output_dim = max_structure_len * max_structure_len
    # This should match the size of the preprocessed_structure, e.g., max_structure_len x max_structure_len

    # Create a Transformer model instance
    model = tfr.Transformer(vocab_size, embed_dim, num_heads, ff_dim, num_blocks, output_dim)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    # Prepare the datasets for training
    batch_size = 32
    train_data = train_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Train the model
    epochs = 10
    history = model.fit(train_data, epochs=epochs, validation_data=val_data)
    model.summary()

    # Evaluate the model
    test_data = test_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_results = model.evaluate(test_data)

    # Print the test results
    print("Test results - Loss: {:.4f}, Mean Squared Error: {:.4f}".format(test_results[0], test_results[1]))

    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
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
