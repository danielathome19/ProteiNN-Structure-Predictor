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


class ProteinTransformer:
    def __init__(self, config, learning_rate=0.001, opt_epsilon=1e-6):
        self.config = config
        self.lr = learning_rate
        self.eps = opt_epsilon
        self.model = self.build_model()
        self.weightpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.h5")

    def build_model(self):
        inputs = keras.Input(shape=(None,), dtype=tf.int32)
        x = layers.Embedding(self.config["input_vocab_size"], self.config["d_model"])(inputs)

        for _ in range(self.config["num_layers"]):
            attn_layer = layers.MultiHeadAttention(num_heads=self.config["num_heads"],
                                                   key_dim=self.config["d_model"] // self.config["num_heads"])
            attn_output = attn_layer(x, x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization(epsilon=self.eps)(x)

            ffn_output = layers.Dense(self.config["dff"], activation='relu')(x)
            ffn_output = layers.Dense(self.config["d_model"])(ffn_output)
            x = layers.Add()([x, ffn_output])
            x = layers.LayerNormalization(epsilon=self.eps)(x)

        x = layers.TimeDistributed(layers.Dense(self.config["output_vocab_size"]))(x)

        optimizer = keras.optimizers.Adam(learning_rate=self.lr, epsilon=self.eps)
        model = keras.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def train(self, dataset, epochs, batch_size, val_data=None):
        dataset = dataset.shuffle(buffer_size=len(dataset)).batch(batch_size)
        checkpoint = keras.callbacks.ModelCheckpoint(self.weightpath, save_weights_only=True,
                                                     save_best_only=True, monitor='mae', verbose=1)
        if val_data is not None:
            val_data = val_data.batch(batch_size)
            history = self.model.fit(dataset, epochs=epochs, callbacks=[checkpoint], validation_data=val_data)
        else:
            history = self.model.fit(dataset, epochs=epochs, callbacks=[checkpoint])
        return history

    def predict(self, sequence):
        sequence = np.array([sequence])
        predicted_structure = self.model.predict(sequence)
        return predicted_structure

    def summary(self):
        self.model.summary()

    def plot_model(self):
        # Get path of current python file
        keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True, expand_nested=True, dpi=300,
                               to_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.png"))

    def save_pdb(self, predicted_structure, output_file):
        pdb_structure = Model.Model(0)

        for i, (x, y, z) in enumerate(predicted_structure[0]):
            chain = Chain.Chain("A")
            res_id = (" ", i + 1, " ")
            res = Residue.Residue(res_id, "ALA", " ")
            atom = Atom.Atom("CA", np.array([x, y, z]), 1.0, 1.0, " ", "CA", 1)
            res.add(atom)
            chain.add(res)
            pdb_structure.add(chain)

        pdb_io = PDBIO()
        pdb_io.set_structure(pdb_structure)
        pdb_io.save(output_file)

    @staticmethod
    def preprocess_data(data):
        # Flatten the dictionaries and convert the nested arrays into tensors
        input_sequences = []
        output_sequences = []
        for _, row in data.iterrows():
            input_sequences.append(row['primary'])
            output_sequences.append(row['tertiary'])

        # Pad the input sequences
        input_sequences_padded = pad_sequences(input_sequences, padding='post', dtype='int64')
        input_tensors = tf.constant(input_sequences_padded, dtype=tf.int64)

        # Pad the output sequences
        max_length = max([len(seq) for seq in output_sequences])
        output_sequences_padded = [np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant') for seq in
                                   output_sequences]
        output_tensors = tf.constant(output_sequences_padded, dtype=tf.float32)

        # Input tensor shape: (3455, 1520)
        # Output tensor shape: (3455, 4560, 3)
        # Pad the input sequences with 0s to match the output sequence length
        input_tensors = tf.pad(input_tensors, [[0, 0], [0, max_length - input_tensors.shape[1]]], constant_values=0)

        return input_tensors, output_tensors


def load_and_preprocess_data():
    (data, metadata) = tfds.load('protein_net', split=['train_100', 'validation', 'test'],
                                 as_supervised=True, with_info=True)
    train_data, val_data, test_data = data

    # Cut datasets down to save memory
    train_data = train_data.take(len(list(train_data)) // 20)
    val_data = val_data.take(len(list(val_data)) // 20)
    test_data = test_data.take(len(list(test_data)) // 20)

    train_df = tfds.as_dataframe(train_data, metadata)
    val_df = tfds.as_dataframe(val_data, metadata)
    test_df = tfds.as_dataframe(test_data, metadata)
    train_input_tensors, train_output_tensors = ProteinTransformer.preprocess_data(train_df)
    # print("Input tensor shape:", train_input_tensors.shape)
    # print("Output tensor shape:", train_output_tensors.shape)
    val_input_tensors, val_output_tensors = ProteinTransformer.preprocess_data(val_df)
    test_input_tensors, test_output_tensors = ProteinTransformer.preprocess_data(test_df)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input_tensors, train_output_tensors))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_input_tensors, val_output_tensors))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_input_tensors, test_output_tensors))
    return train_dataset, val_dataset, test_dataset


def encode_sequence(sequence):
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
    return [aa_to_int[aa] for aa in sequence]


def main():
    config = {
        "num_layers": 3,  # 6 (try 3-5)
        "d_model": 128,  # 256 (try 128 or 512)
        "num_heads": 4,  # 8 (try 4 or 16)
        "dff": 256,  # 512 (try 256 or 1024)
        "input_vocab_size": 21,
        "output_vocab_size": 3,
        "max_position_encoding": 10000,
        "dropout_rate": 0.1,
    }
    train_data, val_data, test_data = load_and_preprocess_data()

    # Initialize ProteinTransformer
    model = ProteinTransformer(config, learning_rate=0.000001)
    model.summary()
    # model.plot_model()

    # Train the model
    history = model.train(train_data, epochs=100, batch_size=4, val_data=val_data)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Predict the protein structure
    protein_sequence = [2, 2, 5, 16, 5, 13, 8, 13, 7, 18, 14, 7, 3, 5, 13, 12, 0]
    predicted_structure = model.predict(protein_sequence)

    # Save the predicted structure as a PDB file
    model.save_pdb(predicted_structure, "predicted_structure.pdb")


if __name__ == "__main__":
    main()
