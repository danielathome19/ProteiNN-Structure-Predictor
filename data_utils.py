import tensorflow as tf
import numpy as np
from tensorflow import keras as K
import transformer as tfr


def shift_right(sequence):
    return tf.concat(([0], sequence[:-1]), axis=0)


def map_function(seq, struct):
    return (seq, shift_right(struct)), struct


# train_data = train_data.map(map_function)
# val_data = val_data.map(map_function)
# test_data = test_data.map(map_function)
# train_data = train_data.padded_batch(batch_size, padded_shapes=(([None], [None]), [None]))
# val_data = val_data.padded_batch(batch_size, padded_shapes=(([None], [None]), [None]))
# test_data = test_data.padded_batch(batch_size, padded_shapes=(([None], [None]), [None]))


"""
    # print(train_data)
    # return

    # Preprocess sequences
    seq_len = 256
    max_structure_len = 256
    # train_data = train_data.map(preprocess_data)
    # val_data = val_data.map(preprocess_data)
    # test_data = test_data.map(preprocess_data)

    # Parameters
    vocab_size = 21  # 20 amino acids + 1 for padding
    embed_dim = 64
    num_heads = 8
    ff_dim = 128
    num_blocks = 6
    output_dim = max_structure_len * max_structure_len
    output_shape = (max_structure_len, max_structure_len)
    # This should match the size of the preprocessed_structure, e.g., max_structure_len x max_structure_len

    def get_encoder_only_model(*args, **kwargs):
        model = ktr.get_model(*args, **kwargs)
        encoder_input = model.get_layer('Encoder-Input').output
        encoder_output = model.get_layer('Encoder-6-FeedForward-Norm').output
        return K.Model(encoder_input, encoder_output)

    model = get_encoder_only_model(token_num=vocab_size,
                                   embed_dim=embed_dim,
                                   encoder_num=num_blocks,
                                   decoder_num=num_blocks,
                                   head_num=num_heads,
                                   hidden_dim=ff_dim,
                                   attention_activation='gelu',
                                   feed_forward_activation='gelu',
                                   dropout_rate=0.1,
                                   embed_weights=None,
                                   )

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train the model
    tertiary_output = K.layers.TimeDistributed(K.layers.Dense(3, activation='linear'))(model.output)
    final_model = K.Model(model.input, tertiary_output)

    final_model.compile(optimizer='adam', loss='mse')  # Use mean squared error loss for regression tasks
    final_model.summary()

    # Train the model
    history = final_model.fit(train_data, validation_data=val_data, epochs=10)
    # history = model.fit(train_data, validation_data=val_data, epochs=10)
    K.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True)

    # Print the test results
    test_loss, test_acc = model.evaluate(test_data)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    # Predict using a decoder
    decoded = ktr.decode(model, test_data, start_token=0, end_token=0, pad_token=0, max_len=256)
    print(decoded)
    # token_dict_rev = {v: k for k, v in token_dict.items()}
    # for i in range(len(decoded)):
    #     print(' '.join(map(lambda x: token_dict_rev[x], decoded[i][1:-1])))
"""

# region DEPRECATED


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


# def tokenize_sequences(sequence):
#     amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
#     token_dict = {amino_acid: idx + 1 for idx, amino_acid in enumerate(amino_acids)}
#     tokenized_seq = [token_dict.get(aa, 0) for aa in sequence.numpy().decode()]
#     return tokenized_seq


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


# # Create a Transformer model instance
# # model = tfr.Transformer(vocab_size, embed_dim, num_heads, ff_dim, num_blocks, output_dim)
# model = tfr.Transformer(vocab_size, embed_dim, num_heads, ff_dim, num_blocks, output_shape)
#
# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss=tf.keras.losses.MeanSquaredError(),
#               metrics=[tf.keras.metrics.MeanSquaredError()])
#
# # Prepare the datasets for training
# batch_size = 32
# train_data = train_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
# val_data = val_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
#
# # Train the model
# epochs = 10
# history = model.fit(train_data, epochs=epochs, validation_data=val_data)
# model.summary()
#
# # Evaluate the model
# test_data = test_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
# test_results = model.evaluate(test_data)

def parse_structure_string(structure_string):
    structure = []
    lines = structure_string.strip().split("\n")
    for line in lines:
        x, y, z = map(float, line.split())
        structure.append((x, y, z))
    return structure


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


# endregion DEPRECATED
