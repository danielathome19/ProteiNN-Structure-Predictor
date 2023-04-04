import time
import datetime
import os
import random
from glob import glob

import keras.utils
import numpy as np
import pandas as pd
from keras.models import Model, load_model
import keras.layers as layers
from keras import optimizers
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Embedding, Conv1D, GlobalMaxPooling1D, Bidirectional, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam


class SeqGAN:
    def __init__(self, input_dim, embedding_dim, hidden_dim, max_length, num_classes):
        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_classes = num_classes

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        # Build the generator model
        input_noise = Input(shape=(self.max_length, self.input_dim))
        x = LSTM(self.hidden_dim, return_sequences=True)(input_noise)
        x = Dense(self.num_classes, activation='softmax')(x)
        generator = Model(input_noise, x)
        generator._name = 'generator'
        return generator

    def build_discriminator(self):
        # Build the discriminator model
        input_seq = Input(shape=(self.max_length, self.num_classes))
        x = Bidirectional(LSTM(self.hidden_dim, return_sequences=True))(input_seq)
        x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
        discriminator = Model(input_seq, x)
        discriminator._name = 'discriminator'
        return discriminator

    def compile(self, generator_optimizer, discriminator_optimizer):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    def fit(self, real_data, epochs, batch_size):
        history = []
        # Training loop for SeqGAN
        for epoch in range(epochs):
            # Train the discriminator
            # Generate fake data using the generator
            noise = tf.random.normal((batch_size, self.max_length, self.input_dim))
            fake_data = self.generator.predict(noise)

            # Train the discriminator on real and fake data
            real_labels = tf.ones((batch_size, self.max_length, 1))
            fake_labels = tf.zeros((batch_size, self.max_length, 1))

            with tf.GradientTape() as tape:
                real_predictions = self.discriminator(real_data)
                fake_predictions = self.discriminator(fake_data)

                real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_predictions)
                fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_predictions)
                total_loss = real_loss + fake_loss

            grads = tape.gradient(total_loss, self.discriminator.trainable_weights)
            self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            # Train the generator
            with tf.GradientTape() as tape:
                fake_data = self.generator(noise)
                fake_predictions = self.discriminator(fake_data)
                generator_loss = tf.keras.losses.binary_crossentropy(real_labels, fake_predictions)

            grads = tape.gradient(generator_loss, self.generator.trainable_weights)
            self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
            history += {'generator_loss': generator_loss.numpy(), 'discriminator_loss': total_loss.numpy()}
            print('Epoch: {}, Generator Loss: {}, Discriminator Loss: {}'
                  .format(epoch, generator_loss.numpy(), total_loss.numpy()))
        return history

    def generate(self, num_samples):
        noise = tf.random.normal((num_samples, self.max_length, self.input_dim))
        generated_samples = self.generator.predict(noise)
        return generated_samples

    def save(self, path):
        self.generator.save(path + 'generator.h5')
        self.discriminator.save(path + 'discriminator.h5')

    def load(self, path):
        self.generator = load_model(path + 'generator.h5')
        self.discriminator = load_model(path + 'discriminator.h5')

    def plot(self, path):
        keras.utils.plot_model(self.generator, to_file=path + 'generator.png', show_shapes=True)
        keras.utils.plot_model(self.discriminator, to_file=path + 'discriminator.png', show_shapes=True)

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()


# class SeqGAN:
#     def __init__(self, seq_length, vocab_size, embedding_dim, gen_hidden_dim, dis_hidden_dim):
#         self.seq_length = seq_length
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.gen_hidden_dim = gen_hidden_dim
#         self.dis_hidden_dim = dis_hidden_dim
#
#         self.generator = self.build_generator()
#         self.discriminator = self.build_discriminator()
#         self.gan = self.build_gan()
#
#     def build_generator(self):
#         noise_input = Input(shape=(self.seq_length,))
#         x = Embedding(self.vocab_size, self.embedding_dim)(noise_input)
#         x = LSTM(self.gen_hidden_dim, return_sequences=True)(x)
#         x = Dense(self.vocab_size, activation='softmax')(x)
#         return Model(noise_input, x)
#
#     def build_discriminator(self):
#         sequence_input = Input(shape=(self.seq_length, self.vocab_size))
#         x = Conv1D(self.dis_hidden_dim, kernel_size=3, padding='same', activation='relu')(sequence_input)
#         x = GlobalMaxPooling1D()(x)
#         x = Dense(1, activation='sigmoid')(x)
#         return Model(sequence_input, x)
#
#     def build_gan(self):
#         self.discriminator.trainable = False
#         noise_input = Input(shape=(self.seq_length,))
#         generated_sequence = self.generator(noise_input)
#         validity = self.discriminator(generated_sequence)
#         return Model(noise_input, validity)
#
#     def compile(self, gen_optimizer, dis_optimizer, loss):
#         self.generator.compile(optimizer=gen_optimizer, loss=loss)
#         self.discriminator.compile(optimizer=dis_optimizer, loss=loss)
#         self.gan.compile(optimizer=gen_optimizer, loss=loss)
#
#     def train(self, real_sequences, epochs, batch_size):
#         real_sequences = tf.one_hot(real_sequences, depth=self.vocab_size)
#
#         valid = np.ones((batch_size, 1))
#         fake = np.zeros((batch_size, 1))
#
#         for epoch in range(epochs):
#             # Train the discriminator
#             noise = np.random.randint(0, self.vocab_size, (batch_size, self.seq_length))
#             generated_sequences = self.generator.predict(noise)
#
#             dis_loss_real = self.discriminator.train_on_batch(real_sequences, valid)
#             dis_loss_fake = self.discriminator.train_on_batch(generated_sequences, fake)
#             dis_loss = 0.5 * np.add(dis_loss_real, dis_loss_fake)
#
#             # Train the generator
#             noise = np.random.randint(0, self.vocab_size, (batch_size, self.seq_length))
#             gen_loss = self.gan.train_on_batch(noise, valid)
#
#             print(f"Epoch {epoch + 1}/{epochs}: [D loss: {dis_loss}] [G loss: {gen_loss}]")
#
#     def generate(self, n):
#         noise = np.random.randint(0, self.vocab_size, (n, self.seq_length))
#         return self.generator.predict(noise)
