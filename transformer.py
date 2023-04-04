import tensorflow as tf
from tensorflow import keras as K
from keras import layers


class TransformerLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self.multihead_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(ff_dim, activation="relu")
        self.dense2 = layers.Dense(embed_dim)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.multihead_attention(inputs, inputs)
        x = self.layer_norm1(inputs + attn_output)
        ff_output = self.dense1(x)
        ff_output = self.dense2(ff_output)
        outputs = self.layer_norm2(x + ff_output)
        return outputs


class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, output_dim, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.transformer_layers = [TransformerLayer(embed_dim, num_heads, ff_dim) for _ in range(num_blocks)]
        self.global_avg_pooling = layers.GlobalAveragePooling1D()
        self.dense_output = layers.Dense(output_dim, activation="linear")

    def call(self, inputs):
        x = self.embedding(inputs)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = self.global_avg_pooling(x)
        outputs = self.dense_output(x)
        return outputs
