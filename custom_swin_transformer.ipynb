import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, LayerNormalization
from tensorflow.keras.layers import Input, Reshape, Permute, Dense, Multiply, Dropout
from tensorflow.keras.models import Model


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, output_filters):
        super(PatchMerging, self).__init__()
        self.output_filters = output_filters

    def build(self, input_shape):
        self.conv = Conv2D(filters=self.output_filters, kernel_size=3, strides=2, padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.proj = Conv2D(filters=self.embed_dim, kernel_size=3, strides=2, padding='same')
        self.rescale = tf.keras.layers.Rescaling(scale=1.0 / tf.math.sqrt(float(self.embed_dim)))

    def call(self, inputs):
        x = self.proj(inputs)
        x = self.rescale(x)
        B, H, W, C = x.shape
        x = tf.reshape(x, (B, H * W, C))
        return x


class ShiftAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(ShiftAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.scale = tf.keras.layers.Scaling(scale=self.embed_dim ** -0.5)
        self.rearrange = Rearrange('b h w c -> b c h w')
        self.conv1 = Conv2D(filters=self.embed_dim * 2, kernel_size=1, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(filters=self.embed_dim, kernel_size=3, strides=1, padding='same')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')

    def call(self, inputs):
        x = self.scale(inputs)
        x = self.rearrange(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        B, C, H, W = x.shape
        x = tf.pad(x, paddings=[(0, 0), (0, 0), (1, 1), (1, 1)], mode='constant')
        x = tf.reshape(x, (B, C, -1))
        x = self.rearrange(x, pattern='b c (h w) -> b c h w', h=H+2, w=W+2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x


class MLP(tf.keras.layers.Layer):
    def __init__(self, embed_dim, mlp_dim):
        super(MLP, self).__init__()
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.fc1 = Dense(self.mlp_dim)
        self.act1 = Activation('gelu')
        self.fc2 = Dense(self.embed_dim)
        self.act2 = Activation('gelu')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(SwinTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.norm1 = LayerNormalization(epsilon=1e-5)
        self.attn = ShiftAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.norm2 = LayerNormalization(epsilon=1e-5)
        self.mlp = MLP(embed_dim=self.embed_dim, mlp_dim=self.mlp_dim)

    def call(self, inputs):
        x = self.norm1(inputs)
        x = self.attn(x)
        x = x + inputs

        y = self.norm2(x)
        y = self.mlp(y)
        return x + y


class SwinTransformer(tf.keras.Model):
    def __init__(self, input_shape, patch_size, num_classes, num_layers, embed_dim, num_heads, mlp_dim):
        super(SwinTransformer, self).__init__()

        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.patch_embed = PatchEmbedding(num_patches=(input_shape[0] // patch_size) * (input_shape[1] // patch_size),
                                          embed_dim=embed_dim)

        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(SwinTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim))

        self.norm = LayerNormalization(epsilon=1e-5)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.patch_embed(inputs)
        for i in range(self.num_layers):
            x = self.blocks[i](x)
        x = self.norm(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


# Example usage
input_shape = (256, 256, 3)
patch_size = 4
num_classes = 10
num_layers = 6
embed_dim = 96
num_heads = 4
mlp_dim = 384

model = SwinTransformer(input_shape, patch_size, num_classes, num_layers, embed_dim, num_heads, mlp_dim)
model.build(input_shape=(None,) + input_shape)
model.summary()
