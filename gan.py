from layers import *
import tensorflow as tf
import tensorflow.keras.layers as KL

w_initializer = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
w_regularizer = Orthogonal_Regularization_Relax(1e-4)


_DEFAULT_CONV_PARAMS = {
    'kernel_initializer' : w_initializer,
    'use_bias' : False,
    'padding' : 'same'
}

_DEFAULT_DENSE_PARAMS = {
    'kernel_initializer' : w_initializer,
    'use_bias' : False,
}

class BigGAN(object):

    def __init__(self,
                 noise_dim=(128,),
                 image_dim=(128,128,3),
                 channel_width_multiplier=32,
                 Generator_init_size=(8,8)):

        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.ch = channel_width_multiplier
        self.G_init_size = Generator_init_size

    def _self_attention_block(self, channel_scale_ratio=8, regularizer=None):

        def f(input_x):

          b, h, w, c = input_x.get_shape().as_list()

          scaled_channel = c // channel_scale_ratio
          fx = Conv2D_SN(filters=scaled_channel, kernel_size=1, strides=1, kernel_regularizer=regularizer, **_DEFAULT_CONV_PARAMS)(input_x)
          fx = KL.MaxPooling2D(pool_size=(2,2))(fx)

          gx = Conv2D_SN(filters=scaled_channel, kernel_size=1, strides=1, kernel_regularizer=regularizer, **_DEFAULT_CONV_PARAMS)(input_x)

          hx = Conv2D_SN(filters=c//2, kernel_size=1, strides=1, kernel_regularizer=regularizer, **_DEFAULT_CONV_PARAMS)(input_x)
          hx = KL.MaxPooling2D(pool_size=(2,2))(hx)

          fx_flatten = KL.Reshape((-1, scaled_channel))(fx)
          gx_flatten = KL.Reshape((-1, scaled_channel))(gx)
          hx_flatten = KL.Reshape((-1, c//2))(hx)

          attn = KL.Lambda(Layer_matmul_transposed_b)([gx_flatten, fx_flatten])
          attn = KL.Activation('softmax', name='attention_map')(attn)

          attn_h = KL.Lambda(Layer_matmul)([attn, hx_flatten])
          attn_h = KL.Reshape((h, w, c//2))(attn_h)
          attn_h = Conv2D_SN(filters=c, kernel_size=1, strides=1, kernel_regularizer=regularizer, **_DEFAULT_CONV_PARAMS)(attn_h)

          y = SigmaLayer(name='sigma_layer')(attn_h)
          y = KL.Add()([y, input_x])

          return y

        return f


    def ResBlock_up(self, filters, kernel_size=3, strides=2):
        def f(input_x):

            x = KL.BatchNormalization()(input_x)
            x = KL.Activation('relu')(x)
            x = Conv2DTranspose_SN(filters=filters, kernel_size=kernel_size, strides=strides, kernel_regularizer = w_regularizer, **_DEFAULT_CONV_PARAMS)(x)
            x = KL.BatchNormalization()(x)
            x = KL.Activation('relu')(x)
            x = Conv2DTranspose_SN(filters=filters, kernel_size=kernel_size, strides=1, kernel_regularizer = w_regularizer, **_DEFAULT_CONV_PARAMS)(x)

            x_skip = Conv2DTranspose_SN(filters=filters, kernel_size=1, strides=strides, kernel_regularizer = w_regularizer, **_DEFAULT_CONV_PARAMS)(input_x)

            return KL.Add()([x, x_skip])
        return f

    def ResBlock_down(self, filters, kernel_size=3, strides=2):
        def f(input_x):

            x = KL.BatchNormalization()(input_x)
            x = KL.LeakyReLU(alpha=0.2)(x)
            x = Conv2D_SN(filters=filters, kernel_size=kernel_size, strides=strides, **_DEFAULT_CONV_PARAMS)(x)
            x = KL.BatchNormalization()(x)
            x = KL.LeakyReLU(alpha=0.2)(x)
            x = Conv2D_SN(filters=filters, kernel_size=kernel_size, strides=1, **_DEFAULT_CONV_PARAMS)(x)

            x_skip = Conv2D_SN(filters=filters, kernel_size=1, strides=strides, **_DEFAULT_CONV_PARAMS)(input_x)

            return KL.Add()([x, x_skip])
        return f

    def GeneratorNetwork(self):

        noise = KL.Input(shape=self.noise_dim, name='noise_input')

        x = Dense_SN(units= self.G_init_size[0]*self.G_init_size[1]*self.ch*16, kernel_regularizer=w_regularizer, **_DEFAULT_DENSE_PARAMS)(noise)
        x = KL.Reshape((self.G_init_size[0], self.G_init_size[1], self.ch*16))(x)
        x = self.ResBlock_up(filters=self.ch*8, kernel_size=3)(x)
        x = self.ResBlock_up(filters=self.ch*4, kernel_size=3)(x)
        x = self._self_attention_block(channel_scale_ratio=8, regularizer=w_regularizer)(x)
        x = self.ResBlock_up(filters=self.ch*2, kernel_size=3)(x)

        x = KL.BatchNormalization()(x)
        x = KL.Activation('relu')(x)
        x = Conv2D_SN(filters=self.image_dim[-1], kernel_size=3, strides=1, kernel_regularizer=w_regularizer, **_DEFAULT_DENSE_PARAMS)(x)

        output_layer = KL.Activation('tanh', name='fakeimage')(x)

        return tf.keras.models.Model(inputs=[noise,], outputs=[output_layer], name='Generator')

    def DiscriminatorNetwork(self):

        input_layer = KL.Input(shape=self.image_dim, name='image_input')
        x = self.ResBlock_down(filters=self.ch*2, kernel_size=3)(input_layer)
        x = self._self_attention_block(channel_scale_ratio=8, regularizer=None)(x)
        x = self.ResBlock_down(filters=self.ch*4, kernel_size=3)(x)
        x = self.ResBlock_down(filters=self.ch*8, kernel_size=3)(x)
        x = self.ResBlock_down(filters=self.ch*16, kernel_size=3)(x)
        x = KL.LeakyReLU(alpha=0.2)(x)
        x = KL.Lambda(GlobalSumPooling2D, name='Global_sum_pooling')(x)
        output_layer = Dense_SN(units=1, name='validity')(x)

        return tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer], name='Discriminator')
