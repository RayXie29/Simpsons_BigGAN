import tensorflow as tf
from regularizer import Orthogonal_Regularization_Relax
from tensorflow.keras import initializers, regularizers

class Conv2D_SN(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):

        super(Conv2D_SN, self).__init__(trainable=trainable, name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

    def build(self, input_shape):

        input_shape = tf.TensorShape(input_shape)
        input_channel = input_shape[-1]
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel = self.add_weight(
                        name='kernel',
                        shape=kernel_shape,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        trainable=True)

        if self.use_bias:

            self.bias = self.add_weight(
                        name='bias',
                        shape=(self.filters,),
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        trainable=True)
        else:
            self.bias = None

        self.build_input_shape = input_shape
        self.input_channel = input_channel
        self.kernel_shape = kernel_shape

        #For spectral normalization
        self.u = self.add_weight(
                        name='u',
                        shape=[1, kernel_shape[-1]],
                        initializer=tf.initializers.TruncatedNormal(),
                        regularizer=None,
                        trainable=False)

        self.built = True

    def call(self, inputs):

        if isinstance(inputs, list):
            raise ValueError('`Conv2D_SN` Layer only allow 1 input with 4 dimension(b, h, w, c)')
        if self._check_shape(inputs) :
            raise ValueError('`Conv2D_SN` Layer - Shape of inputs in call is different from shape in build')

        is_training = tf.keras.backend.learning_phase()
        with tf.control_dependencies([self._spectral_norm(is_training)]):
            outputs = tf.nn.conv2d(input=inputs,
                                   filters=self.kernel,
                                   strides=(1, self.strides[0], self.strides[1], 1),
                                   padding=self.padding)

            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

            return outputs

    def _spectral_norm(self, is_training):

        u_hat = self.u
        w_mat = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
        '''
        Power iteration for only 1 iteration
        '''
        v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, w_mat, transpose_b=True)) # 1, -1
        u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_mat)) #1, filters

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.squeeze(tf.matmul(tf.matmul(v_hat, w_mat), u_hat, transpose_b=True))
        with tf.control_dependencies([tf.cond(is_training, true_fn=lambda:self.u.assign(u_hat), false_fn=lambda:self.u.assign(self.u))]):
            w_mat = w_mat / sigma
            w_mat = tf.reshape(w_mat, self.kernel_shape)
            return self.kernel.assign(w_mat)


    def _check_shape(self, inputs):

        call_input_shape = inputs.get_shape()
        for axis in range(1, len(call_input_shape)):
            if (call_input_shape[axis] is not None
                and self.build_input_shape[axis] is not None
                and call_input_shape[axis] != self.build_input_shape[axis]):

                return True

        return False

class Conv2DTranspose_SN(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 output_padding=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):

        super(Conv2DTranspose_SN, self).__init__(trainable=trainable, name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.upper()
        self.output_padding = output_padding if isinstance(output_padding, tuple) or isinstance(output_padding, list) or output_padding is None else (output_padding, output_padding)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        assert self.padding in {'SAME', 'VALID'}, 'padding only support `same` or `valid` two modes'

        if self.output_padding:
            for out_pad, stride in zip(self.output_padding, self.strides):
                if out_pad >= strides:
                    raise ValueError('strides `{}` must bigger than output_pa'
                                        'dding `{}`'.format(str(self.strides),str(self.output_paddding)))

    def build(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError('`Conv2DTranspose_SN` Layer only allow 1 input with 4 dimension(b, h, w, c)')

        input_channel = input_shape[-1]
        kernel_shape = self.kernel_size + (self.filters, input_channel)

        self.kernel = self.add_weight(
                        name='kernel',
                        shape=kernel_shape,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(
                            name='bias',
                            shape=(self.filters,),
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            trainable=True)
        else:
            self.bias=None

        self.build_input_shape = input_shape
        self.input_channel = input_channel
        self.kernel_shape = kernel_shape
        #For spectral normalization
        self.u = self.add_weight(
                        name='u',
                        shape=(1, kernel_shape[-1]),
                        dtype=tf.float32,
                        initializer=tf.initializers.TruncatedNormal(),
                        regularizer=None,
                        trainable=False)
        self.built = True

    def call(self, inputs):

        if isinstance(inputs, list):
            raise ValueError('`Conv2DTranspose_SN` Layer only allow 1 input with 4 dimension(b, h, w, c)')

        if self._check_shape(inputs):
            raise ValueError('`Conv2DTranspose_SN` Layer, shape of input in call is different from shape in build')

        is_training = tf.keras.backend.learning_phase()
        with tf.control_dependencies([self._spectral_norm(is_training)]):

            output_shape = self._compute_output_size(inputs)

            outputs = tf.nn.conv2d_transpose(input=inputs,
                                             filters=self.kernel,
                                             output_shape=output_shape,
                                             strides=[1, self.strides[0], self.strides[1], 1],
                                             padding=self.padding)
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

            return outputs

    def _compute_output_size(self, inputs):
        call_input_shape = tf.shape(inputs)
        b = call_input_shape[0]
        _, h, w, c = inputs.get_shape()

        if self.output_padding is None:

            if self.padding == 'SAME':
                out_h = h * self.strides[0]
                out_w = w * self.strides[1]
            else :
                out_h = (h-1) * self.strides[0] + self.kernel_size[0]
                out_w = (w-1) * self.strides[1] + self.kernel_size[1]
        else:

            if self.padding == 'SAME':
                h_pad = self.kernel_size[0] // 2
                w_pad = self.kernel_size[1] // 2
            else:
                h_pad, w_pad = 0, 0

            out_h = (h-1) * self.strides[0] + self.kernel_size[0] - 2*h_pad + self.output_padding[0]
            out_w = (w-1) * self.strides[1] + self.kernel_size[1] - 2*w_pad + self.output_padding[1]

        output_shape = (b, out_h, out_w, self.filters)
        output_shape = tf.stack(output_shape)
        return output_shape

    def _check_shape(self, inputs):

        call_input_shape = inputs.get_shape()
        for axis in range(1, len(call_input_shape)):
            if (call_input_shape[axis] is not None
                and self.build_input_shape[axis] is not None
                and call_input_shape[axis] != self.build_input_shape[axis]):

                return True

        return False

    def _spectral_norm(self, is_training):

        u_hat = self.u
        w_mat = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
        '''
        Power iteration for only 1 iteration
        '''
        v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, w_mat, transpose_b=True)) # 1, -1
        u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_mat)) #1, filters

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.squeeze(tf.matmul(tf.matmul(v_hat, w_mat), u_hat, transpose_b=True))
        with tf.control_dependencies([tf.cond(is_training, true_fn=lambda:self.u.assign(u_hat), false_fn=lambda:self.u.assign(self.u))]):
            w_mat = w_mat / sigma
            w_mat = tf.reshape(w_mat, self.kernel_shape)
            return self.kernel.assign(w_mat)


class Dense_SN(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer=None,
                 bias_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):

        super(Dense_SN, self).__init__(name=name, trainable=trainable, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

    def build(self, input_shape):

        if isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('`Dense_SN` layer only allow 1 input with 2 dimension(b, c)')

        input_channel = input_shape[-1]
        kernel_shape = (input_channel, self.units)

        self.kernel = self.add_weight(
                            name='kernel',
                            shape=kernel_shape,
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(
                            name='bias',
                            shape=(self.units,),
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            trainable=True)
        else:
            self.bias = None

        self.input_channel = input_channel
        self.kernel_shape = kernel_shape
        self.build_input_shape = input_shape

        #For spectral normalization
        self.u = self.add_weight(
                        name='u',
                        shape=(1, self.units),
                        dtype=tf.float32,
                        initializer=tf.initializers.TruncatedNormal(),
                        regularizer=None,
                        trainable=False)
        self.built = True

    def call(self, inputs):
        input_shape = inputs.get_shape()
        if isinstance(inputs, list) or len(input_shape) != 2:
            raise ValueError('`Dense_SN` layer only allow 1 input with 2 dimension(b, c)')

        if self._check_shape(inputs):
            raise ValueError('`Dense_SN` layer, shape of input in call is different from shape of input in build')
        is_training = tf.keras.backend.learning_phase()
        with tf.control_dependencies([self._spectral_norm(is_training)]):
            if tf.keras.backend.is_sparse(inputs):
                outputs = tf.sparse_tensor_dense_matmul(inputs, self.kernel)
            else:
                outputs = tf.matmul(inputs, self.kernel)

            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, self.bias)

            return outputs

    def _check_shape(self, inputs):

        call_input_shape = inputs.get_shape()
        for axis in range(1, len(call_input_shape)):
            if (call_input_shape[axis] is not None
                and self.build_input_shape[axis] is not None
                and call_input_shape[axis] != self.build_input_shape[axis]):

                return True

        return False

    def _spectral_norm(self, is_training):

        u_hat = self.u
        w_mat = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
        '''
        Power iteration for only 1 iteration
        '''
        v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, w_mat, transpose_b=True)) # 1, -1
        u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_mat)) #1, filters

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.squeeze(tf.matmul(tf.matmul(v_hat, w_mat), u_hat, transpose_b=True))
        with tf.control_dependencies([tf.cond(is_training, true_fn=lambda:self.u.assign(u_hat), false_fn=lambda:self.u.assign(self.u))]):
            w_mat = w_mat / sigma
            w_mat = tf.reshape(w_mat, self.kernel_shape)
            return self.kernel.assign(w_mat)

class ConditionalBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(ConditionalBatchNormalization, self).__init__(name=name, **kwargs)
    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Conditional_BatchNormalization` layer should be called on a list of 2 inputs')

        #shape of input feature(tensor)
        shape1 = input_shape[0]
        #shape of conditional feature(categorical information/noise for generator...)
        shape2 = input_shape[1]

        if shape1 is None or shape2 is None:
            raise ValueError('None in inputs is not allowed, input_1 shape : {}, input_2 shape : {}'.format(shape1, shape2))


        feature_channels = shape1[-1]
        info_units=shape2[-1]

        self.moving_mean = self.add_weight(
                            name='moving_mean',
                            shape=[feature_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0),
                            trainable=False,
                            )

        self.moving_var = self.add_weight(
                            name='moving_variance',
                            shape=[feature_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(1.0),
                            trainable=False,
                            )

        self.gamma_kernel = self.add_weight(
                                name='gamma_kernel',
                                shape=[info_units, feature_channels],
                                dtype=tf.float32,
                                initializer=tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
                                regularizer=Orthogonal_Regularization_Relax(1e-4)  ,
                                trainable=True
                                )

        self.beta_kernel = self.add_weight(
                                name='beta',
                                shape=[info_units, feature_channels],
                                dtype=tf.float32,
                                initializer=tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
                                regularizer=Orthogonal_Regularization_Relax(1e-4)  ,
                                trainable=True
                                )
        self.built=True

    def call(self, inputs, training=None):

        if training is None:
            training = tf.keras.backend.learning_phase()

        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('A `Conditional_BatchNormalization` layer should be called on a list of 2 inputs')

        decay = 0.9
        epsilon = 1e-05

        input_feature = inputs[0]
        z = inputs[1]

        gamma = tf.matmul(z, self.gamma_kernel)
        beta = tf.matmul(z, self.beta_kernel)

        c = gamma.get_shape().as_list()[-1]

        gamma = tf.reshape(gamma, [-1, 1, 1, c])
        beta = tf.reshape(beta, [-1, 1, 1, c])

        gamma = tf.cast(gamma, input_feature.dtype)
        beta = tf.cast(beta, input_feature.dtype)

        batch_mean, batch_var = tf.nn.moments(input_feature, axes=[0,1,2])

        if training:
            self.moving_mean.assign(self.moving_mean * decay + batch_mean * (1-decay))
            self.moving_var.assign(self.moving_var * decay + batch_var * (1-decay))
            return tf.nn.batch_normalization(input_feature, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            mean = tf.convert_to_tensor(self.moving_mean)
            var = tf.convert_to_tensor(self.moving_var)
            return tf.nn.batch_normalization(input_feature, mean, var, beta, gamma, epsilon)



class SigmaLayer(tf.keras.layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(SigmaLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('`SigmaLayer` only for self-attention block with 4 dimension feature map input(b, h, w, c)')

        self.sigma = self.add_weight(
                        name='sigma',
                        shape=[],
                        initializer=tf.constant_initializer(0.0),
                        trainable=True
        )

        self.built=True

    def call(self, inputs):

        if isinstance(inputs, list):
            raise ValueError('`SigmaLayer` only allowed 1 input with 4 dimension feature map(b, h, w, c)')

        return self.sigma*inputs

def Layer_matmul_transposed_b(args):
    fx_flatten, gx_flatten = args
    return tf.matmul(fx_flatten, gx_flatten, transpose_b=True)

def Layer_matmul(args):
    attn, hx_flatten = args
    return tf.matmul(attn, hx_flatten)

def GlobalSumPooling2D(self,args):
    x = args
    return tf.reduce_sum(x, axis=[1,2])
