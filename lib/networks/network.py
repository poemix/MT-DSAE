# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : network.py
# @Software : PyCharm


import tensorflow as tf


def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, trainable=True, reuse=False):
        self.inputs = []
        self.layers = dict(inputs)
        self.params = dict()
        self.trainable = trainable
        self.reuse = reuse
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, sess, saver):
        raise NotImplementedError('Must be subclassed.')

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(self.layers.keys())
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(self.layers.keys())
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def _get_activation(self, activation='relu'):
        if activation == 'relu':
            act_flag = True
            act_func = tf.nn.relu
        elif activation == 'tanh':
            act_flag = True
            act_func = tf.nn.tanh
        elif activation == 'sigmoid':
            act_flag = True
            act_func = tf.nn.sigmoid
        elif activation == 'leaky_relu':
            act_flag = True
            act_func = tf.nn.leaky_relu
        elif activation == 'elu':
            act_flag = True
            act_func = tf.nn.elu
        elif activation == 'swish':
            act_flag = True
            act_func = self._swish
        elif activation == 'softmax':
            act_flag = True
            act_func = tf.nn.softmax
        elif activation is False:
            act_flag = False
            act_func = None
        else:
            raise ValueError('nonsupport activation function.')
        return act_flag, act_func

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def _get_shape(self, input):
        return tf.shape(input)

    def _swish(self, input, name=None):
        return tf.multiply(input, tf.nn.sigmoid(input), name=name)

    def _reshape(self, input, shape, name=None):
        return tf.reshape(input, shape=shape, name=name)

    def _pool1d(self, input, k_s, s, name=None, p_t='MAX', padding='SAME'):
        return tf.nn.pool(input, window_shape=[k_s], strides=[s], pooling_type=p_t, name=name, padding=padding)

    def _global_average_pool1d(self, input, keepdims, name=None):
        return tf.reduce_mean(input, axis=[1], keepdims=keepdims, name=name)

    def _softmax(self, input, name=None):
        return tf.nn.softmax(input, name=name)

    def _fc(self, input, size_out, name=None, biased=True, activation='relu',
            weight_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            trainable=True, reuse=None):
        act_flag, act_func = self._get_activation(activation=activation)

        with tf.variable_scope(name, reuse=reuse):
            input_shape = input.get_shape()
            if 5 > input_shape.ndims > 1:
                # [batch_size, height, width, channel]
                # [batch_size, seq_len, word_size]
                # [batch_size, n_features]
                dim = int(input_shape[-1])
                feed_in = tf.reshape(input, [-1, dim])
            else:
                raise ValueError('nonsupport dim {}.'.format(input_shape.ndims))
            init_weight = weight_initializer
            init_bias = bias_initializer

            weight = self.make_var('weight', [dim, size_out], initializer=init_weight, trainable=trainable)

            output = tf.matmul(feed_in, weight)
            output = tf.reshape(output, tf.concat([tf.shape(input)[:-1], [size_out]], 0))
            if biased:
                bias = self.make_var('bias', [size_out], initializer=init_bias, trainable=trainable)
                output = tf.nn.bias_add(output, bias)
            if act_flag:
                output = act_func(output)

            return output

    def _conv1d(self, input, k_s, c_o, s, name=None, biased=True, activation='relu', padding='SAME',
                trainable=True, reuse=False):
        act_flag, act_func = self._get_activation(activation=activation)

        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv1d(i, k, stride=s, padding=padding)
        with tf.variable_scope(name, reuse=reuse):
            init_weight = tf.contrib.layers.xavier_initializer()
            init_bias = tf.constant_initializer(0.)
            kernel = self.make_var('weight', [k_s, c_i, c_o],
                                   initializer=init_weight,
                                   trainable=trainable)
            if biased:
                bias = self.make_var('bias', [c_o],
                                     initializer=init_bias,
                                     trainable=trainable)
                conv = convolve(input, kernel)
                if act_flag:
                    return act_func(tf.nn.bias_add(conv, bias))
                return tf.nn.bias_add(conv, bias)
            else:
                conv = convolve(input, kernel)
                if act_flag:
                    return act_func(conv)
                return conv

    def _deconv1d(self, input, k_s, c_o, o_s, s, name=None, biased=True, activation='relu', padding='SAME',
                  trainable=True, reuse=False):
        act_flag, act_func = self._get_activation(activation=activation)
        c_i = input.get_shape()[-1]
        deconvolve = lambda i, k: tf.contrib.nn.conv1d_transpose(i, k, output_shape=o_s, stride=s, padding=padding)
        with tf.variable_scope(name, reuse=reuse):
            init_weight = tf.contrib.layers.xavier_initializer()
            init_bias = tf.constant_initializer(0.)
            kernel = self.make_var('weight', [k_s, c_o, c_i],
                                   initializer=init_weight,
                                   trainable=trainable)

            if biased:
                bias = self.make_var('bias', [c_o],
                                     initializer=init_bias,
                                     trainable=trainable)
                deconv = deconvolve(input, kernel)
                if act_flag:
                    return act_func(tf.nn.bias_add(deconv, bias))
                return tf.nn.bias_add(deconv, bias)
            else:
                deconv = deconvolve(input, kernel)
                if act_flag:
                    return act_func(deconv)
                return deconv

    def _dense(self, input, output_size, name, biased=True, seq_len=None, reuse=False, trainable=True):
        with tf.variable_scope(name, reuse=reuse):
            input_size = int(input.shape[-1])
            init_weight = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
            init_bias = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)

            weight = self.make_var('weight', [input_size, output_size], initializer=init_weight, trainable=trainable)
            xw = tf.matmul(tf.reshape(input, (-1, input_size)), weight)
            xw = tf.reshape(xw, tf.concat([tf.shape(input)[:-1], [output_size]], 0))
            if biased:
                bias = self.make_var('bias', [output_size], initializer=init_bias, trainable=trainable)
                output = tf.nn.bias_add(xw, bias)
            else:
                output = xw

            if seq_len is not None:
                output = self._mask(output, seq_len, 'mul')
        return output

    def _mask(self, input, seq_len, mode='mul'):
        """
        :param input: 是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
        :param seq_len: 是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
        :param mode: 
        分为mul和add，
        mul是指把多出部分全部置零，一般用于全连接层之前；
        add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
        :return: 
        """
        if seq_len is None:
            return input
        else:
            m = tf.cast(tf.sequence_mask(seq_len), tf.float32)
            for _ in range(len(input.shape) - 2):
                m = tf.expand_dims(m, 2)
            if mode == 'mul':
                return input * m
            if mode == 'add':
                return input - (1 - m) * 1e12

    def _layer_normalize(self, input, epsilon=1e-8, name="ln", trainable=True, reuse=None):
        """Applies layer normalization.

        Args:
          input: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          name: Optional scope for `variable_scope`.
          trainable: 
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor with the same shape and data dtype as `input`.
        """
        with tf.variable_scope(name, reuse=reuse):
            input_shape = input.get_shape()
            param_shape = input_shape[-1:]

            mean, variance = tf.nn.moments(input, [-1], keep_dims=True)
            beta = self.make_var('beta', param_shape, initializer=tf.zeros_initializer, trainable=trainable)
            gamma = self.make_var('gamma', param_shape, initializer=tf.ones_initializer, trainable=trainable)
            normalized = (input - mean) / ((variance + epsilon) ** 0.5)
            output = gamma * normalized + beta

        return output

    def _multi_head_attention(self, q, k, v, nb_head, size_head, name, q_len=None, v_len=None, reuse=False):
        """
        :param q: [batch_size, 1, n_features]
        :param k: 
        :param v: 
        :param nb_head: 8
        :param size_head: 64
        :param name: 
        :param q_len: 
        :param v_len: 
        :param reuse: 
        :return: 
        """
        with tf.variable_scope(name, reuse=reuse):
            # 对q, k, v分别做线性映射
            q = self._dense(q, nb_head * size_head, name='a', biased=False,
                            reuse=reuse)  # [batch_size, len, nb_head*size_hed]
            q = tf.reshape(q, (-1, tf.shape(q)[1], nb_head, size_head))  # [batch_size, len, nb_head, size_head]
            q = tf.transpose(q, [0, 2, 1, 3])  # [batch_size, nb_head, len, size_head]

            k = self._dense(k, nb_head * size_head, name='b', biased=False,
                            reuse=reuse)  # [batch_size, len, nb_head*size_hed]
            k = tf.reshape(k, (-1, tf.shape(k)[1], nb_head, size_head))  # [batch_size, len, nb_head, size_head]
            k = tf.transpose(k, [0, 2, 1, 3])  # [batch_size, nb_head, len, size_head]

            v = self._dense(v, nb_head * size_head, name='c', biased=False,
                            reuse=reuse)  # [batch_size, len, nb_head*size_hed]
            v = tf.reshape(v, (-1, tf.shape(v)[1], nb_head, size_head))  # [batch_size, len, nb_head, size_head]
            v = tf.transpose(v, [0, 2, 1, 3])  # [batch_size, nb_head, len, size_head]

            # 计算内积，mask, softmax
            a = tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(size_head))  # [batch_size, nb_head, len, len]
            a = tf.transpose(a, [0, 3, 2, 1])  # [batch_size, len, len, nb_head]
            a = self._mask(a, v_len, mode='add')
            a = tf.transpose(a, [0, 3, 2, 1])  # [batch_size, nb_head, len, len]
            a = tf.nn.softmax(a)  # [batch_size, nb_head, len, len]

            # 输出并mask
            o = tf.matmul(a, v)  # [batch_size, nb_head, len, size_head]
            o = tf.transpose(o, [0, 2, 1, 3])  # [batch_size, len, nb_head, size_head]
            o = tf.reshape(o, (-1, tf.shape(o)[1], nb_head * size_head))  # [batch_size, len, nb_head * size_head]
            o = self._mask(o, q_len, 'mul')

            return o

    def _attention(self, q, k, v, nb_head, size_head, name, trainable=True, reuse=None):
        """
        :param q: [batch_size, 1, n_features]
        :param k: 
        :param v: 
        :param nb_head: 8
        :param size_head: 64
        :param name: 
        :param reuse: 
        :return: 
        """
        with tf.variable_scope(name, reuse=reuse):
            # 对q, k, v分别做线性映射
            q = self._fc(q, nb_head * size_head, name='a', biased=True, activation='relu',
                         trainable=trainable, reuse=reuse)  # [batch_size, len, nb_head*size_hed]
            q = tf.reshape(q, (-1, tf.shape(q)[1], nb_head, size_head))  # [batch_size, len, nb_head, size_head]
            q = tf.transpose(q, [0, 2, 1, 3])  # [batch_size, nb_head, len, size_head]

            k = self._fc(k, nb_head * size_head, name='b', biased=True, activation='relu',
                         trainable=trainable, reuse=reuse)  # [batch_size, len, nb_head*size_hed]
            k = tf.reshape(k, (-1, tf.shape(k)[1], nb_head, size_head))  # [batch_size, len, nb_head, size_head]
            k = tf.transpose(k, [0, 2, 1, 3])  # [batch_size, nb_head, len, size_head]

            v = self._fc(v, nb_head * size_head, name='c', biased=True, activation='relu',
                         trainable=trainable, reuse=reuse)  # [batch_size, len, nb_head*size_hed]
            v = tf.reshape(v, (-1, tf.shape(v)[1], nb_head, size_head))  # [batch_size, len, nb_head, size_head]
            v = tf.transpose(v, [0, 2, 1, 3])  # [batch_size, nb_head, len, size_head]

            # 计算内积，mask, softmax
            a = tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(size_head))  # [batch_size, nb_head, len, len]
            a = tf.nn.softmax(a)  # [batch_size, nb_head, len, len]

            # 输出
            o = tf.matmul(a, v)  # [batch_size, nb_head, len, size_head]
            o = tf.transpose(o, [0, 2, 1, 3])  # [batch_size, len, nb_head, size_head]
            o = tf.reshape(o, (-1, tf.shape(o)[1], nb_head * size_head))  # [batch_size, len, nb_head * size_head]

            return o

    @layer
    def relu(self, input, name=None):
        return tf.nn.relu(input, name=name)

    @layer
    def elu(self, input, name=None):
        return tf.nn.elu(input, name=name)

    @layer
    def concat(self, inputs, axis, name=None):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def maxpool1d(self, input, padding, name=None):
        return tf.nn.pool(input, window_shape=[2], strides=[2], pooling_type='MAX', name=name, padding=padding)

    @layer
    def deconv1d(self, input, k_s, c_o, o_s, s, name=None, biased=True, activation='relu', padding='SAME',
                 trainable=True, reuse=False):
        return self._deconv1d(input, k_s, c_o, o_s, s, name, biased, activation, padding, trainable, reuse)

    @layer
    def conv1d(self, input, k_s, c_o, s, name=None, biased=True, activation='relu', padding='SAME',
               trainable=True, reuse=False):
        return self._conv1d(input, k_s, c_o, s, name, biased, activation, padding, trainable, reuse)

    @layer
    def fc(self, input, num_out, name=None, biased=True, activation='relu',
           trainable=True, reuse=False):
        return self._fc(input, num_out, name, biased, activation, trainable, reuse)

    @layer
    def softmax(self, input, name=None):
        return tf.nn.softmax(input, name=name)

    @layer
    def tanh(self, input, name=None):
        return tf.nn.tanh(input, name=name)

    @layer
    def dropout(self, input, keep_prob, name=None):
        return tf.nn.dropout(input, keep_prob, name=name)

    @layer
    def add(self, input, name=None):
        assert len(input) == 2
        return tf.add(input[0], input[1], name=name)

    @layer
    def multiply(self, input, name=None):
        assert len(input) == 2
        return tf.multiply(input[0], input[1], name=name)
