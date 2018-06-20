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

    def _get_shape(self, tensor):
        return tf.shape(tensor)

    def _swish(self, input, name=None):
        return tf.multiply(input, tf.nn.sigmoid(input), name=name)

    def _reshape(self, input, shape, name=None):
        return tf.reshape(input, shape=shape, name=name)

    def _pool1d(self, input, k_s, s, name, p_t='MAX', padding='SAME'):
        return tf.nn.pool(input, window_shape=[k_s], strides=[s], pooling_type=p_t, name=name, padding=padding)

    def _global_average_pool1d(self, input, keepdims, name):
        return tf.reduce_mean(input, axis=[1], keepdims=keepdims, name=name)

    def _softmax(self, input, name):
        return tf.nn.softmax(input, name=name)

    def _fc(self, input, num_out, name, biased=True, activation='relu', trainable=True, reuse=False):
        act_flag, act_func = self._get_activation(activation=activation)

        with tf.variable_scope(name, reuse=reuse):
            if isinstance(input, tuple):
                input = input[0]  # only use the first input

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim])
            elif input_shape.ndims == 2:
                feed_in, dim = (input, int(input_shape[-1]))
            else:
                raise ValueError('nonsupport dim {}.'.format(input_shape.ndims))
            init_weight = tf.contrib.layers.xavier_initializer()
            init_bias = tf.constant_initializer(0.)

            weight = self.make_var('weight', [dim, num_out], initializer=init_weight, trainable=trainable)

            xw = tf.matmul(feed_in, weight)
            if biased:
                bias = self.make_var('bias', [num_out], initializer=init_bias, trainable=trainable)
                if act_flag:
                    return act_func(tf.nn.bias_add(xw, bias))
                return tf.nn.bias_add(xw, bias)
            else:
                if act_flag:
                    return act_func(xw)
                return xw

    def _conv1d(self, input, k_s, c_o, s, name, biased=True, activation='relu', padding='SAME',
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

    def _deconv1d(self, input, k_s, c_o, o_s, s, name, biased=True, activation='relu', padding='SAME', trainable=True, reuse=False):
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

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def elu(self, input, name):
        return tf.nn.elu(input, name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def maxpool1d(self, input, name, padding):
        return tf.nn.pool(input, window_shape=[2], strides=[2], pooling_type='MAX', name=name, padding=padding)

    @layer
    def deconv1d(self, input, k_s, c_o, o_s, s, name, biased=True, activation='relu', padding='SAME',
                 trainable=True, reuse=False):
        return self._deconv1d(input, k_s, c_o, o_s, s, name, biased, activation, padding, trainable, reuse)

    @layer
    def conv1d(self, input, k_s, c_o, s, name, biased=True, activation='relu', padding='SAME',
               trainable=True, reuse=False):
        return self._conv1d(input, k_s, c_o, s, name, biased, activation, padding, trainable, reuse)

    @layer
    def fc(self, input, num_out, name, biased=True, activation='relu',
           trainable=True, reuse=False):
        return self._fc(input, num_out, name, biased, activation, trainable, reuse)

    @layer
    def softmax(self, input, name):
        return tf.nn.softmax(input, name=name)

    @layer
    def tanh(self, input, name):
        return tf.nn.tanh(input, name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    @layer
    def add(self, input, name):
        assert len(input) == 2
        return tf.add(input[0], input[1], name=name)

    @layer
    def multiply(self, input, name):
        assert len(input) == 2
        return tf.multiply(input[0], input[1], name=name)
