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
        # construct graph here.
        raise NotImplementedError('Must be sub-classed.')

    def load(self, data_path, sess, saver):
        # load model params here.
        raise NotImplementedError('Must be sub-classed.')

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer_name in args:
            if isinstance(layer_name, str):
                try:
                    layer = self.layers[layer_name]
                    self.inputs.append(layer)
                    print(layer)
                except KeyError:
                    print(self.layers.keys())
                    raise KeyError('Unknown layer name fed: %s' % layer_name)
            else:
                raise KeyError('Only str')
        return self

    def get_output(self, layer_name):
        try:
            layer = self.layers[layer_name]
        except KeyError:
            print(self.layers.keys())
            raise KeyError('Unknown layer name fed: %s' % layer_name)
        return layer

    def get_activation_func(self, activation='relu'):
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
        elif activation == 'selu':
            act_flag = True
            act_func = tf.nn.selu
        elif activation == 'swish':
            act_flag = True
            act_func = self.swish
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

    def make_var(self, name, shape, dtype=tf.float32, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, dtype=dtype, initializer=initializer, trainable=trainable,
                               regularizer=regularizer)

    def get_shape(self, input):
        return tf.shape(input)

    def selu(self, input):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(input >= 0.0, input, alpha * tf.nn.elu(input))

    def swish(self, input, name=None):
        return tf.multiply(input, tf.nn.sigmoid(input), name=name)

    def reshape(self, input, shape, name=None):
        return tf.reshape(input, shape=shape, name=name)

    def pool1d(self, input, k_s, s_s, name=None, p_t='MAX', padding='SAME'):
        return tf.nn.pool(input, window_shape=[k_s], strides=[s_s], pooling_type=p_t, name=name, padding=padding)

    def global_average_pool1d(self, input, keepdims, name=None):
        return tf.reduce_mean(input, axis=[1], keepdims=keepdims, name=name)

    def softmax(self, input, name=None):
        return tf.nn.softmax(input, name=name)

    def fc(self, input, num_out, name, biased=True, activation='relu',
           weight_initializer=tf.contrib.layers.xavier_initializer(),
           bias_initializer=tf.zeros_initializer(),
           weight_regularizer=None,
           trainable=True, reuse=None):
        act_flag, act_func = self.get_activation_func(activation=activation)

        with tf.variable_scope(name, reuse=reuse):
            input_shape = input.get_shape()
            if 5 > input_shape.ndims > 1:
                # ndims = 4 [N, H, W, C] IMAGE
                # ndims = 2 [N, N_FEATURE] STRUCTURED DATA
                # ndims = 3 [N, LENGTH, SIZE] NLP
                dim = int(input_shape[-1])
                feed_in = tf.reshape(input, [-1, dim])
            else:
                raise ValueError('nonsupport dim {}.'.format(input_shape.ndims))

            weight = self.make_var('weight', [dim, num_out], initializer=weight_initializer,
                                   regularizer=weight_regularizer, trainable=trainable)

            output = tf.matmul(feed_in, weight)
            output = tf.reshape(output, tf.concat([tf.shape(input)[:-1], [num_out]], 0))
            if biased:
                bias = self.make_var('bias', [num_out], initializer=bias_initializer, trainable=trainable)
                output = tf.nn.bias_add(output, bias)
            if act_flag:
                output = act_func(output)

            return output

    def conv1d(self, input, k_s, c_o, s_s, name, biased=True, activation='relu', padding='SAME',
               weight_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer(),
               trainable=True, reuse=None):
        act_flag, act_func = self.get_activation_func(activation=activation)

        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv1d(i, filters=k, stride=s_s, padding=padding)
        with tf.variable_scope(name, reuse=reuse):
            kernel = self.make_var('weight', [k_s, c_i, c_o], initializer=weight_initializer, trainable=trainable)
            output = convolve(input, kernel)
            if biased:
                bias = self.make_var('bias', [c_o], initializer=bias_initializer, trainable=trainable)
                output = tf.nn.bias_add(output, bias)

            if act_flag:
                output = act_func(output)
            return output

    def deconv1d(self, input, k_s, c_o, o_s, s_s, name, biased=True, activation='relu', padding='SAME',
                 weight_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer(),
                 trainable=True, reuse=None):
        act_flag, act_func = self.get_activation_func(activation=activation)
        c_i = input.get_shape()[-1]
        deconvolve = lambda i, k: tf.contrib.nn.conv1d_transpose(i, k, output_shape=o_s, stride=s_s, padding=padding)
        with tf.variable_scope(name, reuse=reuse):
            kernel = self.make_var('weight', [k_s, c_o, c_i], initializer=weight_initializer, trainable=trainable)

            output = deconvolve(input, kernel)
            if biased:
                bias = self.make_var('bias', [c_o], initializer=bias_initializer, trainable=trainable)
                output = tf.nn.bias_add(output, bias)

            if act_flag:
                output = act_func(output)

            return output

    def global_avg_pool(self, input, name=None, keepdims=True):
        return tf.reduce_mean(input, [1], name=name, keepdims=keepdims)

    def dropout(self, input, keep_prob, name=None):
        return tf.nn.dropout(input, keep_prob=keep_prob, name=name)

    @layer
    def relu_layer(self, input, name=None):
        return tf.nn.relu(input, name=name)

    @layer
    def elu_layer(self, input, name=None):
        return tf.nn.elu(input, name=name)

    @layer
    def concat_layer(self, inputs, axis, name=None):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def maxpool1d_layer(self, input, padding, name=None):
        return tf.nn.pool(input, window_shape=[2], strides=[2], pooling_type='MAX', name=name, padding=padding)

    @layer
    def deconv1d_layer(self, input, k_s, c_o, o_s, s_s, name, biased=True, activation='relu', padding='SAME',
                       weight_initializer=tf.contrib.layers.xavier_initializer(),
                       bias_initializer=tf.zeros_initializer(),
                       trainable=True, reuse=None):
        return self.deconv1d(input=input, k_s=k_s, c_o=c_o, o_s=o_s, s_s=s_s, name=name, biased=biased,
                             activation=activation, padding=padding, weight_initializer=weight_initializer,
                             bias_initializer=bias_initializer, trainable=trainable, reuse=reuse)

    @layer
    def conv1d_layer(self, input, k_s, c_o, s_s, name, biased=True, activation='relu', padding='SAME',
                     weight_initializer=tf.contrib.layers.xavier_initializer(),
                     bias_initializer=tf.zeros_initializer(),
                     trainable=True, reuse=None):
        return self.conv1d(input=input, k_s=k_s, c_o=c_o, s_s=s_s, name=name, biased=biased, activation=activation,
                           padding=padding, weight_initializer=weight_initializer,
                           bias_initializer=bias_initializer, trainable=trainable, reuse=reuse)

    @layer
    def fc_layer(self, input, num_out, name, biased=True, activation='relu',
                 weight_initializer=tf.contrib.layers.xavier_initializer(),
                 bias_initializer=tf.zeros_initializer(),
                 trainable=True, reuse=None):
        return self.fc(input=input, num_out=num_out, name=name, biased=biased, activation=activation,
                       weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                       trainable=trainable, reuse=reuse)

    @layer
    def softmax_layer(self, input, name=None):
        return tf.nn.softmax(input, name=name)

    @layer
    def tanh_layer(self, input, name=None):
        return tf.nn.tanh(input, name=name)

    @layer
    def dropout_layer(self, input, keep_prob, name=None):
        return tf.nn.dropout(input, keep_prob, name=name)

    @layer
    def add_layer(self, input, name=None):
        assert len(input) == 2
        return tf.add(input[0], input[1], name=name)

    @layer
    def multiply_layer(self, input, name=None):
        assert len(input) == 2
        return tf.multiply(input[0], input[1], name=name)
