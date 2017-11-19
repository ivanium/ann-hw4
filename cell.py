import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class BasicRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

        self.W = weight_variable([FLAGS.embed_units, self.state_size])
        self.U = weight_variable([self.state_size, self.state_size])
        self.b = bias_variable([self.state_size])

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            #todo: implement the new_state calculation given inputs and state
            new_state = self._activation(tf.matmul(inputs, self.W) + tf.matmul(state, self.U) + self.b)

        return new_state, new_state

class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

        self.W_z = weight_variable([FLAGS.embed_units, self.state_size])
        self.W_r = weight_variable([FLAGS.embed_units, self.state_size])
        self.W_c = weight_variable([FLAGS.embed_units, self.state_size])

        self.U_z = weight_variable([self.state_size, self.state_size])
        self.U_r = weight_variable([self.state_size, self.state_size])
        self.U_c = weight_variable([self.state_size, self.state_size])

        self.b_z = bias_variable([self.state_size], init_value = 1)
        self.b_r = bias_variable([self.state_size], init_value = 1)
        self.b_c = bias_variable([self.state_size], init_value = 1)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            #We start with bias of 1.0 to not reset and not update.
            #todo: implement the new_h calculation given inputs and state
            new_z = tf.sigmoid(tf.matmul(inputs, self.W_z) + tf.matmul(state, self.U_z) + self.b_z)
            new_r = tf.sigmoid(tf.matmul(inputs, self.W_r) + tf.matmul(state, self.U_r) + self.b_r)
            temp_h = self._activation(tf.matmul(inputs, self.W_c) + tf.matmul(tf.multiply(new_r, state), self.U_c) + self.b_c)
            new_h = tf.multiply(1-new_z, state) + tf.multiply(new_z, temp_h)
        return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

        self.W_i = weight_variable([FLAGS.embed_units, self.state_size[0]])
        self.W_o = weight_variable([FLAGS.embed_units, self.state_size[0]])
        self.W_f = weight_variable([FLAGS.embed_units, self.state_size[0]])
        self.W_c = weight_variable([FLAGS.embed_units, self.state_size[0]])
        
        self.U_i = weight_variable([self.state_size[0], self.state_size[0]])
        self.U_o = weight_variable([self.state_size[0], self.state_size[0]])
        self.U_f = weight_variable([self.state_size[0], self.state_size[0]])
        self.U_c = weight_variable([self.state_size[0], self.state_size[0]])

        self.b_i = bias_variable([self.state_size[0]])
        self.b_o = bias_variable([self.state_size[0]])
        self.b_f = bias_variable([self.state_size[0]], init_value = self._forget_bias)
        self.b_c = bias_variable([self.state_size[0]])

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            #todo: implement the new_c, new_h calculation given inputs and state (c, h)
            input_gate = tf.sigmoid(tf.matmul(inputs, self.W_i) + tf.matmul(h, self.U_i) + self.b_i)
            output_gate = tf.sigmoid(tf.matmul(inputs, self.W_o) + tf.matmul(h, self.U_o) + self.b_o)
            forget_gate = tf.sigmoid(tf.matmul(inputs, self.W_f) + tf.matmul(h, self.U_f) + self.b_f)
            temp_c = self._activation(tf.matmul(inputs, self.W_c) + tf.matmul(h, self.U_c) + self.b_c)
            new_c = tf.multiply(forget_gate, c) + tf.multiply(input_gate, temp_c)
            new_h = tf.multiply(output_gate, self._activation(new_c))
        return new_h, (new_c, new_h)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape, init_value = 0.1):  # you can use this func to build new variables
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial)
