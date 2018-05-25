import tensorflow as tf

class RNN:
    def __init__(self, internal_size, in_size, out_size):
        self.internal_size = internal_size
        self.in_state = tf.placeholder(tf.float32, shape=(None,internal_size,), name="Hin")
        #recording and persistance gates
        self.gate_record = tf.Variable(tf.zeros((internal_size + internal_size,internal_size,)))
        self.bias_record = tf.Variable(tf.zeros((internal_size,)))
        self.gate_persist = tf.Variable(tf.zeros((internal_size, internal_size,)))
        #gate -> in
        self.gate_in = tf.Variable(tf.zeros(in_size+(internal_size,)))
        self.bias_in = tf.Variable(tf.zeros((internal_size,)))
        #state -> output
        self.gate_out = tf.Variable(tf.zeros((internal_size,)+out_size))
        self.bias_out = tf.Variable(tf.zeros(out_size))
    
    def unroll(self, data_tensor, n):
        '''Returns the output state, and an output'''
        state = tf.concat(
                    tf.nn.relu(self.in_state @ self.gate_persist),
                    tf.nn.relu(data_tensor[0] @ self.gate_in + self.bias_in)) @ self.gate_record + self.bias_record
        for i in range(1, n):
            state = tf.concat(
                tf.nn.relu(state @ self.gate_persist),
                tf.nn.relu(data_tensor[i] @ self.gate_in + self.bias_in)) @ self.gate_record + self.bias_record
        return state, state @ self.gate_out + self.bias_out