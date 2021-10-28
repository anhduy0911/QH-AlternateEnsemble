import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.recurrent import LSTMCell

class AttentionRNN(layers.Layer):
    def __init__(self, batch_size, window_size, target_timesteps, output_dim, hidden_state=64, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.window_size = window_size
        self.target_timesteps = target_timesteps
        self.output_dim = output_dim
        self.hidden_state = hidden_state

        self.cell = LSTMCell(self.hidden_state)
        self.dense_state = Dense(self.output_dim)
        self.dense = Dense(self.output_dim, use_bias=False)


    def _attention(self, hidden_state, cell_state, input):
        attn_input = tf.concat([hidden_state, cell_state], axis=1)
        attn_input = tf.reshape(tf.tile(attn_input, [1, input.shape[1]]), [self.batch_size, input.shape[1], hidden_state.shape[1] * 2])

        z = tf.tanh(self.dense_state(attn_input)) + self.dense(input)
        presoftmax = Dense(1)(z)

        return tf.nn.softmax(presoftmax)


    def call(self, input):
        initial_state = self.cell.get_initial_state(input, self.batch_size, tf.float32)
        state = initial_state
        s, h = state
        outputs = []

        for i in range(self.window_size):
            alpha = self._attention(h, s, input)
            x_tilde = tf.squeeze(alpha) * input[:, :, i]

            cell_out, state = self.cell(x_tilde, state)
            s, h = state
            outputs.append(h)

        result = tf.transpose(tf.stack(outputs, axis=0), perm=(1,0,2))

        return result

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'batch_size': self.batch_size,
            'window_size': self.window_size,
            'target_timesteps': self.target_timesteps,
            'output_dim': self.output_dim,
            'hidden_state': self.hidden_state
        })
        return config    

