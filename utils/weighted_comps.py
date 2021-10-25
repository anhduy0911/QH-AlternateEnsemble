import tensorflow as tf

class WeightedComps(tf.keras.constraints.Constraint):
  """
  Constrains the weights to be non-negative.
  Using softmax and multiply with input shape.
  """
  def __call__(self, w):
    return tf.exp(w) / tf.reduce_sum(tf.exp(w), axis=0) * w.shape[0]