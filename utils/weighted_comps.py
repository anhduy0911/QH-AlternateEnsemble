import tensorflow as tf

class WeightedComps(tf.keras.constraints.Constraint):
  """
  Constrains the weights to be non-negative.
  Using softmax and multiply with input shape.
  """
  def __call__(self, w):
    # print(f'WEIGHT SHAPE: {w.shape}')
    # return tf.nn.softmax(w, axis=0) * w.shape[0]
    return w