import tensorflow.keras.backend as be 
import tensorflow as tf

def shrinkage_loss(y_gt, y_pred):
    a = 5
    c = 0.1
    diff = be.abs(y_pred - y_gt)
    weight = 1 / (be.exp(a * (c - diff)))
    l2 = be.square(diff)

    return weight * l2

def focal_loss(y_pred, y_gt):
    diff = be.abs(y_pred - y_gt)
    return be.pow(diff, 3)

def linex_loss(y_pred, y_gt):
    alpha = -0.1
    diff = y_pred - y_gt
    # diff = y_pred / y_gt - 1
    exp = be.exp(alpha * diff)
    linear = alpha * diff
    return be.clip(exp - linear - 1, 1e-7, 1e7)


def fair_weight_mse(mean, sd):
    def f_loss(y_gt, y_pred):
        a = 2
        diff = be.abs(mean - y_gt)
        weight = 1 / (be.exp(a * diff))
        return be.square(y_gt - y_pred) * weight
    def linex_loss(y_gt, y_pred):
      # weight = be.exp(be.abs(y_gt - mean) / sd)
      a = 2
      c = 0.1
      diff = be.abs(mean - y_gt)
      weight = 1 / (be.exp(a * diff))
      # l2 = be.square(diff)
      # beta = 0
      # weight = beta * sd / be.abs(y_gt - mean)
      alpha = -0.1
      diff = y_pred - y_gt
      # diff = y_pred / y_gt - 1
      exp = be.exp(alpha * diff)
      linear = alpha * diff
      return be.clip((exp - linear - 1) * weight, 1e-10, 1e10) 
    return f_loss

# def linex_loss(y_pred, y_gt):
#     # alpha = -0.01
#     diff = y_pred - y_gt
#     # diff = y_pred / y_gt - 1
#     # exp = be.exp(alpha * diff)
#     sign = be.sign(diff)
#     # linear = alpha * diff

#     # poly = be.pow(diff, - sign * 2)
#     poly = tf.math.pow(diff, -sign * 2)
#     print(poly)
#     return be.clip(poly + diff, 1e-6, 1e6)


if __name__ == '__main__':
  a = tf.convert_to_tensor([[0.5,2.5,1], [0.5, 2.5, 1]], dtype=tf.float32)
  b = tf.convert_to_tensor([[1,2,0], [1, 2, 0]], dtype=tf.float32)
  print(linex_loss(a, b))
#   print(linex_loss_2(a, b))