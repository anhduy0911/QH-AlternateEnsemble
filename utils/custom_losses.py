import tensorflow.keras.backend as be 


def shrinkage_loss(y_pred, y_gt):
    a = 5
    c = 0.1
    diff = be.abs(y_pred - y_gt)
    weight = 1 / (be.exp(a * (c - diff)))
    l2 = be.square(diff)

    return weight * l2
