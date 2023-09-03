from skynet_ml.deep.losses.losses import *

loss_map = {
    "mse": mse,
    "bce": binary_cross_entropy
}

d_loss_map = {
    "mse": d_mse,
    "bce": d_binary_cross_entropy
}