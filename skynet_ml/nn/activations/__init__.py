from skynet_ml.nn.activations.leaky_relu import LeakyReLU
from skynet_ml.nn.activations.softmax import Softmax
from skynet_ml.nn.activations.sigmoid import Sigmoid
from skynet_ml.nn.activations.linear import Linear
from skynet_ml.nn.activations.relu import ReLU
from skynet_ml.nn.activations.tanh import Tanh


activations_map = {
    "leaky_relu": LeakyReLU,
    "softmax": Softmax,
    "sigmoid": Sigmoid,
    "linear": Linear,
    "relu": ReLU,
    "tanh": Tanh
}
