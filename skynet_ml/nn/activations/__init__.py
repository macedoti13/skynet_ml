from skynet_ml.nn.activations.sigmoid import Sigmoid
from skynet_ml.nn.activations.relu import ReLU
from skynet_ml.nn.activations.leaky_relu import LeakyReLU
from skynet_ml.nn.activations.linear import Linear
from skynet_ml.nn.activations.softmax import Softmax
from skynet_ml.nn.activations.tanh import Tanh


ACTIVATIONS_MAP = {
    "sigmoid": Sigmoid,
    "Sigmoid": Sigmoid,
    "relu": ReLU,
    "ReLU": ReLU,
    "leaky_relu": LeakyReLU,
    "LeakyReLU": LeakyReLU,
    "linear": Linear,
    "Linear": Linear,
    "softmax": Softmax,
    "Softmax": Softmax,
    "tanh": Tanh,
    "Tanh": Tanh
}