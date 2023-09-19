from skynet_ml.nn.activation_functions.leaky_relu import LeakyReLUActivation
from skynet_ml.nn.activation_functions.sigmoid import SigmoidActivation
from skynet_ml.nn.activation_functions.softmax import SoftmaxActivation
from skynet_ml.nn.activation_functions.linear import LinearActivation
from skynet_ml.nn.activation_functions.relu import ReLUActivation
from skynet_ml.nn.activation_functions.tanh import TanhActivation

ACTIVATIONS_MAP = {
    "leaky_relu": LeakyReLUActivation,
    "sigmoid": SigmoidActivation,
    "softmax": SoftmaxActivation,
    "linear": LinearActivation,
    "relu": ReLUActivation,
    "tanh": TanhActivation
}
