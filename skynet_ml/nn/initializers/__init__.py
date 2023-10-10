from skynet_ml.nn.initializers.constant import Constant
from skynet_ml.nn.initializers.normal import Normal
from skynet_ml.nn.initializers.uniform import Uniform
from skynet_ml.nn.initializers.xavier import XavierNormal, XavierUniform
from skynet_ml.nn.initializers.he import HeNormal, HeUniform


INITIALIZERS_MAP = {
    "constant": Constant,
    "Constant": Constant,
    "normal": Normal,
    "Normal": Normal,
    "uniform": Uniform,
    "Uniform": Uniform,
    "xavier_normal": XavierNormal,
    "xavier_uniform": XavierUniform,
    "Xavier Normal": XavierUniform,
    "Xavier Uniform": XavierUniform,
    "he_normal": HeNormal,
    "He Normal": HeNormal,
    "He Uniform": HeUniform,
    "he_uniform": HeUniform
}


ACTIVATIONS_INITIALIZER_MAP = {
    'relu': 'he_normal',
    'leaky_relu': 'he_normal',
    'sigmoid': 'xavier_normal',
    'tanh': 'xavier_normal',
    'softmax': 'xavier_uniform',
    'linear': 'xavier_normal',
}
