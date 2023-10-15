from skynet_ml.nn.initializers.xavier import XavierNormal, XavierUniform
from skynet_ml.nn.initializers.he import HeNormal, HeUniform
from skynet_ml.nn.initializers.constant import Constant
from skynet_ml.nn.initializers.uniform import Uniform
from skynet_ml.nn.initializers.normal import Normal


initializers_map = {
    "constant": Constant,
    "normal": Normal,
    "uniform": Uniform,
    "xavier_normal": XavierNormal,
    "xavier_uniform": XavierUniform,
    "he_normal": HeNormal,
    "he_uniform": HeUniform
}


activations_initializers_map = {
    'relu': 'he_normal',
    'leaky_relu': 'he_normal',
    'sigmoid': 'xavier_normal',
    'tanh': 'xavier_normal',
    'softmax': 'xavier_uniform',
    'linear': 'xavier_normal',
}
