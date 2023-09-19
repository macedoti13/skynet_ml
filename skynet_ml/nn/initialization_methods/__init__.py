from skynet_ml.nn.initialization_methods.constant import ConstantInitializer
from skynet_ml.nn.initialization_methods.normal import NormalInitializer
from skynet_ml.nn.initialization_methods.uniform import UniformInitializer
from skynet_ml.nn.initialization_methods.xavier import XavierNormalInitializer, XavierUniformInitializer
from skynet_ml.nn.initialization_methods.he import HeNormalInitializer, HeUniformInitializer

INITIALIZERS_MAP = {
    'normal': NormalInitializer,
    'uniform': UniformInitializer,
    'constant': ConstantInitializer,
    'he_normal': HeNormalInitializer,
    'xavier_normal': XavierNormalInitializer,
    'he_uniform': HeUniformInitializer,
    'xavier_uniform': XavierUniformInitializer
}

ACTIVATIONS_INITIALIZER_MAP = {
    'relu': 'he_normal',
    'leaky_relu': 'he_normal',
    'sigmoid': 'xavier_normal',
    'tanh': 'xavier_normal',
    'softmax': 'xavier_uniform',
    'linear': 'xavier_normal',
}