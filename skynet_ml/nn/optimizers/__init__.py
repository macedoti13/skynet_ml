from skynet_ml.nn.optimizers.sgd import StochasticGradientDescent
from skynet_ml.nn.optimizers.adagrad import AdaGrad
from skynet_ml.nn.optimizers.rmsprop import RMSProp
from skynet_ml.nn.optimizers.adam import Adam

OPTIMIZERS_MAP = {
    "sgd": StochasticGradientDescent,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adam": Adam,
}
