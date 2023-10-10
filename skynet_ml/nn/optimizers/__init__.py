from skynet_ml.nn.optimizers.sgd import SGD
from skynet_ml.nn.optimizers.adam import Adam
from skynet_ml.nn.optimizers.rmsprop import RMSProp
from skynet_ml.nn.optimizers.adagrad import AdaGrad


OPTIMIZERS_MAP = {
    "sgd": SGD,
    "adam": Adam,
    "rmsprop": RMSProp,
    "adagrad": AdaGrad
}
