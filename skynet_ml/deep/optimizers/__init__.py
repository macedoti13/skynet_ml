from skynet_ml.deep.optimizers.sgd import SGD
from skynet_ml.deep.optimizers.adagrad import AdaGrad
from skynet_ml.deep.optimizers.rmsprop import RMSProp
from skynet_ml.deep.optimizers.adam import Adam

optimizer_map = {
    "sgd": SGD(),
    "adagrad": AdaGrad(),
    "rmsprop": RMSProp(),
    "adam": Adam()
}