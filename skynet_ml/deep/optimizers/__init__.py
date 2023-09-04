from skynet_ml.deep.optimizers.sgd import SGD
from skynet_ml.deep.optimizers.adagrad import AdaGrad

optimizer_map = {
    "sgd": SGD(),
    "adagrad": AdaGrad()
}