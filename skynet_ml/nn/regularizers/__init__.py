from skynet_ml.nn.regularizers.l1regularizer import L1Regularizer
from skynet_ml.nn.regularizers.l2regularizer import L2Regularizer
from skynet_ml.nn.regularizers.l1l2regularizer import L1L2Regularizer

REGULARIZERS_MAP = {
    "L1": L1Regularizer,
    "L2": L2Regularizer,
    "L1L2": L1L2Regularizer
}
