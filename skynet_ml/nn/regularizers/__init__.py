from skynet_ml.nn.regularizers.l1 import L1
from skynet_ml.nn.regularizers.l2 import L2
from skynet_ml.nn.regularizers.l1l2 import L1L2


REGULARIZERS_MAP = {
    "l1": L1,
    "L1": L1,
    "l2": L2,
    "L2": L2,
    "l1l2": L1L2,
    "L1L2": L1L2,
}
