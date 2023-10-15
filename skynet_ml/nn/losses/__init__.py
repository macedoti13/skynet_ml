from skynet_ml.nn.losses.cce import CategoricalCrossEntropy
from skynet_ml.nn.losses.bce import BinaryCrossEntropy
from skynet_ml.nn.losses.mse import MeanSquaredError


losses_map = {
    "categorical_crossentropy": CategoricalCrossEntropy,
    "binary_crossentropy": BinaryCrossEntropy,
    "mean_squared_error": MeanSquaredError,
    "cce": CategoricalCrossEntropy,
    "bce": BinaryCrossEntropy,
    "mse": MeanSquaredError
}
