from skynet_ml.metrics.classification.confusion_matrix import ConfusionMatrix
from skynet_ml.metrics.regression.rmse import RootMeanSquaredError
from skynet_ml.metrics.classification.precision import Precision
from skynet_ml.metrics.classification.accuracy import Accuracy
from skynet_ml.metrics.regression.mae import MeanAbsoluteError
from skynet_ml.metrics.regression.mse import MeanSquaredError
from skynet_ml.metrics.classification.recall import Recall
from skynet_ml.metrics.classification.fscore import FScore
from skynet_ml.metrics.regression.rsquared import RSquared


metrics_map = {
    "root_mean_squared_error": RootMeanSquaredError,
    "mean_absolute_error": MeanAbsoluteError,
    "mean_squared_error": MeanSquaredError,
    "rmse": RootMeanSquaredError,
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "precision": Precision,
    "rsquared": RSquared,
    "accuracy": Accuracy,
    "recall": Recall,
    "fscore": FScore,
    "r2": RSquared,
}
