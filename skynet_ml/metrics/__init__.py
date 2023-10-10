from skynet_ml.metrics.classification.accuracy import Accuracy
from skynet_ml.metrics.classification.precision import Precision
from skynet_ml.metrics.classification.recall import Recall
from skynet_ml.metrics.classification.fscore import FScore
from skynet_ml.metrics.classification.confusion_matrix import ConfusionMatrix
from skynet_ml.metrics.regression.mae import MAE
from skynet_ml.metrics.regression.mse import MSE
from skynet_ml.metrics.regression.rmse import RMSE
from skynet_ml.metrics.regression.rsquared import RSquared


METRICS_MAP = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "fscore": FScore,
    "mae": MAE, 
    "mse": MSE,
    "rmse": RMSE,
    "rsquared": RSquared
}
