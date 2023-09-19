from skynet_ml.metrics.mse import MseMetric
from skynet_ml.metrics.rsquared import RSquaredMetric
from skynet_ml.metrics.accuracy import AccuracyMetric
from skynet_ml.metrics.precision import PrecisionMetric
from skynet_ml.metrics.recall import RecallMetric
from skynet_ml.metrics.fscore import FScoreMetric

METRICS_MAP = {
    "mse": MseMetric,
    "mean_squared_error": MseMetric,
    "rsquared": RSquaredMetric,
    "r2": RSquaredMetric,
    "accuracy": AccuracyMetric,
    "precision": PrecisionMetric,
    "recall": RecallMetric,
    "f1score": FScoreMetric,
    "f1": FScoreMetric,
}
