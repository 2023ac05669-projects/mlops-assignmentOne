# src/metrics.py
from prometheus_client import Counter, Histogram

PREDICTION_COUNTER = Counter(
    "predictions_total", "Number of predictions served", ["model_name", "status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Latency of prediction endpoint"
)
RMSE_GAUGE = Histogram(
    "eval_rmse", "RMSE observed during evaluation runs"   # use Histogram to see distribution
)
