"""Prometheus metrics of the running service, at two levels.

System level (GET /metrics, scraped by Prometheus):
  - http_requests_total{method,route,status}, http_request_duration_seconds{route},
    http_requests_in_progress: traffic, latency and errors per route
  - process / host CPU, memory and disk gauges (psutil), refreshed at each scrape
  - model_inference_seconds: time spent in the neural network alone

Model level:
  - model_confidence, model_water_pct: distribution of what the model outputs
  - data_drift_psi{feature} and data_drift_status: input drift against the
    training reference (src/monitoring.py)
  - forecast_skill{horizon}: 1 - model error / no-change error, over matured
    forecasts (concept drift: falls toward 0 or below when the model stops
    earning its place)

Routes are labelled by their path template (/forecast/{site}), never by the
concrete URL, so a user cannot inflate the number of time series.
"""
import os
import time

import psutil
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily

REGISTRY = CollectorRegistry()
STATUS_CODE = {"ok": 0, "warn": 1, "alert": 2}
LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY)
LATENCY = Histogram("http_request_duration_seconds", "HTTP request duration", ["route"], buckets=LATENCY_BUCKETS,
                    registry=REGISTRY)
IN_PROGRESS = Gauge("http_requests_in_progress", "Requests being served", registry=REGISTRY)
INFERENCE = Histogram("model_inference_seconds", "Neural-network forward pass", buckets=LATENCY_BUCKETS, registry=REGISTRY)
CONFIDENCE = Histogram("model_confidence", "Mean pixel confidence per prediction", buckets=[i / 10 for i in range(1, 11)],
                       registry=REGISTRY)
WATER_PCT = Histogram("model_water_pct", "Predicted water share per image (%)", buckets=(1, 5, 10, 20, 35, 50, 75, 100),
                      registry=REGISTRY)
PREDICTIONS = Counter("model_predictions_total", "Segmentation predictions served", registry=REGISTRY)
DRIFT_PSI = Gauge("data_drift_psi", "Population stability index of a feature vs the training reference", ["feature"],
                  registry=REGISTRY)
DRIFT_STATUS = Gauge("data_drift_status", "0 ok, 1 warn, 2 alert (worst feature)", registry=REGISTRY)
FORECAST_SKILL = Gauge("forecast_skill", "1 - model MAE / no-change MAE over matured forecasts", ["horizon"],
                       registry=REGISTRY)
RETRAIN_NEEDED = Gauge("retraining_needed", "1 when monitoring says the models should be retrained", registry=REGISTRY)


class SystemCollector:
    """Resource gauges, read at scrape time."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(None)
        self.started = time.time()

    def collect(self):
        values = {
            "process_cpu_percent": ("CPU use of this service (% of one core)", self.process.cpu_percent(None)),
            "process_memory_bytes": ("Resident memory of this service", self.process.memory_info().rss),
            "process_uptime_seconds": ("Seconds since the service started", time.time() - self.started),
            "host_cpu_percent": ("CPU use of the machine", psutil.cpu_percent(None)),
            "host_memory_percent": ("Memory use of the machine", psutil.virtual_memory().percent),
            "host_disk_percent": ("Disk use of the volume holding the app", psutil.disk_usage(os.path.abspath(os.sep)).percent),
        }
        for name, (doc, value) in values.items():
            gauge = GaugeMetricFamily(name, doc)
            gauge.add_metric([], float(value))
            yield gauge


REGISTRY.register(SystemCollector())


def system_snapshot():
    process = psutil.Process(os.getpid())
    return {
        "process_cpu_percent": process.cpu_percent(0.1), "process_memory_mb": round(process.memory_info().rss / 2**20, 1),
        "host_cpu_percent": psutil.cpu_percent(0.1), "host_memory_percent": psutil.virtual_memory().percent,
        "host_disk_percent": psutil.disk_usage(os.path.abspath(os.sep)).percent, "threads": process.num_threads(),
    }


def route_of(request):
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


async def metrics_middleware(request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    IN_PROGRESS.inc()
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        IN_PROGRESS.dec()
        route = route_of(request)
        REQUESTS.labels(request.method, route, str(status)).inc()
        LATENCY.labels(route).observe(time.perf_counter() - start)


def record_prediction(inference_seconds, features):
    PREDICTIONS.inc()
    INFERENCE.observe(inference_seconds)
    if features.get("confidence") is not None:
        CONFIDENCE.observe(features["confidence"])
    if features.get("water_pct") is not None:
        WATER_PCT.observe(features["water_pct"])


def publish_monitoring(report):
    """Mirrors a src.monitoring.check_all() report into the gauges."""
    drift = report["data_drift"]
    for name, feature in drift.get("features", {}).items():
        DRIFT_PSI.labels(name).set(feature["psi"])
    DRIFT_STATUS.set(STATUS_CODE.get(drift["status"], 0))
    for horizon, entry in report["forecast_performance"].get("horizons", {}).items():
        if entry["skill"] is not None:
            FORECAST_SKILL.labels(str(horizon)).set(entry["skill"])
    RETRAIN_NEEDED.set(1 if report["retrain_needed"] else 0)


def render():
    return generate_latest(REGISTRY)
