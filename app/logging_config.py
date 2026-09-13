"""Structured (JSON) logging so log lines are machine-parseable by whatever
aggregator you point at this (Loki, CloudWatch, ELK, etc.) instead of being
free-form text that only a human tailing the container can make sense of.
"""
import json
import logging
import os
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("camera_id", "job_id", "duration_ms", "status_code", "path"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Quiet noisy libraries down to warnings unless the operator asks for more.
    for noisy in ("uvicorn.access", "celery", "kombu"):
        logging.getLogger(noisy).setLevel(os.environ.get("LOG_LEVEL_LIBS", "WARNING"))
