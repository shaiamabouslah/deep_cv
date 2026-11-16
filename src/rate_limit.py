"""Minimal in-memory rate limiter for endpoints that trigger a Google Earth
Engine call — protects the free quota from being burned by a runaway script,
an accidental loop, or multiple people clicking fast during a demo.

Per-process, in-memory sliding window: fine for a single-worker dev/demo
deployment. A multi-worker production deployment would need a shared store
(Redis) instead — noted as a known limitation, not implemented here."""
import time
from collections import defaultdict

from fastapi import HTTPException, Request

_calls_by_key = defaultdict(list)


def rate_limit(max_calls: int = 10, window_seconds: int = 60):
    def dependency(request: Request):
        key = request.client.host if request.client else "unknown"
        now = time.time()
        calls = _calls_by_key[key]
        calls[:] = [t for t in calls if now - t < window_seconds]
        if len(calls) >= max_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Trop de requêtes ({max_calls} max par {window_seconds}s). Réessaie dans un instant.",
            )
        calls.append(now)
    return dependency
