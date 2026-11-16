from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.rate_limit import rate_limit


def _fake_request(ip="1.2.3.4"):
    req = MagicMock()
    req.client.host = ip
    return req


def test_allows_calls_under_the_limit():
    dependency = rate_limit(max_calls=3, window_seconds=60)
    req = _fake_request()
    for _ in range(3):
        dependency(req)  # should not raise


def test_blocks_calls_over_the_limit():
    dependency = rate_limit(max_calls=2, window_seconds=60)
    req = _fake_request("9.9.9.9")
    dependency(req)
    dependency(req)
    with pytest.raises(HTTPException) as exc_info:
        dependency(req)
    assert exc_info.value.status_code == 429


def test_different_ips_have_independent_limits():
    dependency = rate_limit(max_calls=1, window_seconds=60)
    dependency(_fake_request("1.1.1.1"))
    dependency(_fake_request("2.2.2.2"))  # different key, should not raise
