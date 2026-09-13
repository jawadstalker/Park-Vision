import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

RATE_LIMIT = os.environ.get("RATE_LIMIT", "60/minute")
DETECT_RATE_LIMIT = os.environ.get("DETECT_RATE_LIMIT", RATE_LIMIT)


def rate_limit_key(request: Request) -> str:
    # Rate-limit per API key when one is presented (so one tenant's traffic
    # can't starve another sharing the same egress IP); fall back to IP.
    api_key = request.headers.get("X-API-Key")
    return api_key or get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key, default_limits=[RATE_LIMIT])
