import os
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_ENV_VAR = "PARK_VISION_API_KEYS"
_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

# Format: "key1:cam-a,cam-b;key2:*;key3:cam-c"
# "*" (or omitting the ":scope" part) grants access to every camera.
# Example: PARK_VISION_API_KEYS="ops-team:*;vendor-x:cam-mashhad-01,cam-mashhad-02"


def _parse_scoped_keys() -> dict:
    raw = os.environ.get(API_KEY_ENV_VAR, "")
    scoped = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            key, scope = entry.split(":", 1)
            cameras = {c.strip() for c in scope.split(",") if c.strip()}
        else:
            key, cameras = entry, {"*"}
        key = key.strip()
        if key:
            scoped[key] = cameras or {"*"}
    return scoped


async def require_api_key(api_key: str = Security(_header_scheme)) -> Optional[str]:
    """FastAPI dependency guarding write/inference endpoints.

    Returns the resolved API key (for camera-scope checks downstream), or
    None if auth is disabled. If PARK_VISION_API_KEYS is unset, auth is
    disabled (local/dev mode) -- set it in any real deployment.
    """
    scoped_keys = _parse_scoped_keys()
    if not scoped_keys:
        return None
    if api_key is None or api_key not in scoped_keys:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    return api_key


def authorize_camera_access(api_key: Optional[str], camera_id: str) -> None:
    """Call after require_api_key once camera_id is known (it usually comes
    from the request body/form, so it isn't available to the auth dependency
    itself). No-op when auth is disabled (api_key is None)."""
    if api_key is None:
        return
    scoped_keys = _parse_scoped_keys()
    allowed = scoped_keys.get(api_key, set())
    if "*" not in allowed and camera_id not in allowed:
        raise HTTPException(
            status_code=403, detail=f"API key is not authorized for camera '{camera_id}'."
        )
