"""OAuth2 token management for Schwab's Trader API / Market Data API.

I can't verify this against the live Schwab API from my environment (no
internet access, and even with it, I don't have your credentials) -- this is
built from my general knowledge of Schwab's OAuth2 flow (inherited from the
TD Ameritrade API it replaced), which may have shifted since my training
data. Verify the URLs/endpoints below against developer.schwab.com's current
docs before relying on this, and expect to need a small fix or two on first
run if something has changed.

FLOW OVERVIEW
1. One-time, manual, browser-based: register an app at developer.schwab.com,
   get a client_id ("App Key") and client_secret ("Secret"), and set a
   callback URL (many guides use https://127.0.0.1 -- there's no server
   actually listening there; you'll see a browser error page after granting
   consent, but the authorization code is in that page's URL).
2. Run schwab_setup.py once (see that file) to exchange the authorization
   code for an access_token + refresh_token, saved to cache/schwab_token.json.
3. From then on, get_valid_access_token() below handles refreshing the
   access_token automatically using the refresh_token, transparently, on
   every call that needs one -- BUT the refresh_token itself is reportedly
   short-lived (~7 days), so every so often you'll need to re-run
   schwab_setup.py's browser step again. This module can't do anything about
   that -- it's an inherent limitation of Schwab's token lifetime policy, not
   a bug here.
"""

import json
import time
from pathlib import Path
from typing import Optional

import requests

from core.paths import CACHE_DIR, ensure_cache_dir

SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
SCHWAB_AUTHORIZE_URL = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_CACHE_PATH = CACHE_DIR / "schwab_token.json"

# Refresh the access token this many seconds BEFORE it actually expires, to
# avoid a request failing mid-flight right at the expiry boundary.
ACCESS_TOKEN_REFRESH_MARGIN_SECONDS = 120


class SchwabAuthError(Exception):
    """Raised for anything auth-related: no token file yet, refresh failed,
    credentials missing, etc. Callers should show this message directly --
    it's written to be actionable, not a generic wrapper."""
    pass


def _load_token_cache() -> Optional[dict]:
    if not TOKEN_CACHE_PATH.exists():
        return None
    try:
        return json.loads(TOKEN_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_token_cache(token_data: dict) -> None:
    ensure_cache_dir()
    # Store an absolute expiry timestamp (not just "expires_in" seconds from
    # whenever the response arrived) so expiry checks later don't need to
    # remember when the token was issued.
    token_data = dict(token_data)
    if "expires_in" in token_data and "obtained_at" not in token_data:
        token_data["obtained_at"] = time.time()
    TOKEN_CACHE_PATH.write_text(json.dumps(token_data))


def exchange_code_for_tokens(client_id: str, client_secret: str, redirect_uri: str, authorization_code: str) -> dict:
    """One-time exchange of the authorization code (from the manual browser
    step) for an initial access_token + refresh_token pair. Called by
    schwab_setup.py, not typically called directly elsewhere."""
    resp = requests.post(
        SCHWAB_TOKEN_URL,
        auth=(client_id, client_secret),
        data={
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if not resp.ok:
        raise SchwabAuthError(
            f"Schwab rejected the authorization code exchange (HTTP {resp.status_code}): {resp.text}\n"
            f"Common causes: the authorization code was already used (they're single-use), the code "
            f"expired (they're short-lived, use it within a minute or two of getting it), or the "
            f"redirect_uri here doesn't exactly match what's registered for this app."
        )
    token_data = resp.json()
    token_data["client_id"] = client_id
    token_data["client_secret"] = client_secret
    _save_token_cache(token_data)
    return token_data


def _refresh_access_token(token_data: dict) -> dict:
    client_id = token_data.get("client_id")
    client_secret = token_data.get("client_secret")
    refresh_token = token_data.get("refresh_token")
    if not all([client_id, client_secret, refresh_token]):
        raise SchwabAuthError(
            "Token cache is missing client_id/client_secret/refresh_token -- "
            "run schwab_setup.py again to re-authenticate."
        )
    resp = requests.post(
        SCHWAB_TOKEN_URL,
        auth=(client_id, client_secret),
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    if not resp.ok:
        raise SchwabAuthError(
            f"Schwab rejected the refresh token (HTTP {resp.status_code}): {resp.text}\n"
            f"This usually means the refresh token has expired (Schwab's are reportedly "
            f"short-lived, ~7 days) -- run schwab_setup.py again to get a fresh one via the "
            f"browser authorization step."
        )
    new_token_data = resp.json()
    # Schwab issues a new refresh_token on every refresh in many OAuth2
    # implementations -- preserve it (and client credentials) in the cache.
    merged = dict(token_data)
    merged.update(new_token_data)
    _save_token_cache(merged)
    return merged


def get_connection_status() -> dict:
    """Lightweight status check for display purposes -- does NOT make a
    network call (no refresh attempt), just inspects the local token cache
    so it's safe to call on every page load. Returns:
        {"connected": bool, "detail": str}
    "connected" reflects whether a token file exists and its access token
    isn't obviously expired -- it does NOT guarantee the refresh_token is
    still valid (that can only be confirmed by actually trying a refresh,
    which get_valid_access_token() already does when real requests happen).
    """
    token_data = _load_token_cache()
    if token_data is None:
        return {"connected": False, "detail": "Not connected -- run `python3 -m providers.schwab.setup`"}

    expires_in = token_data.get("expires_in", 0)
    obtained_at = token_data.get("obtained_at", 0)
    expires_at = obtained_at + expires_in
    if time.time() >= expires_at:
        return {"connected": True, "detail": "Connected (access token will auto-refresh on next use)"}
    return {"connected": True, "detail": "Connected"}


def get_valid_access_token() -> str:
    """Returns a valid access_token, transparently refreshing it first if
    it's expired or close to expiring. Raises SchwabAuthError with an
    actionable message if there's no token yet or refreshing fails."""
    token_data = _load_token_cache()
    if token_data is None:
        raise SchwabAuthError(
            "No Schwab token found yet. Run `python3 schwab_setup.py` once to "
            "complete the one-time browser authorization step."
        )

    expires_in = token_data.get("expires_in", 0)
    obtained_at = token_data.get("obtained_at", 0)
    expires_at = obtained_at + expires_in

    if time.time() >= (expires_at - ACCESS_TOKEN_REFRESH_MARGIN_SECONDS):
        token_data = _refresh_access_token(token_data)

    access_token = token_data.get("access_token")
    if not access_token:
        raise SchwabAuthError("Token cache exists but has no access_token -- run schwab_setup.py again.")
    return access_token


# ---------------------------------------------------------------------------
# Pending credentials — used by the web-based OAuth flow.
# When the user clicks "Connect Schwab" on the Setup page, we store their
# credentials temporarily, redirect them to Schwab's authorization page,
# and look them up again when Schwab redirects back to /schwab/callback.
# ---------------------------------------------------------------------------

import secrets as _secrets
from urllib.parse import urlencode as _urlencode

PENDING_AUTH_PATH = CACHE_DIR / "schwab_pending.json"


def build_authorization_url(client_id: str, redirect_uri: str) -> str:
    """Build the Schwab OAuth authorization URL."""
    params = {"client_id": client_id, "redirect_uri": redirect_uri}
    return f"{SCHWAB_AUTHORIZE_URL}?{_urlencode(params)}"


def save_pending_credentials(
    client_id: str, client_secret: str, redirect_uri: str,
) -> str:
    """Store OAuth credentials temporarily and return a random state token
    used to look them up on the callback."""
    state = _secrets.token_urlsafe(24)
    ensure_cache_dir()
    pending = {
        "state": state,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "created_at": time.time(),
    }
    PENDING_AUTH_PATH.write_text(json.dumps(pending))
    return state


def load_pending_credentials(state: str) -> Optional[dict]:
    """Load temporarily stored OAuth credentials, validating the state token
    matches.  Returns None if the state doesn't match or the pending file
    doesn't exist (e.g. expired, already consumed, or never created)."""
    if not PENDING_AUTH_PATH.exists():
        return None
    try:
        pending = json.loads(PENDING_AUTH_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if pending.get("state") != state:
        return None
    # Consume the pending file — each state token is single-use
    PENDING_AUTH_PATH.unlink()
    # Expire after 10 minutes
    if time.time() - pending.get("created_at", 0) > 600:
        return None
    return pending