"""Tests for providers/schwab/auth.py."""

import json
import time
from unittest.mock import patch, mock_open, MagicMock

import pytest

from providers.schwab.auth import (
    SchwabAuthError,
    get_valid_access_token,
    exchange_code_for_tokens,
)


# ---------------------------------------------------------------------------
# No token file -> error with actionable text
# ---------------------------------------------------------------------------

class TestNoTokenFile:
    @patch("providers.schwab.auth._load_token_cache")
    def test_raises_actionable_error(self, mock_load):
        mock_load.return_value = None
        with pytest.raises(SchwabAuthError, match="schwab_setup"):
            get_valid_access_token()


# ---------------------------------------------------------------------------
# Token exchange saves cache with obtained_at
# ---------------------------------------------------------------------------

class TestTokenExchange:
    @patch("providers.schwab.auth.requests.post")
    @patch("providers.schwab.auth.TOKEN_CACHE_PATH")
    @patch("providers.schwab.auth.ensure_cache_dir")
    def test_exchange_saves_with_obtained_at(self, mock_ensure, mock_path, mock_post):
        """exchange_code_for_tokens delegates to _save_token_cache, which
        adds an obtained_at timestamp before writing the cache file.
        Verify the file written to disk includes that timestamp."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "access_token": "abc123",
            "refresh_token": "ref456",
            "expires_in": 1800,
        }
        mock_post.return_value = mock_resp

        exchange_code_for_tokens(
            client_id="test_id",
            client_secret="test_secret",
            redirect_uri="https://127.0.0.1",
            authorization_code="authcode",
        )

        # _save_token_cache should have written to TOKEN_CACHE_PATH
        assert mock_path.write_text.called
        saved_json = mock_path.write_text.call_args[0][0]
        saved = json.loads(saved_json)
        assert "obtained_at" in saved
        assert saved["access_token"] == "abc123"
        assert saved["refresh_token"] == "ref456"


# ---------------------------------------------------------------------------
# Fresh (non-expired) token returned without refresh
# ---------------------------------------------------------------------------

class TestFreshToken:
    @patch("providers.schwab.auth._refresh_access_token")
    @patch("providers.schwab.auth._load_token_cache")
    def test_fresh_token_no_refresh(self, mock_load, mock_refresh):
        mock_load.return_value = {
            "access_token": "fresh_token",
            "refresh_token": "ref123",
            "client_id": "id",
            "client_secret": "secret",
            "expires_in": 1800,
            "obtained_at": time.time(),  # just obtained (fresh)
        }
        token = get_valid_access_token()
        assert token == "fresh_token"
        mock_refresh.assert_not_called()


# ---------------------------------------------------------------------------
# Expired token triggers exactly one refresh
# ---------------------------------------------------------------------------

class TestExpiredToken:
    @patch("providers.schwab.auth.requests.post")
    @patch("providers.schwab.auth._save_token_cache")
    @patch("providers.schwab.auth._load_token_cache")
    def test_expired_token_refreshed_once(self, mock_load, mock_save, mock_post):
        mock_load.return_value = {
            "access_token": "old_token",
            "refresh_token": "ref123",
            "client_id": "id",
            "client_secret": "secret",
            "expires_in": 1800,
            "obtained_at": time.time() - 3600,  # 1 hour ago (expired)
        }
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "access_token": "new_token",
            "refresh_token": "ref789",
            "expires_in": 1800,
        }
        mock_post.return_value = mock_resp

        token = get_valid_access_token()
        assert token == "new_token"
        # refresh should have been called exactly once
        assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# Failed refresh -> actionable error
# ---------------------------------------------------------------------------

class TestFailedRefresh:
    @patch("providers.schwab.auth.requests.post")
    @patch("providers.schwab.auth._load_token_cache")
    def test_failed_refresh_raises_actionable(self, mock_load, mock_post):
        mock_load.return_value = {
            "access_token": "old_token",
            "refresh_token": "ref123",
            "client_id": "id",
            "client_secret": "secret",
            "expires_in": 1800,
            "obtained_at": time.time() - 3600,  # expired
        }
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 400
        mock_resp.text = "invalid_grant"
        mock_post.return_value = mock_resp

        with pytest.raises(SchwabAuthError, match="schwab_setup"):
            get_valid_access_token()
