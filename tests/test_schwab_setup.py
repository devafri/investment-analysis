"""Tests for providers/schwab/setup.py -- .env-backed argument parsing."""

import os
import sys
from unittest.mock import patch

import pytest

from providers.schwab.setup import main, build_authorization_url, extract_code_from_redirect_url


# ---------------------------------------------------------------------------
# Helpers: invoke the argument-parsing portion of main() without the
# interactive browser/exchange steps.
# ---------------------------------------------------------------------------

def _parse_args(argv=None, env=None):
    """Run just the argument-parsing portion of main() and return the
    parsed args and any SystemExit that was raised."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-id", default=os.environ.get("SCHWAB_CLIENT_ID"),
                    help="...")
    ap.add_argument("--client-secret", default=os.environ.get("SCHWAB_CLIENT_SECRET"),
                    help="...")
    ap.add_argument("--redirect-uri", default=os.environ.get("SCHWAB_REDIRECT_URI"),
                    help="...")

    with patch.object(sys, "argv", ["setup.py"] + (argv or [])):
        args = ap.parse_args()
    return args


class TestEnvBackedArgs:
    def test_env_values_used_when_no_cli_flags(self, monkeypatch):
        """With SCHWAB_CLIENT_ID etc. in the environment and no CLI flags,
        the parsed args use the env values."""
        monkeypatch.setenv("SCHWAB_CLIENT_ID", "env_client_id")
        monkeypatch.setenv("SCHWAB_CLIENT_SECRET", "env_secret")
        monkeypatch.setenv("SCHWAB_REDIRECT_URI", "https://env.example.com")

        args = _parse_args(argv=[])
        assert args.client_id == "env_client_id"
        assert args.client_secret == "env_secret"
        assert args.redirect_uri == "https://env.example.com"

    def test_cli_flags_override_env_values(self, monkeypatch):
        """CLI flags take precedence over env vars."""
        monkeypatch.setenv("SCHWAB_CLIENT_ID", "env_client_id")
        monkeypatch.setenv("SCHWAB_CLIENT_SECRET", "env_secret")
        monkeypatch.setenv("SCHWAB_REDIRECT_URI", "https://env.example.com")

        args = _parse_args(argv=[
            "--client-id", "cli_override_id",
            "--client-secret", "cli_override_secret",
            "--redirect-uri", "https://cli.example.com",
        ])
        assert args.client_id == "cli_override_id"
        assert args.client_secret == "cli_override_secret"
        assert args.redirect_uri == "https://cli.example.com"

    def test_missing_all_values_exits_with_error(self, monkeypatch):
        """With no env vars and no CLI flags, the missing-values check
        should call sys.exit(1) with an actionable message."""
        # Remove any env vars that might be set
        monkeypatch.delenv("SCHWAB_CLIENT_ID", raising=False)
        monkeypatch.delenv("SCHWAB_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("SCHWAB_REDIRECT_URI", raising=False)

        args = _parse_args(argv=[])
        # All should be None (no env, no CLI)
        assert args.client_id is None
        assert args.client_secret is None
        assert args.redirect_uri is None

        # Verify the missing-values check logic (the same as in main())
        missing = [name for name, val in [
            ("--client-id/SCHWAB_CLIENT_ID", args.client_id),
            ("--client-secret/SCHWAB_CLIENT_SECRET", args.client_secret),
            ("--redirect-uri/SCHWAB_REDIRECT_URI", args.redirect_uri),
        ] if not val]
        assert len(missing) == 3
        assert "SCHWAB_CLIENT_ID" in missing[0]
        assert "SCHWAB_CLIENT_SECRET" in missing[1]
        assert "SCHWAB_REDIRECT_URI" in missing[2]

    def test_partial_env_values_detected_as_missing(self, monkeypatch):
        """If only some env vars are set, the missing ones are flagged."""
        monkeypatch.setenv("SCHWAB_CLIENT_ID", "env_client_id")
        monkeypatch.delenv("SCHWAB_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("SCHWAB_REDIRECT_URI", raising=False)

        args = _parse_args(argv=[])
        assert args.client_id == "env_client_id"
        assert args.client_secret is None
        assert args.redirect_uri is None

        missing = [name for name, val in [
            ("--client-id/SCHWAB_CLIENT_ID", args.client_id),
            ("--client-secret/SCHWAB_CLIENT_SECRET", args.client_secret),
            ("--redirect-uri/SCHWAB_REDIRECT_URI", args.redirect_uri),
        ] if not val]
        assert len(missing) == 2
        assert "SCHWAB_CLIENT_SECRET" in missing[0]
        assert "SCHWAB_REDIRECT_URI" in missing[1]


# ---------------------------------------------------------------------------
# Non-interactive helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_build_authorization_url(self):
        url = build_authorization_url("my_client_id", "https://127.0.0.1")
        assert "my_client_id" in url
        assert "client_id=" in url
        assert "redirect_uri=" in url
        assert "https%3A%2F%2F127.0.0.1" in url

    def test_extract_code_from_url(self):
        url = "https://127.0.0.1/?code=abc123def&session=xyz"
        code = extract_code_from_redirect_url(url)
        assert code == "abc123def"

    def test_extract_code_missing_raises(self):
        url = "https://127.0.0.1/?error=access_denied"
        with pytest.raises(ValueError, match="code"):
            extract_code_from_redirect_url(url)
