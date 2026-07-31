#!/usr/bin/env python3
"""
schwab_setup.py
----------------
Run this ONCE (and again whenever your refresh token expires, reportedly
every ~7 days) to complete Schwab's OAuth2 browser authorization step and
save a working token to cache/schwab_token.json.

BEFORE RUNNING THIS
1. Register an app at https://developer.schwab.com if you haven't already.
2. Note your client_id ("App Key") and client_secret ("Secret").
3. Note the callback/redirect URL you registered for the app. Many guides
   use https://127.0.0.1 -- there's no server actually listening there, so
   after you log in and grant consent, your browser will land on an error
   page. That's expected. The authorization code you need is in THAT page's
   URL (in the address bar), not in its content.
4. Copy .env.example to .env and fill in your Schwab credentials:

       SCHWAB_CLIENT_ID=your_app_key_here
       SCHWAB_CLIENT_SECRET=your_secret_here
       SCHWAB_REDIRECT_URI=https://127.0.0.1

USAGE
    # With .env file (recommended -- keeps secrets out of shell history):
    python3 -m providers.schwab.setup

    # Or, overriding .env values / passing explicitly:
    python3 -m providers.schwab.setup --client-id YOUR_CLIENT_ID --client-secret YOUR_SECRET --redirect-uri https://127.0.0.1

WHAT HAPPENS
1. Prints an authorization URL. Open it in a browser, log into Schwab,
   grant access.
2. You'll land on your redirect_uri with an error-looking page -- copy the
   FULL URL from the address bar (it will contain `?code=...`).
3. Paste that full URL back into this script when prompted.
4. The script extracts the code, exchanges it for tokens, and saves them.

I can't verify this flow against the live Schwab API from my environment --
if a step doesn't match what you actually see (wrong URL format, different
query param name, etc.), that's the API having moved since my knowledge was
current. Note whatever's different and it's an easy fix.
"""

import argparse
import os
import sys
from urllib.parse import urlparse, parse_qs

from providers.schwab.auth import build_authorization_url, exchange_code_for_tokens, SchwabAuthError


def extract_code_from_redirect_url(redirect_url: str) -> str:
    parsed = urlparse(redirect_url.strip())
    query = parse_qs(parsed.query)
    if "code" not in query:
        raise ValueError(
            f"Couldn't find a 'code' parameter in that URL. Make sure you pasted the FULL "
            f"address-bar URL after granting consent (the one that looks like an error page), "
            f"not the authorization URL you opened first."
        )
    return query["code"][0]


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--client-id", default=os.environ.get("SCHWAB_CLIENT_ID"),
                    help="Your Schwab app's client_id (App Key). Defaults to SCHWAB_CLIENT_ID from .env/environment.")
    ap.add_argument("--client-secret", default=os.environ.get("SCHWAB_CLIENT_SECRET"),
                    help="Your Schwab app's client_secret (Secret). Defaults to SCHWAB_CLIENT_SECRET from .env/environment.")
    ap.add_argument("--redirect-uri", default=os.environ.get("SCHWAB_REDIRECT_URI"),
                    help="The callback URL registered for this app, e.g. https://127.0.0.1. Defaults to SCHWAB_REDIRECT_URI from .env/environment.")
    args = ap.parse_args()

    missing = [name for name, val in [("--client-id/SCHWAB_CLIENT_ID", args.client_id),
                                       ("--client-secret/SCHWAB_CLIENT_SECRET", args.client_secret),
                                       ("--redirect-uri/SCHWAB_REDIRECT_URI", args.redirect_uri)] if not val]
    if missing:
        print(f"ERROR: missing required value(s): {', '.join(missing)}")
        print("Either pass them as CLI flags, or create a .env file (see .env.example) with:")
        print("  SCHWAB_CLIENT_ID=...")
        print("  SCHWAB_CLIENT_SECRET=...")
        print("  SCHWAB_REDIRECT_URI=...")
        sys.exit(1)

    auth_url = build_authorization_url(args.client_id, args.redirect_uri)
    print("\n1. Open this URL in a browser and log into Schwab:\n")
    print(f"   {auth_url}\n")
    print("2. Grant access. You'll land on a page at your redirect URI that likely looks")
    print("   like a browser error (\"can't connect\", \"refused to connect\", etc.) -- that's")
    print("   expected, since nothing is actually listening there.\n")
    print("3. Copy the FULL URL from your browser's address bar at that point.\n")

    redirect_url = input("Paste that full URL here and press Enter: ").strip()

    try:
        code = extract_code_from_redirect_url(redirect_url)
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    try:
        exchange_code_for_tokens(args.client_id, args.client_secret, args.redirect_uri, code)
    except SchwabAuthError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\nSuccess -- token saved to cache/schwab_token.json.")
    print("The app will use this automatically and refresh it as needed.")
    print("If Schwab's refresh tokens expire in ~7 days as commonly reported, you'll need")
    print("to run this script again around then.")


if __name__ == "__main__":
    main()