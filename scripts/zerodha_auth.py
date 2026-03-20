#!/usr/bin/env python3
"""
Helper for generating a Zerodha access token for live universe discovery.

Usage:
  python3 scripts/zerodha_auth.py --api-key ... --api-secret ...
  python3 scripts/zerodha_auth.py --api-key ... --api-secret ... --request-token ...
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_universe import build_zerodha_login_url, generate_zerodha_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Zerodha access token.")
    parser.add_argument("--api-key", default=os.getenv("ZERODHA_API_KEY", ""))
    parser.add_argument("--api-secret", default=os.getenv("ZERODHA_API_SECRET", ""))
    parser.add_argument(
        "--request-token",
        default="",
        help="Request token returned by Zerodha after login redirect.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("Missing --api-key or ZERODHA_API_KEY", file=sys.stderr)
        return 1

    if not args.request_token:
        print("1. Open this URL in a browser and complete the Zerodha login flow:\n")
        print(build_zerodha_login_url(args.api_key))
        print("\n2. Copy the `request_token` from the redirect URL.")
        print("3. Re-run this script with --request-token <value>.")
        return 0

    if not args.api_secret:
        print("Missing --api-secret or ZERODHA_API_SECRET", file=sys.stderr)
        return 1

    session = generate_zerodha_session(
        api_key=args.api_key,
        api_secret=args.api_secret,
        request_token=args.request_token,
    )
    access_token = session.get("access_token", "")
    if not access_token:
        print(f"Unexpected session payload: {session}", file=sys.stderr)
        return 1

    print("Access token generated.\n")
    print(f"access_token: {access_token}")
    if session.get("user_name"):
        print(f"user_name: {session['user_name']}")
    if session.get("user_id"):
        print(f"user_id: {session['user_id']}")

    print("\nExport these before running the app:\n")
    print(f"export ZERODHA_API_KEY='{args.api_key}'")
    print(f"export ZERODHA_ACCESS_TOKEN='{access_token}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
