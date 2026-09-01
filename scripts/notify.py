#!/usr/bin/env python3
"""Send a progress message to Telegram.

Credentials are read from ~/.config/telegram-send.conf (the same config
telegram-send uses); nothing is hardcoded and the token is never printed.

Usage: python3 scripts/notify.py "message"   (or pipe the message on stdin)
"""
import configparser
import os
import sys
import time
import urllib.parse
import urllib.request

CONF = os.path.expanduser("~/.config/telegram-send.conf")
RETRIES = 4
TIMEOUT = 30


def main() -> int:
    text = " ".join(sys.argv[1:]) or sys.stdin.read()
    if not text.strip():
        print("nothing to send", file=sys.stderr)
        return 2

    cfg = configparser.ConfigParser()
    if not cfg.read(CONF):
        print(f"missing config: {CONF}", file=sys.stderr)
        return 2
    token = cfg["telegram"]["token"]
    chat_id = cfg["telegram"]["chat_id"]

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
    ).encode()

    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, data=body, timeout=TIMEOUT) as r:
                if r.status == 200:
                    print("sent")
                    return 0
                print(f"http {r.status}", file=sys.stderr)
        except Exception as exc:  # network flakiness is expected here
            print(f"attempt {attempt}/{RETRIES} failed: {type(exc).__name__}", file=sys.stderr)
        time.sleep(2 * attempt)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
