"""Shared setup for the experiment scripts.

Note: load_dotenv() with no argument searches upward from the *script's*
directory, which is fine here, but we pin the path explicitly so the scripts
work no matter where you run them from.
"""
import os
import pathlib
import time

from dotenv import load_dotenv
from google import genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "out"
OUT.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit(f"GEMINI_API_KEY not found in {ROOT / '.env'}")


def make_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class timed:
    """with timed('label') as t: ...   -> t.ms"""

    def __init__(self, label: str):
        self.label = label
        self.ms = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000
        status = "ok" if exc[0] is None else f"FAILED ({exc[0].__name__})"
        print(f"  [{self.label}] {self.ms:,.0f} ms  {status}")
        return False
