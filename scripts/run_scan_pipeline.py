from __future__ import annotations

import argparse
import time
import traceback

from scripts.check_data_freshness import ensure_freshness
from scripts.run_bytecode_daily_push import run_daily_push, run_prewarm


def main() -> None:
    parser = argparse.ArgumentParser(description="Prewarm data, validate freshness, then run the current daily push.")
    parser.add_argument("--mode", choices=["intraday", "after_hours"], required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--base-sleep", type=int, default=20)
    args = parser.parse_args()

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        print(f"[pipeline] attempt {attempt}/{args.retries} mode={args.mode}")
        try:
            print("[pipeline] prewarm latest data")
            run_prewarm()

            expected, latest = ensure_freshness(args.mode)
            print(f"[pipeline] freshness_ok latest={latest} expected={expected}")

            print("[pipeline] run daily push")
            run_daily_push()
            print("[pipeline] success")
            return
        except Exception as exc:
            last_error = exc
            print(f"[pipeline] attempt {attempt} failed: {exc}")
            traceback.print_exc()
            if attempt < args.retries:
                sleep_for = args.base_sleep * attempt
                print(f"[pipeline] sleep {sleep_for}s before retry")
                time.sleep(sleep_for)

    raise SystemExit(f"scan pipeline failed after {args.retries} attempts: {last_error}")


if __name__ == "__main__":
    main()
