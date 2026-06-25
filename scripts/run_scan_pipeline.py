from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_data_freshness import ensure_freshness
from scripts.fetch_today_bars import fetch_today_bars
from scripts.run_bytecode_daily_push import refresh_reference_cache, run_daily_push, run_prewarm


def main() -> None:
    parser = argparse.ArgumentParser(description="Prewarm data, validate freshness, then run the current daily push.")
    parser.add_argument("--mode", choices=["intraday", "after_hours"], required=True)
    parser.add_argument("--retries", type=int, default=15)
    parser.add_argument("--base-sleep", type=int, default=120)
    args = parser.parse_args()

    current_mode = args.mode
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        print(f"[pipeline] attempt {attempt}/{args.retries} mode={current_mode}")
        try:
            # 1. 拉取当日全市场日线（1次 tushare 调用，绕过逐股配额限制）
            print("[pipeline] fetch today bars (batch)")
            fetch_result = fetch_today_bars()
            print(f"[pipeline] fetch_today_bars: {fetch_result}")

            # 2. 数据源健康预检
            print("[pipeline] prewarm latest data")
            run_prewarm()

            # 3. 探针股票刷新
            print("[pipeline] refresh reference cache")
            refresh_reference_cache()

            # 4. 新鲜度校验
            expected, latest = ensure_freshness(current_mode)
            print(f"[pipeline] freshness_ok latest={latest} expected={expected}")

            # 5. 执行推送
            print("[pipeline] run daily push")
            run_daily_push()
            print("[pipeline] success")
            return
        except Exception as exc:
            last_error = exc
            print(f"[pipeline] attempt {attempt} failed: {exc}")
            traceback.print_exc()
            # 降级兜底：盘后数据延迟，重试到第5次（约等了10分钟）后切换为盘中实时切片兜底
            if args.mode == "after_hours" and current_mode == "after_hours" and attempt >= 5:
                print("[pipeline] WARN after_hours data still unavailable after 5 attempts, "
                      "switching to intraday realtime snapshot fallback")
                current_mode = "intraday"
            if attempt < args.retries:
                sleep_for = args.base_sleep * attempt
                print(f"[pipeline] sleep {sleep_for}s before retry")
                time.sleep(sleep_for)

    raise SystemExit(f"scan pipeline failed after {args.retries} attempts: {last_error}")


if __name__ == "__main__":
    main()
