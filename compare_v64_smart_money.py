import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

from classic_ta.daily_push import BEST_PARAMS
from classic_ta.v64_ambush_model import run_v64_backtest


RESULT_DIR = Path("results") / "daily"
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-06-23"


def to_bs_code(ts_code: str) -> str:
    symbol, market = ts_code.split(".")
    return f"{market.lower()}.{symbol}"


def to_ts_code(bs_code: str) -> str:
    market, symbol = bs_code.split(".")
    return f"{symbol}.{market.upper()}"


def get_sample_stock_codes(sample_size: int, seed: int):
    rs = bs.query_all_stock(day=DEFAULT_END)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)

    code_mask = (
        df["code"].str.startswith("sh.60")
        | df["code"].str.startswith("sh.68")
        | df["code"].str.startswith("sz.00")
        | df["code"].str.startswith("sz.30")
    )
    name_mask = ~df["code_name"].fillna("").str.contains("ST")
    candidates = df[code_mask & name_mask]["code"].tolist()

    rng = random.Random(seed)
    selected = candidates if sample_size >= len(candidates) else rng.sample(candidates, sample_size)
    return [to_ts_code(code) for code in selected]


def fetch_history(ts_code: str, start_date: str, end_date: str):
    fields = "date,open,high,low,close,volume"
    rs = bs.query_history_k_data_plus(
        to_bs_code(ts_code),
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=fields.split(","))
    df["Date"] = pd.to_datetime(df["date"])
    df = df.set_index("Date")
    for col, out_col in [
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ]:
        df[out_col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df["Volume"] > 0]
    if len(df) < 130:
        return None
    return df


def summarize_trades(trades):
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "total_profit": 0.0,
            "avg_hold_days": 0.0,
        }

    profits = np.array([t.profit_pct for t in trades], dtype=float)
    holds = np.array([t.hold_days for t in trades], dtype=float)
    wins = profits > 0
    return {
        "trades": int(len(trades)),
        "win_rate": float(wins.mean() * 100),
        "avg_profit": float(profits.mean()),
        "total_profit": float(profits.sum()),
        "avg_hold_days": float(holds.mean()),
    }


def print_summary(label: str, summary: dict):
    print(
        f"{label:<12} trades={summary['trades']:>4} | "
        f"win_rate={summary['win_rate']:>5.1f}% | "
        f"avg_profit={summary['avg_profit']:>+6.2f}% | "
        f"total_profit={summary['total_profit']:>+7.1f}% | "
        f"avg_hold={summary['avg_hold_days']:>4.1f}d"
    )


def run_comparison(sample_size: int, seed: int, start_date: str, end_date: str, smart_money_min_score: int):
    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login_result.error_msg}")

    try:
        stock_codes = get_sample_stock_codes(sample_size, seed)
        baseline_params = BEST_PARAMS.copy()
        baseline_params["smart_money_structure_enabled"] = False

        enhanced_params = BEST_PARAMS.copy()
        enhanced_params["smart_money_structure_enabled"] = True
        enhanced_params["smart_money_min_score"] = smart_money_min_score

        baseline_trades = []
        enhanced_trades = []
        loaded_codes = []

        started_at = time.time()
        for idx, ts_code in enumerate(stock_codes, start=1):
            df = fetch_history(ts_code, start_date, end_date)
            if df is None:
                continue

            loaded_codes.append(ts_code)
            baseline_trades.extend(run_v64_backtest(df.copy(), params=baseline_params, ts_code=ts_code))
            enhanced_trades.extend(run_v64_backtest(df.copy(), params=enhanced_params, ts_code=ts_code))

            if idx % 25 == 0:
                print(
                    f"progress {idx:>3}/{len(stock_codes)} | "
                    f"loaded={len(loaded_codes):>3} | "
                    f"baseline={len(baseline_trades):>4} | "
                    f"enhanced={len(enhanced_trades):>4}"
                )

        baseline_summary = summarize_trades(baseline_trades)
        enhanced_summary = summarize_trades(enhanced_trades)

        enhanced_keys = {(t.ts_code, str(t.buy_date)) for t in enhanced_trades}
        filtered_trades = [
            t for t in baseline_trades
            if (t.ts_code, str(t.buy_date)) not in enhanced_keys
        ]
        filtered_summary = summarize_trades(filtered_trades)

        elapsed = time.time() - started_at
        print("\nV6.4 current vs smart-money structure")
        print(
            f"sample_size={sample_size} | loaded={len(loaded_codes)} | "
            f"period={start_date}~{end_date} | smart_money_min_score={smart_money_min_score} | elapsed={elapsed:.1f}s"
        )
        print_summary("baseline", baseline_summary)
        print_summary("enhanced", enhanced_summary)
        print_summary("filtered", filtered_summary)

        result = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "sample_size": sample_size,
            "loaded_codes": len(loaded_codes),
            "start_date": start_date,
            "end_date": end_date,
            "smart_money_min_score": smart_money_min_score,
            "baseline": baseline_summary,
            "enhanced": enhanced_summary,
            "filtered_out_by_enhanced": filtered_summary,
            "delta": {
                "trades": enhanced_summary["trades"] - baseline_summary["trades"],
                "win_rate_pp": enhanced_summary["win_rate"] - baseline_summary["win_rate"],
                "avg_profit_pp": enhanced_summary["avg_profit"] - baseline_summary["avg_profit"],
                "total_profit_pp": enhanced_summary["total_profit"] - baseline_summary["total_profit"],
            },
        }

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULT_DIR / f"smart_money_backtest_{sample_size}_{seed}_s{smart_money_min_score}.json"
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved {output_path}")
        return result
    finally:
        bs.logout()


def main():
    parser = argparse.ArgumentParser(description="Compare current V6.4 against smart-money structure enhanced V6.4.")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END)
    parser.add_argument("--smart-money-min-score", type=int, default=3)
    args = parser.parse_args()
    run_comparison(
        args.sample_size,
        args.seed,
        args.start_date,
        args.end_date,
        args.smart_money_min_score,
    )


if __name__ == "__main__":
    main()
