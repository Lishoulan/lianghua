import sys
import os
import time
import json
import numpy as np
import pandas as pd
import tushare as ts
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
pro = ts.pro_api(TUSHARE_TOKEN)

OUTPUT_FILE = Path(__file__).parent.parent / "results" / "classic_ta_scan" / "uptrend_today.txt"
RESULT_FILE = Path(__file__).parent.parent / "results" / "classic_ta_scan" / "uptrend_today.json"


def log(msg):
    print(msg, flush=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_all_a_stocks():
    try:
        stock_basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,industry,list_date")
        a_stocks = stock_basic[
            (stock_basic["ts_code"].str.endswith(".SH"))
            | (stock_basic["ts_code"].str.endswith(".SZ"))
        ]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("*ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("N")]
        a_stocks = a_stocks[a_stocks["list_date"] < "20250101"]
        return [(row["ts_code"], row["name"], row.get("industry", "")) for _, row in a_stocks.iterrows()]
    except Exception as e:
        log(f"获取股票列表失败: {e}")
        return []


def get_stock_data(ts_code):
    try:
        end_date = pd.Timestamp.now().strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date="20240101", end_date=end_date)
        time.sleep(0.3)
        if df is None or len(df) < 130:
            return None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df.set_index("Date", inplace=True)
        df["Open"] = df["open"].astype(float)
        df["High"] = df["high"].astype(float)
        df["Low"] = df["low"].astype(float)
        df["Close"] = df["close"].astype(float)
        df["Volume"] = df["vol"].astype(float)
        df = df[df["Volume"] > 0]
        if df.empty:
            return None
        return df
    except Exception:
        time.sleep(0.5)
        return None


def compute_indicators(df):
    from classic_ta.candlestick_patterns import run_candlestick_detection
    from classic_ta.volume_price_analysis import run_vpa_analysis
    from classic_ta.wyckoff_analysis import run_wyckoff_analysis
    from classic_ta.buy_signal_detector import run_buy_signal_detection

    df["white_line"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["white_line"] = df["white_line"].ewm(span=10, adjust=False).mean()
    df["ma14"] = df["Close"].rolling(window=14).mean()
    df["ma28"] = df["Close"].rolling(window=28).mean()
    df["ma57"] = df["Close"].rolling(window=57).mean()
    df["ma114"] = df["Close"].rolling(window=114).mean()
    df["yellow_line"] = (df["ma14"] + df["ma28"] + df["ma57"] + df["ma114"]) / 4

    low_9 = df["Low"].rolling(window=9, min_periods=1).min()
    high_9 = df["High"].rolling(window=9, min_periods=1).max()
    rsv = (df["Close"] - low_9) / (high_9 - low_9) * 100
    rsv = rsv.fillna(50)
    df["K"] = rsv.ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    df["J"] = df["J"].clip(0, 100)

    df = run_candlestick_detection(df)
    df = run_vpa_analysis(df)
    df = run_wyckoff_analysis(df)
    df = run_buy_signal_detection(df)
    return df


def scan_uptrend_today():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    log("=" * 80)
    log("🚀 主升浪接力信号 — 今日全市场扫描")
    log(f"扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    all_stocks = get_all_a_stocks()
    if not all_stocks:
        log("❌ 无法获取股票列表")
        return

    total = len(all_stocks)
    log(f"📋 扫描股票数: {total}")

    uptrend_candidates = []
    reversal_candidates = []
    processed = 0
    errors = 0
    start_time = time.time()

    for ts_code, name, industry in all_stocks:
        processed += 1

        if processed % 200 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / processed * (total - processed) if processed > 0 else 0
            log(f"  进度: {processed}/{total} ({processed/total*100:.1f}%) | "
                f"🚀接力:{len(uptrend_candidates)} 🔄反转:{len(reversal_candidates)} | "
                f"失败:{errors} | ETA:{eta:.0f}s")

        df = get_stock_data(ts_code)
        if df is None:
            errors += 1
            continue

        try:
            df = compute_indicators(df)
        except Exception:
            errors += 1
            continue

        if df is None or len(df) < 130:
            errors += 1
            continue

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        if pd.isna(latest.get("yellow_line")) or pd.isna(latest.get("white_line")):
            continue

        change_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100

        if latest.get("uptrend_signal", False):
            info = {
                "code": ts_code,
                "name": name,
                "industry": industry,
                "price": float(latest["Close"]),
                "change_pct": round(float(change_pct), 2),
                "wyckoff_phase": latest["wyckoff_phase"],
                "white_line": round(float(latest["white_line"]), 2),
                "yellow_line": round(float(latest["yellow_line"]), 2),
                "J": round(float(latest["J"]), 1),
                "vol_ratio": round(float(latest.get("vol_ratio", 0)), 2),
                "support": round(float(latest["support_level"]), 2),
                "resistance": round(float(latest["resistance_level"]), 2),
            }
            uptrend_candidates.append(info)
            log(f"  🚀 接力信号: {name} ({ts_code}) 价格:{latest['Close']:.2f} 涨幅:{change_pct:+.2f}% "
                f"阶段:{latest['wyckoff_phase']} 白线:{latest['white_line']:.2f} 黄线:{latest['yellow_line']:.2f} J:{latest['J']:.1f}")

        if latest.get("reversal_signal", False):
            info = {
                "code": ts_code,
                "name": name,
                "industry": industry,
                "price": float(latest["Close"]),
                "change_pct": round(float(change_pct), 2),
                "wyckoff_phase": latest["wyckoff_phase"],
                "white_line": round(float(latest["white_line"]), 2),
                "yellow_line": round(float(latest["yellow_line"]), 2),
                "J": round(float(latest["J"]), 1),
                "vol_ratio": round(float(latest.get("vol_ratio", 0)), 2),
                "support": round(float(latest["support_level"]), 2),
                "resistance": round(float(latest["resistance_level"]), 2),
            }
            reversal_candidates.append(info)
            log(f"  🔄 反转信号: {name} ({ts_code}) 价格:{latest['Close']:.2f} 涨幅:{change_pct:+.2f}% "
                f"阶段:{latest['wyckoff_phase']} 白线:{latest['white_line']:.2f} 黄线:{latest['yellow_line']:.2f} J:{latest['J']:.1f}")

    elapsed = time.time() - start_time

    log("\n" + "=" * 80)
    log("🎯 扫描结果汇总")
    log("=" * 80)
    log(f"扫描总数: {total} | 成功: {processed - errors} | 失败: {errors}")
    log(f"🚀 主升浪接力信号: {len(uptrend_candidates)} 只")
    log(f"🔄 抄底/反转信号: {len(reversal_candidates)} 只")
    log(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    if uptrend_candidates:
        log("\n" + "-" * 80)
        log("🚀 主升浪接力模型候选股")
        log("-" * 80)
        log(f"  {'#':>3} {'名称':<10} {'代码':<12} {'行业':<8} {'价格':>8} {'涨幅':>8} {'阶段':<6} {'白线':>10} {'黄线':>10} {'J':>6}")
        for i, s in enumerate(uptrend_candidates, 1):
            log(f"  {i:>3} {s['name']:<10} {s['code']:<12} {s['industry']:<8} "
                f"{s['price']:>8.2f} {s['change_pct']:>+7.2f}% {s['wyckoff_phase']:<6} "
                f"{s['white_line']:>10.2f} {s['yellow_line']:>10.2f} {s['J']:>6.1f}")

    if reversal_candidates:
        log("\n" + "-" * 80)
        log("🔄 抄底/反转模型候选股")
        log("-" * 80)
        log(f"  {'#':>3} {'名称':<10} {'代码':<12} {'行业':<8} {'价格':>8} {'涨幅':>8} {'阶段':<6} {'白线':>10} {'黄线':>10} {'J':>6}")
        for i, s in enumerate(reversal_candidates, 1):
            log(f"  {i:>3} {s['name']:<10} {s['code']:<12} {s['industry']:<8} "
                f"{s['price']:>8.2f} {s['change_pct']:>+7.02f}% {s['wyckoff_phase']:<6} "
                f"{s['white_line']:>10.2f} {s['yellow_line']:>10.2f} {s['J']:>6.1f}")

    result = {
        "scan_time": datetime.now().isoformat(),
        "total_scanned": processed - errors,
        "uptrend_count": len(uptrend_candidates),
        "reversal_count": len(reversal_candidates),
        "uptrend_candidates": uptrend_candidates,
        "reversal_candidates": reversal_candidates,
    }

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"\n💾 结果已保存: {RESULT_FILE}")


if __name__ == "__main__":
    scan_uptrend_today()
