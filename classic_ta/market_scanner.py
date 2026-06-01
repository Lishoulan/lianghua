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

load_dotenv(Path(__file__).parent.parent.parent / ".env")

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
pro = ts.pro_api(TUSHARE_TOKEN)


def get_all_a_stocks():
    stock_basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,industry,list_date")
    a_stocks = stock_basic[
        (stock_basic["ts_code"].str.endswith(".SH"))
        | (stock_basic["ts_code"].str.endswith(".SZ"))
    ]
    a_stocks = a_stocks[~a_stocks["name"].str.startswith("ST")]
    a_stocks = a_stocks[~a_stocks["name"].str.startswith("*ST")]
    return a_stocks


def get_stock_data_full(ts_code):
    try:
        end_date = pd.Timestamp.now().strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date="20230101", end_date=end_date)
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


def compute_all_indicators(df):
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


def scan_market(max_stocks=None):
    print("=" * 90)
    print("📊 经典技术分析全市场扫描器 — 寻找完美买入信号")
    print("扫描时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 90)
    print("🔍 信号1: 抄底/反转模型 — 底部横盘+放量异动+吞没阳线")
    print("🔍 信号2: 主升浪接力模型 — 突破箱体+良性放量+饱满阳线")
    print("=" * 90)

    stock_list = get_all_a_stocks()
    if max_stocks:
        stock_list = stock_list.head(max_stocks)

    total = len(stock_list)
    print(f"📋 扫描股票数: {total}")

    reversal_candidates = []
    uptrend_candidates = []
    processed = 0
    errors = 0
    start_time = time.time()

    for _, row in stock_list.iterrows():
        ts_code = row["ts_code"]
        name = row["name"]
        industry = row.get("industry", "未知")
        processed += 1

        if processed % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  进度: {processed}/{total} ({processed/total*100:.1f}%) | "
                  f"反转:{len(reversal_candidates)} 接力:{len(uptrend_candidates)} | "
                  f"耗时:{elapsed:.0f}s")

        df = get_stock_data_full(ts_code)
        if df is None:
            errors += 1
            continue

        try:
            df = compute_all_indicators(df)
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

        if latest.get("reversal_signal", False):
            reversal_candidates.append({
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
            })
            print(f"  🔄 反转信号: {name} ({ts_code}) 价格:{latest['Close']:.2f} 涨幅:{change_pct:+.2f}%")

        if latest.get("uptrend_signal", False):
            uptrend_candidates.append({
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
            })
            print(f"  🚀 接力信号: {name} ({ts_code}) 价格:{latest['Close']:.2f} 涨幅:{change_pct:+.2f}%")

    elapsed = time.time() - start_time

    print("\n" + "=" * 90)
    print("🎯 扫描结果汇总")
    print("=" * 90)
    print(f"扫描总数: {total} | 成功: {processed - errors} | 失败: {errors}")
    print(f"抄底/反转信号: {len(reversal_candidates)} 只")
    print(f"主升浪接力信号: {len(uptrend_candidates)} 只")
    print(f"总耗时: {elapsed:.1f}s")

    if reversal_candidates:
        print("\n" + "-" * 90)
        print("🔄 抄底/反转模型候选股")
        print("-" * 90)
        print(f"  {'#':>3} {'名称':<10} {'代码':<12} {'行业':<8} {'价格':>8} {'涨幅':>8} {'阶段':<6} {'白线':>10} {'黄线':>10} {'J':>6}")
        for i, s in enumerate(reversal_candidates, 1):
            print(f"  {i:>3} {s['name']:<10} {s['code']:<12} {s['industry']:<8} "
                  f"{s['price']:>8.2f} {s['change_pct']:>+7.2f}% {s['wyckoff_phase']:<6} "
                  f"{s['white_line']:>10.2f} {s['yellow_line']:>10.2f} {s['J']:>6.1f}")

    if uptrend_candidates:
        print("\n" + "-" * 90)
        print("🚀 主升浪接力模型候选股")
        print("-" * 90)
        print(f"  {'#':>3} {'名称':<10} {'代码':<12} {'行业':<8} {'价格':>8} {'涨幅':>8} {'阶段':<6} {'白线':>10} {'黄线':>10} {'J':>6}")
        for i, s in enumerate(uptrend_candidates, 1):
            print(f"  {i:>3} {s['name']:<10} {s['code']:<12} {s['industry']:<8} "
                  f"{s['price']:>8.2f} {s['change_pct']:>+7.02f}% {s['wyckoff_phase']:<6} "
                  f"{s['white_line']:>10.2f} {s['yellow_line']:>10.2f} {s['J']:>6.1f}")

    result = {
        "scan_time": datetime.now().isoformat(),
        "total_scanned": processed - errors,
        "reversal_count": len(reversal_candidates),
        "uptrend_count": len(uptrend_candidates),
        "reversal_candidates": reversal_candidates,
        "uptrend_candidates": uptrend_candidates,
    }

    output_dir = Path(__file__).parent.parent / "results" / "classic_ta_scan"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"market_scan_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 扫描结果已保存: {report_file}")

    return result


if __name__ == "__main__":
    max_n = None
    if len(sys.argv) > 1:
        try:
            max_n = int(sys.argv[1])
        except ValueError:
            pass
    scan_market(max_stocks=max_n)
