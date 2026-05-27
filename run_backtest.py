import os
import sys
import json
import time
import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("d:/Solo/TradingAgents/.env"))

from tradingagents import TradingAgentsGraph, TradingAgentsConfig

OUTPUT_DIR = Path("d:/Solo/TradingAgents/backtest_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_llm_config():
    config = TradingAgentsConfig(
        llm_provider="litellm",
        deep_think_llm="deepseek/deepseek-chat",
        quick_think_llm="deepseek/deepseek-chat",
        max_debate_rounds=2,
        max_risk_discuss_rounds=1,
        max_recur_limit=100,
        response_language="zh-CN",
    )
    return config

def run_single_analysis(ticker, date_str):
    config = get_llm_config()
    ta = TradingAgentsGraph(debug=True, config=config)
    _, decision = ta.propagate(ticker, date_str)
    return decision

def run_backtest():
    tickers_and_dates = [
        ("NVDA", "2025-05-15"),
        ("AAPL", "2025-05-15"),
        ("TSLA", "2025-05-15"),
    ]

    results = []
    total = len(tickers_and_dates)
    current = 0

    for ticker, date_str in tickers_and_dates:
        current += 1
        print(f"\n{'='*60}")
        print(f"[{current}/{total}] Analyzing {ticker} on {date_str}")
        print(f"{'='*60}")
        try:
            decision = run_single_analysis(ticker, date_str)
            result = {
                "ticker": ticker,
                "date": date_str,
                "decision": str(decision),
                "status": "success",
                "timestamp": datetime.datetime.now().isoformat(),
            }
            print(f"Decision for {ticker} on {date_str}: {str(decision)[:300]}...")
        except Exception as e:
            result = {
                "ticker": ticker,
                "date": date_str,
                "decision": None,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            print(f"Error analyzing {ticker} on {date_str}: {e}")

        results.append(result)

        result_file = OUTPUT_DIR / f"{ticker}_{date_str}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        time.sleep(30)

    summary_file = OUTPUT_DIR / "backtest_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = r["status"]
        ticker = r["ticker"]
        if status == "success":
            dec = r["decision"][:200] if r["decision"] else "N/A"
            print(f"  {ticker}: SUCCESS - {dec}")
        else:
            print(f"  {ticker}: ERROR - {r.get('error', 'Unknown')}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "single" and len(sys.argv) >= 4:
            ticker = sys.argv[2]
            date_str = sys.argv[3]
            print(f"Running single analysis: {ticker} on {date_str}")
            decision = run_single_analysis(ticker, date_str)
            print(f"\nDecision:\n{decision}")
        else:
            print("Usage:")
            print("  python run_backtest.py single TICKER DATE  - Run single analysis")
            print("  python run_backtest.py                     - Run full backtest")
    else:
        run_backtest()
