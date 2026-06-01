import sys
import os
import json
import time
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


def get_all_a_stocks():
    try:
        stock_basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,industry,list_date")
        a_stocks = stock_basic[
            stock_basic["ts_code"].str.endswith(".SH") | stock_basic["ts_code"].str.endswith(".SZ")
        ]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("*ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("N")]
        a_stocks = a_stocks[a_stocks["list_date"] < "20240101"]
        return [(row["ts_code"], row["name"]) for _, row in a_stocks.iterrows()]
    except Exception:
        return []


def get_oamv_filter(start_date="20200101", end_date="20251231"):
    from ml_strategy.oamv_filter import OAMVHysteresisFilter
    try:
        index_df = pro.index_daily(ts_code="000300.SH", start_date=start_date, end_date=end_date)
        if index_df is None or len(index_df) < 40:
            return None
        index_df = index_df.sort_values("trade_date").reset_index(drop=True)
        index_df["Date"] = pd.to_datetime(index_df["trade_date"], format="%Y%m%d")
        index_df.set_index("Date", inplace=True)
        index_df["Close"] = index_df["close"].astype(float)
        index_df["Volume"] = index_df["vol"].astype(float)
        index_df["amount"] = index_df["amount"].astype(float)

        oamv = OAMVHysteresisFilter(
            upper_threshold=4.0, lower_threshold=-2.3, cost_ma_period=34,
            roc_period=1, weekly_ema_period=5, weekly_use_ema=True,
        )
        oamv.fit(index_df)
        state_df = oamv.get_state_df()
        if state_df is None:
            return None
        state_df["oamv_allowed"] = state_df["oamv_state"] == 1
        weekly_allowed = {}
        for date in state_df.index:
            weekly_allowed[date] = oamv.is_trading_allowed(date, require_weekly=True)
        state_df["weekly_allowed"] = pd.Series(weekly_allowed)
        state_df["weekly_allowed"] = state_df["weekly_allowed"].fillna(False)
        return state_df
    except Exception as e:
        print(f"  ⚠️ OAMV滤波器计算失败: {e}")
        return None


def get_stock_data(ts_code, start_date="20200101", end_date="20251231"):
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
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
        time.sleep(0.3)
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


def backtest_portfolio(stock_data, signal_col, initial_cash=1000000, max_positions=5, market_filter=None):
    cash = initial_cash
    holdings = {}
    trades = []
    equity_curve = []
    all_dates = sorted(set().union(*[s.index for s in stock_data.values()]))

    for date in all_dates:
        daily_signals = []
        for ts_code, df in stock_data.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row.get("white_line")):
                continue
            if ts_code in holdings:
                continue
            if row.get(signal_col, False):
                allow_buy = True
                if market_filter is not None and date in market_filter.index:
                    mkt = market_filter.loc[date]
                    if not mkt.get("weekly_allowed", True):
                        allow_buy = False
                if allow_buy:
                    daily_signals.append((ts_code, row["Open"]))

        if len(daily_signals) > 0 and len(holdings) < max_positions:
            slots = max_positions - len(holdings)
            selected = daily_signals[:slots]
            per_stock_cash = cash / (len(holdings) + len(selected)) if (len(holdings) + len(selected)) > 0 else cash
            for ts_code, buy_price in selected:
                if pd.isna(buy_price) or buy_price <= 0:
                    continue
                shares = int(per_stock_cash * 0.95 / buy_price)
                if shares > 0:
                    cost = shares * buy_price
                    if cost > cash:
                        shares = int(cash * 0.95 / buy_price)
                        cost = shares * buy_price
                    if shares > 0:
                        cash -= cost
                        holdings[ts_code] = {
                            "shares": shares, "entry_price": buy_price, "entry_date": date,
                            "hold_days": 0, "max_profit_pct": 0.0,
                            "trailing_active": False, "breakeven_active": False,
                        }

        to_sell = []
        for ts_code, h in holdings.items():
            df = stock_data.get(ts_code)
            if df is None or date not in df.index:
                continue
            row = df.loc[date]
            current_price = row["Close"]
            if pd.isna(current_price):
                continue

            h["hold_days"] += 1
            high_pct = (row["High"] - h["entry_price"]) / h["entry_price"]
            pnl_pct = (current_price - h["entry_price"]) / h["entry_price"]

            if high_pct > h["max_profit_pct"]:
                h["max_profit_pct"] = high_pct
            if h["max_profit_pct"] >= 0.15:
                h["trailing_active"] = True
            if h["max_profit_pct"] >= 0.08:
                h["breakeven_active"] = True

            current_stop = 0.01 if h["breakeven_active"] else -0.07
            exit_reason = None

            if pnl_pct <= current_stop:
                exit_reason = "保本止损" if h["breakeven_active"] and current_stop > 0 else "硬止损"
            elif h["trailing_active"] and h["max_profit_pct"] - pnl_pct >= 0.03:
                exit_reason = "追踪止盈"
            else:
                vol_ma5 = df["Volume"].iloc[max(0, df.index.get_loc(date) - 5):df.index.get_loc(date)].mean() if df.index.get_loc(date) >= 5 else df["Volume"].iloc[:df.index.get_loc(date) + 1].mean()
                body = abs(row["Close"] - row["Open"])
                upper_shadow = row["High"] - max(row["Close"], row["Open"])
                vpa_top = False
                if row["Volume"] > vol_ma5 * 2 and body > 0:
                    if upper_shadow > body * 1.5:
                        vpa_top = True
                    idx = df.index.get_loc(date)
                    if idx > 0:
                        prev = df.iloc[idx - 1]
                        if row["Close"] < row["Open"] and prev["Close"] > prev["Open"]:
                            if row["Open"] > prev["Close"] and row["Close"] < prev["Open"]:
                                vpa_top = True
                if vpa_top:
                    exit_reason = "VPA见顶"

            if exit_reason is None:
                cross_down = False
                idx = df.index.get_loc(date)
                if idx > 0:
                    prev_row = df.iloc[idx - 1]
                    cross_down = (row["white_line"] < row["yellow_line"]) and (prev_row["white_line"] >= prev_row["yellow_line"])
                if h["hold_days"] >= 10:
                    if pnl_pct < 0:
                        exit_reason = "时间止损"
                    elif pnl_pct >= 0 and not cross_down and h["hold_days"] < 20:
                        pass
                    else:
                        exit_reason = "死叉止损" if cross_down else "超时平仓"
                if exit_reason is None and h["hold_days"] >= 20:
                    exit_reason = "超时平仓"
                if exit_reason is None and cross_down:
                    exit_reason = "死叉止损"

            if exit_reason:
                cash += h["shares"] * current_price
                stock_name = stock_names.get(ts_code, ts_code)
                trades.append({
                    "stock": stock_name, "code": ts_code,
                    "buy_date": h["entry_date"].strftime("%Y-%m-%d"),
                    "sell_date": date.strftime("%Y-%m-%d"),
                    "buy_price": round(float(h["entry_price"]), 2),
                    "sell_price": round(float(current_price), 2),
                    "shares": h["shares"], "hold_days": h["hold_days"],
                    "profit_pct": round(float(pnl_pct * 100), 2),
                    "max_profit_pct": round(float(h["max_profit_pct"] * 100), 2),
                    "exit_reason": exit_reason,
                    "signal_type": "抄底反转" if signal_col == "reversal_signal" else "主升浪接力",
                })
                to_sell.append(ts_code)

        for ts_code in to_sell:
            del holdings[ts_code]

        portfolio_value = cash
        for ts_code, h in holdings.items():
            df = stock_data.get(ts_code)
            if df is not None and date in df.index:
                portfolio_value += h["shares"] * df.loc[date, "Close"]
        equity_curve.append(portfolio_value)

    for ts_code, h in list(holdings.items()):
        df = stock_data.get(ts_code)
        if df is not None and len(df) > 0:
            final_price = df.iloc[-1]["Close"]
            pnl_pct = (final_price - h["entry_price"]) / h["entry_price"]
            cash += h["shares"] * final_price
            stock_name = stock_names.get(ts_code, ts_code)
            trades.append({
                "stock": stock_name, "code": ts_code,
                "buy_date": h["entry_date"].strftime("%Y-%m-%d"),
                "sell_date": df.index[-1].strftime("%Y-%m-%d"),
                "buy_price": round(float(h["entry_price"]), 2),
                "sell_price": round(float(final_price), 2),
                "shares": h["shares"], "hold_days": h["hold_days"],
                "profit_pct": round(float(pnl_pct * 100), 2),
                "max_profit_pct": round(float(h["max_profit_pct"] * 100), 2),
                "exit_reason": "回测结束",
                "signal_type": "抄底反转" if signal_col == "reversal_signal" else "主升浪接力",
            })

    total_return = (cash - initial_cash) / initial_cash * 100
    max_drawdown = 0.0
    if len(equity_curve) > 0:
        eq = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_drawdown = float(np.max(drawdown)) * 100

    return trades, total_return, cash, max_drawdown


stock_names = {}


def run_full_market_backtest(start_date="20200101", end_date="20251231"):
    global stock_names

    print("=" * 100)
    print("📊 全市场回测 v3.0 — 两阶段扫描")
    print(f"回测区间: {start_date} ~ {end_date}")
    print("=" * 100)

    all_stocks = get_all_a_stocks()
    if not all_stocks:
        print("❌ 无法获取全市场股票列表")
        return None
    print(f"📋 全A股数量: {len(all_stocks)} 只（已过滤ST、次新股）")

    for code, name in all_stocks:
        stock_names[code] = name

    print("\n📈 计算OAMV活跃市值迟滞滤波器...")
    market_filter = get_oamv_filter(start_date, end_date)
    if market_filter is not None:
        allowed = market_filter["weekly_allowed"]
        print(f"  OAMV数据: {len(market_filter)}天, 允许开仓: {allowed.sum()}天 ({allowed.sum()/len(allowed)*100:.1f}%)")
    else:
        print("  ⚠️ OAMV计算失败, 跳过大盘择时")

    print("\n" + "=" * 100)
    print("🔍 阶段1: 全市场扫描 — 找出有信号的股票")
    print("=" * 100)

    signal_stocks = {}
    processed = 0
    errors = 0
    start_time = time.time()

    for ts_code, name in all_stocks:
        processed += 1
        if processed % 200 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / processed * (len(all_stocks) - processed)
            print(f"  扫描进度: {processed}/{len(all_stocks)} ({processed/len(all_stocks)*100:.1f}%) | "
                  f"有信号: {len(signal_stocks)} | 失败: {errors} | ETA: {eta:.0f}s")

        df = get_stock_data(ts_code, start_date, end_date)
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

        has_rev = df["reversal_signal"].sum() > 0
        has_up = df["uptrend_signal"].sum() > 0

        if has_rev or has_up:
            signal_stocks[ts_code] = df

    elapsed = time.time() - start_time
    print(f"\n✅ 阶段1完成! 耗时: {elapsed:.0f}s")
    print(f"  扫描总数: {processed} | 成功: {processed - errors} | 失败: {errors}")
    print(f"  有信号的股票: {len(signal_stocks)} 只")

    if not signal_stocks:
        print("❌ 没有找到任何有信号的股票")
        return None

    print("\n" + "=" * 100)
    print("📊 阶段2: 组合回测 — 对有信号的股票做资金管理回测")
    print(f"  股票数: {len(signal_stocks)} | 初始资金: ¥1,000,000 | 最多持仓: 5只")
    print("=" * 100)

    for signal_col, signal_name in [("reversal_signal", "🔄 抄底/反转"), ("uptrend_signal", "🚀 主升浪接力")]:
        print(f"\n{'─' * 100}")
        print(f"  {signal_name}模型 — 全市场组合回测")
        print(f"{'─' * 100}")

        trades, total_return, final_cash, max_dd = backtest_portfolio(
            signal_stocks, signal_col, initial_cash=1000000, max_positions=5, market_filter=market_filter
        )

        if not trades:
            print("  无交易记录")
            continue

        total_trades = len(trades)
        win_trades = [t for t in trades if t["profit_pct"] > 0]
        lose_trades = [t for t in trades if t["profit_pct"] <= 0]
        win_rate = len(win_trades) / total_trades * 100

        avg_profit = np.mean([t["profit_pct"] for t in trades])
        avg_win = np.mean([t["profit_pct"] for t in win_trades]) if win_trades else 0
        avg_lose = np.mean([t["profit_pct"] for t in lose_trades]) if lose_trades else 0
        avg_hold = np.mean([t["hold_days"] for t in trades])
        avg_max_profit = np.mean([t.get("max_profit_pct", 0) for t in trades])

        exit_counts = {}
        for t in trades:
            exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1

        print(f"  总交易次数: {total_trades}")
        print(f"  盈利次数: {len(win_trades)} | 亏损次数: {len(lose_trades)}")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  平均每笔收益: {avg_profit:+.2f}%")
        print(f"  平均盈利: {avg_win:+.2f}% | 平均亏损: {avg_lose:+.2f}%")
        print(f"  盈亏比: {abs(avg_win / avg_lose):.2f}" if avg_lose != 0 else "  盈亏比: ∞")
        print(f"  平均持仓天数: {avg_hold:.1f}")
        print(f"  平均最大浮盈: {avg_max_profit:+.2f}%")
        print(f"  总收益率: {total_return:+.2f}%")
        print(f"  最大回撤: {max_dd:.2f}%")
        print(f"  退出原因分布: {dict(exit_counts)}")

        print(f"\n  {'#':>3} {'股票':<10} {'买入日':<12} {'卖出日':<12} {'买价':>8} {'卖价':>8} {'收益%':>8} {'最高浮盈':>8} {'持仓天':>6} {'退出原因':<8}")
        print(f"  {'─'*95}")
        for i, t in enumerate(trades[:30], 1):
            max_p = t.get('max_profit_pct', 0)
            print(f"  {i:>3} {t['stock']:<10} {t['buy_date']:<12} {t['sell_date']:<12} "
                  f"{t['buy_price']:>8.2f} {t['sell_price']:>8.2f} {t['profit_pct']:>+7.2f}% "
                  f"{max_p:>+7.2f}% {t['hold_days']:>6} {t['exit_reason']:<8}")
        if len(trades) > 30:
            print(f"  ... 还有 {len(trades) - 30} 笔交易未显示")

    result = {
        "version": "3.0-fullmarket",
        "backtest_time": datetime.now().isoformat(),
        "period": f"{start_date}~{end_date}",
        "total_stocks_scanned": processed - errors,
        "signal_stocks": len(signal_stocks),
    }

    output_dir = Path(__file__).parent.parent / "results" / "classic_ta_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"fullmarket_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {report_file}")

    return result


if __name__ == "__main__":
    run_full_market_backtest()
