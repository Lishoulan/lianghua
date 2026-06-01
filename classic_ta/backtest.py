import sys
import os
import json
import time
import random
import numpy as np
import pandas as pd
import tushare as ts
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
pro = ts.pro_api(TUSHARE_TOKEN)

BACKTEST_STOCKS = [
    ("600519.SH", "贵州茅台"),
    ("601318.SH", "中国平安"),
    ("600036.SH", "招商银行"),
    ("000001.SZ", "平安银行"),
    ("000333.SZ", "美的集团"),
    ("002415.SZ", "海康威视"),
    ("300750.SZ", "宁德时代"),
    ("300059.SZ", "东方财富"),
    ("600570.SH", "恒生电子"),
    ("601899.SH", "紫金矿业"),
    ("000858.SZ", "五粮液"),
    ("000651.SZ", "格力电器"),
    ("601668.SH", "中国建筑"),
    ("600362.SH", "江西铜业"),
    ("601888.SH", "中国中免"),
    ("002230.SZ", "科大讯飞"),
    ("600900.SH", "长江电力"),
    ("601012.SH", "隆基绿能"),
    ("000002.SZ", "万科A"),
    ("600276.SH", "恒瑞医药"),
]


def get_backtest_data(ts_code, start_date="20200101", end_date="20251231"):
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
        time.sleep(0.5)
        return None


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
            upper_threshold=4.0,
            lower_threshold=-2.3,
            cost_ma_period=34,
            roc_period=1,
            weekly_ema_period=5,
            weekly_use_ema=True,
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


def backtest_single_stock(df, signal_col, initial_cash=100000, market_filter=None):
    cash = initial_cash
    position = 0
    entry_price = 0
    entry_idx = 0
    hold_days = 0
    max_profit_pct = 0.0
    trailing_active = False
    breakeven_active = False
    trades = []
    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        current_price = row["Close"]
        current_open = row["Open"]

        if pd.isna(current_price) or pd.isna(row.get("white_line")):
            equity_curve.append(cash + position * current_price if position > 0 else cash)
            continue

        if position > 0:
            hold_days += 1
            high_pct = (row["High"] - entry_price) / entry_price
            pnl_pct = (current_price - entry_price) / entry_price

            if high_pct > max_profit_pct:
                max_profit_pct = high_pct

            if max_profit_pct >= 0.15:
                trailing_active = True

            if max_profit_pct >= 0.08:
                breakeven_active = True

            current_stop = 0.01 if breakeven_active else -0.07

            if pnl_pct <= current_stop:
                exit_reason = "保本止损" if breakeven_active and current_stop > 0 else "硬止损"
                cash += position * current_price
                trades.append({
                    "buy_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                    "sell_date": df.index[i].strftime("%Y-%m-%d"),
                    "buy_price": round(float(entry_price), 2),
                    "sell_price": round(float(current_price), 2),
                    "shares": position,
                    "hold_days": hold_days,
                    "profit_pct": round(float(pnl_pct * 100), 2),
                    "max_profit_pct": round(float(max_profit_pct * 100), 2),
                    "exit_reason": exit_reason,
                })
                position = 0
                entry_price = 0
                hold_days = 0
                max_profit_pct = 0.0
                trailing_active = False
                breakeven_active = False
                equity_curve.append(cash)
                continue

            if trailing_active and max_profit_pct - pnl_pct >= 0.03:
                cash += position * current_price
                trades.append({
                    "buy_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                    "sell_date": df.index[i].strftime("%Y-%m-%d"),
                    "buy_price": round(float(entry_price), 2),
                    "sell_price": round(float(current_price), 2),
                    "shares": position,
                    "hold_days": hold_days,
                    "profit_pct": round(float(pnl_pct * 100), 2),
                    "max_profit_pct": round(float(max_profit_pct * 100), 2),
                    "exit_reason": "追踪止盈",
                })
                position = 0
                entry_price = 0
                hold_days = 0
                max_profit_pct = 0.0
                trailing_active = False
                breakeven_active = False
                equity_curve.append(cash)
                continue

            vol_ma5 = df["Volume"].iloc[max(0, i - 5):i].mean() if i >= 5 else df["Volume"].iloc[:i + 1].mean()
            body = abs(row["Close"] - row["Open"])
            upper_shadow = row["High"] - max(row["Close"], row["Open"])
            vpa_top = False

            if row["Volume"] > vol_ma5 * 2 and body > 0:
                if upper_shadow > body * 1.5:
                    vpa_top = True
                if i > 0:
                    prev = df.iloc[i - 1]
                    if row["Close"] < row["Open"] and prev["Close"] > prev["Open"]:
                        if row["Open"] > prev["Close"] and row["Close"] < prev["Open"]:
                            vpa_top = True

            if vpa_top:
                cash += position * current_price
                trades.append({
                    "buy_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                    "sell_date": df.index[i].strftime("%Y-%m-%d"),
                    "buy_price": round(float(entry_price), 2),
                    "sell_price": round(float(current_price), 2),
                    "shares": position,
                    "hold_days": hold_days,
                    "profit_pct": round(float(pnl_pct * 100), 2),
                    "max_profit_pct": round(float(max_profit_pct * 100), 2),
                    "exit_reason": "VPA见顶",
                })
                position = 0
                entry_price = 0
                hold_days = 0
                max_profit_pct = 0.0
                trailing_active = False
                breakeven_active = False
                equity_curve.append(cash)
                continue

            cross_down = False
            if i > 0:
                cross_down = (row["white_line"] < row["yellow_line"]) and (
                    df.iloc[i - 1]["white_line"] >= df.iloc[i - 1]["yellow_line"]
                )

            if hold_days >= 10:
                if pnl_pct < 0:
                    cash += position * current_price
                    trades.append({
                        "buy_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                        "sell_date": df.index[i].strftime("%Y-%m-%d"),
                        "buy_price": round(float(entry_price), 2),
                        "sell_price": round(float(current_price), 2),
                        "shares": position,
                        "hold_days": hold_days,
                        "profit_pct": round(float(pnl_pct * 100), 2),
                        "max_profit_pct": round(float(max_profit_pct * 100), 2),
                        "exit_reason": "时间止损",
                    })
                    position = 0
                    entry_price = 0
                    hold_days = 0
                    max_profit_pct = 0.0
                    trailing_active = False
                    breakeven_active = False
                    equity_curve.append(cash)
                    continue
                elif pnl_pct >= 0 and not cross_down and hold_days < 20:
                    pass
                else:
                    cash += position * current_price
                    exit_reason = "死叉止损" if cross_down else "超时平仓"
                    trades.append({
                        "buy_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                        "sell_date": df.index[i].strftime("%Y-%m-%d"),
                        "buy_price": round(float(entry_price), 2),
                        "sell_price": round(float(current_price), 2),
                        "shares": position,
                        "hold_days": hold_days,
                        "profit_pct": round(float(pnl_pct * 100), 2),
                        "max_profit_pct": round(float(max_profit_pct * 100), 2),
                        "exit_reason": exit_reason,
                    })
                    position = 0
                    entry_price = 0
                    hold_days = 0
                    max_profit_pct = 0.0
                    trailing_active = False
                    breakeven_active = False
                    equity_curve.append(cash)
                    continue

            if hold_days >= 20:
                cash += position * current_price
                trades.append({
                    "buy_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                    "sell_date": df.index[i].strftime("%Y-%m-%d"),
                    "buy_price": round(float(entry_price), 2),
                    "sell_price": round(float(current_price), 2),
                    "shares": position,
                    "hold_days": hold_days,
                    "profit_pct": round(float(pnl_pct * 100), 2),
                    "max_profit_pct": round(float(max_profit_pct * 100), 2),
                    "exit_reason": "超时平仓",
                })
                position = 0
                entry_price = 0
                hold_days = 0
                max_profit_pct = 0.0
                trailing_active = False
                breakeven_active = False
                equity_curve.append(cash)
                continue

            if cross_down:
                cash += position * current_price
                trades.append({
                    "buy_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                    "sell_date": df.index[i].strftime("%Y-%m-%d"),
                    "buy_price": round(float(entry_price), 2),
                    "sell_price": round(float(current_price), 2),
                    "shares": position,
                    "hold_days": hold_days,
                    "profit_pct": round(float(pnl_pct * 100), 2),
                    "max_profit_pct": round(float(max_profit_pct * 100), 2),
                    "exit_reason": "死叉止损",
                })
                position = 0
                entry_price = 0
                hold_days = 0
                max_profit_pct = 0.0
                trailing_active = False
                breakeven_active = False
                equity_curve.append(cash)
                continue

        if position == 0 and row.get(signal_col, False):
            allow_buy = True
            if market_filter is not None:
                cur_date = df.index[i]
                if cur_date in market_filter.index:
                    mkt = market_filter.loc[cur_date]
                    if not mkt.get("weekly_allowed", True):
                        allow_buy = False

            if allow_buy:
                buy_price = current_open if i + 1 <= len(df) else current_price
                shares = int(cash * 0.95 / buy_price)
                if shares > 0:
                    position = shares
                    entry_price = buy_price
                    entry_idx = i
                    hold_days = 0
                    max_profit_pct = 0.0
                    trailing_active = False
                    breakeven_active = False
                    cash -= shares * buy_price

        equity_curve.append(cash + position * current_price if position > 0 else cash)

    if position > 0:
        final_price = df.iloc[-1]["Close"]
        pnl_pct = (final_price - entry_price) / entry_price
        cash += position * final_price
        trades.append({
            "buy_date": df.index[entry_idx].strftime("%Y-%m-%d"),
            "sell_date": df.index[-1].strftime("%Y-%m-%d"),
            "buy_price": round(float(entry_price), 2),
            "sell_price": round(float(final_price), 2),
            "shares": position,
            "hold_days": hold_days,
            "profit_pct": round(float(pnl_pct * 100), 2),
            "max_profit_pct": round(float(max_profit_pct * 100), 2),
            "exit_reason": "回测结束",
        })

    total_return = (cash - initial_cash) / initial_cash * 100

    max_drawdown = 0.0
    if len(equity_curve) > 0:
        eq = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        drawdown = (peak - eq) / peak
        max_drawdown = float(np.max(drawdown)) * 100

    return trades, total_return, cash, max_drawdown


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
                            "shares": shares,
                            "entry_price": buy_price,
                            "entry_date": date,
                            "hold_days": 0,
                            "max_profit_pct": 0.0,
                            "trailing_active": False,
                            "breakeven_active": False,
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
                    cross_down = (row["white_line"] < row["yellow_line"]) and (
                        prev_row["white_line"] >= prev_row["yellow_line"]
                    )

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
                stock_name = BACKTEST_STOCKS_DICT.get(ts_code, ts_code)
                trades.append({
                    "stock": stock_name,
                    "code": ts_code,
                    "buy_date": h["entry_date"].strftime("%Y-%m-%d"),
                    "sell_date": date.strftime("%Y-%m-%d"),
                    "buy_price": round(float(h["entry_price"]), 2),
                    "sell_price": round(float(current_price), 2),
                    "shares": h["shares"],
                    "hold_days": h["hold_days"],
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
            stock_name = BACKTEST_STOCKS_DICT.get(ts_code, ts_code)
            trades.append({
                "stock": stock_name,
                "code": ts_code,
                "buy_date": h["entry_date"].strftime("%Y-%m-%d"),
                "sell_date": df.index[-1].strftime("%Y-%m-%d"),
                "buy_price": round(float(h["entry_price"]), 2),
                "sell_price": round(float(final_price), 2),
                "shares": h["shares"],
                "hold_days": h["hold_days"],
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


BACKTEST_STOCKS_DICT = {code: name for code, name in BACKTEST_STOCKS}


def get_random_stocks(n=50):
    try:
        stock_basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
        a_stocks = stock_basic[
            stock_basic["ts_code"].str.endswith(".SH") | stock_basic["ts_code"].str.endswith(".SZ")
        ]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("*ST")]
        known_codes = {code for code, _ in BACKTEST_STOCKS}
        cold = a_stocks[~a_stocks["ts_code"].isin(known_codes)]
        sampled = cold.sample(n=min(n, len(cold)), random_state=42)
        return [(row["ts_code"], row["name"]) for _, row in sampled.iterrows()]
    except Exception:
        return []


def run_backtest(start_date="20200101", end_date="20251231", stocks=None, use_market_filter=True, mode="single"):
    if stocks is None:
        stocks = BACKTEST_STOCKS

    print("=" * 100)
    print("📊 经典技术分析信号回测系统 v3.0")
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"回测模式: {'组合回测(等权分配)' if mode == 'portfolio' else '单票独立回测'}")
    print(f"回测股票: {len(stocks)} 只")
    print(f"大盘择时: {'✅ OAMV活跃市值迟滞滤波+周线滤波' if use_market_filter else '❌ 无择时'}")
    print(f"退出机制: 硬止损-7% | 追踪止盈(15%触发,回撤3%) | 保本保护(8%触发→止损+1%)")
    print(f"          智能时间止损(10日亏平,20日强平) | VPA见顶强平 | 死叉清仓")
    if mode == "portfolio":
        print(f"组合管理: 最多同时持有5只, 等权分配, 初始资金¥1,000,000")
    print("=" * 100)

    market_filter = None
    if use_market_filter:
        print("\n📈 计算OAMV活跃市值迟滞滤波器(大盘择时)...")
        market_filter = get_oamv_filter(start_date, end_date)
        if market_filter is not None:
            allowed = market_filter["weekly_allowed"]
            allowed_pct = allowed.sum() / len(allowed) * 100
            print(f"  OAMV数据: {len(market_filter)}天, 允许开仓天数: {allowed.sum()}天 ({allowed_pct:.1f}%)")
        else:
            print("  ⚠️ OAMV滤波器计算失败, 跳过大盘择时")

    stock_data = {}
    processed = 0
    errors = 0
    start_time = time.time()
    is_large = len(stocks) > 100

    for ts_code, name in stocks:
        processed += 1

        if is_large:
            if processed % 100 == 0 or processed == len(stocks):
                elapsed = time.time() - start_time
                eta = elapsed / processed * (len(stocks) - processed) if processed > 0 else 0
                print(f"  进度: {processed}/{len(stocks)} ({processed/len(stocks)*100:.1f}%) | "
                      f"成功: {len(stock_data)} | 失败: {errors} | "
                      f"耗时: {elapsed:.0f}s | ETA: {eta:.0f}s")
        else:
            print(f"\n[{processed}/{len(stocks)}] {name} ({ts_code})")

        df = get_backtest_data(ts_code, start_date, end_date)
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

        stock_data[ts_code] = df
        rev_count = df["reversal_signal"].sum()
        up_count = df["uptrend_signal"].sum()
        print(f"  🔄 反转信号: {rev_count}次 | 🚀 接力信号: {up_count}次")

    elapsed_data = time.time() - start_time
    print(f"\n⏱️ 数据准备耗时: {elapsed_data:.1f}s | 成功: {len(stock_data)} | 失败: {errors}")

    if mode == "portfolio":
        print("\n" + "=" * 100)
        print("📊 组合回测模式")
        print("=" * 100)

        for signal_col, signal_name in [("reversal_signal", "🔄 抄底/反转"), ("uptrend_signal", "🚀 主升浪接力")]:
            print(f"\n{'─' * 100}")
            print(f"  {signal_name}模型 — 组合回测")
            print(f"{'─' * 100}")

            trades, total_return, final_cash, max_dd = backtest_portfolio(
                stock_data, signal_col, initial_cash=1000000, max_positions=5, market_filter=market_filter
            )

            _print_trade_stats(trades, total_return, max_dd)

        all_trades_rev, all_ret_rev, _, all_dd_rev = backtest_portfolio(
            stock_data, "reversal_signal", initial_cash=1000000, max_positions=5, market_filter=market_filter
        )
        all_trades_up, all_ret_up, _, all_dd_up = backtest_portfolio(
            stock_data, "uptrend_signal", initial_cash=1000000, max_positions=5, market_filter=market_filter
        )

        result = {
            "version": "3.0",
            "mode": "portfolio",
            "backtest_time": datetime.now().isoformat(),
            "period": f"{start_date}~{end_date}",
            "market_filter": use_market_filter,
            "reversal_trades": all_trades_rev,
            "reversal_return": round(all_ret_rev, 2),
            "reversal_max_dd": round(all_dd_rev, 2),
            "uptrend_trades": all_trades_up,
            "uptrend_return": round(all_ret_up, 2),
            "uptrend_max_dd": round(all_dd_up, 2),
        }

    else:
        all_reversal_trades = []
        all_uptrend_trades = []
        stock_summaries = []

        for ts_code, df in stock_data.items():
            name = BACKTEST_STOCKS_DICT.get(ts_code, ts_code)

            rev_trades, rev_return, rev_cash, rev_dd = backtest_single_stock(
                df, "reversal_signal", market_filter=market_filter
            )
            up_trades, up_return, up_cash, up_dd = backtest_single_stock(
                df, "uptrend_signal", market_filter=market_filter
            )

            for t in rev_trades:
                t["stock"] = name
                t["code"] = ts_code
                t["signal_type"] = "抄底反转"
            for t in up_trades:
                t["stock"] = name
                t["code"] = ts_code
                t["signal_type"] = "主升浪接力"

            all_reversal_trades.extend(rev_trades)
            all_uptrend_trades.extend(up_trades)

            stock_summaries.append({
                "code": ts_code,
                "name": name,
                "reversal_trades": len(rev_trades),
                "reversal_return": round(rev_return, 2),
                "reversal_max_dd": round(rev_dd, 2),
                "uptrend_trades": len(up_trades),
                "uptrend_return": round(up_return, 2),
                "uptrend_max_dd": round(up_dd, 2),
            })

        print("\n" + "=" * 100)
        print("📋 回测结果汇总 v3.0")
        print("=" * 100)

        _print_signal_report("🔄 抄底/反转模型", all_reversal_trades, stock_summaries, "reversal")
        _print_signal_report("🚀 主升浪接力模型", all_uptrend_trades, stock_summaries, "uptrend")

        result = {
            "version": "3.0",
            "mode": "single",
            "backtest_time": datetime.now().isoformat(),
            "period": f"{start_date}~{end_date}",
            "market_filter": use_market_filter,
            "reversal_trades": all_reversal_trades,
            "uptrend_trades": all_uptrend_trades,
            "stock_summaries": stock_summaries,
        }

    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.1f}s")

    output_dir = Path(__file__).parent.parent / "results" / "classic_ta_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"backtest_v3_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"💾 回测结果已保存: {report_file}")

    return result


def run_oos_test(start_date="20200101", end_date="20251231", n_stocks=50):
    print("=" * 100)
    print("📊 样本外测试 (Out-of-Sample) — 随机冷门股盲测")
    print("=" * 100)

    random_stocks = get_random_stocks(n_stocks)
    if not random_stocks:
        print("❌ 无法获取随机股票列表")
        return None

    print(f"📋 随机选取 {len(random_stocks)} 只冷门股")
    for code, name in random_stocks[:10]:
        print(f"  {code} {name}")
    if len(random_stocks) > 10:
        print(f"  ... 还有 {len(random_stocks) - 10} 只")

    return run_backtest(start_date, end_date, stocks=random_stocks, use_market_filter=True, mode="single")


def run_full_market_backtest(start_date="20200101", end_date="20251231"):
    print("=" * 100)
    print("📊 全市场回测 — 扫描所有A股")
    print("=" * 100)

    all_stocks = get_all_a_stocks()
    if not all_stocks:
        print("❌ 无法获取全市场股票列表")
        return None

    print(f"📋 全A股数量: {len(all_stocks)} 只（已过滤ST、次新股）")
    return run_backtest(start_date, end_date, stocks=all_stocks, use_market_filter=True, mode="portfolio")


def _print_trade_stats(trades, total_return, max_dd):
    if not trades:
        print("  无交易记录")
        return

    total_trades = len(trades)
    win_trades = [t for t in trades if t["profit_pct"] > 0]
    lose_trades = [t for t in trades if t["profit_pct"] <= 0]
    win_rate = len(win_trades) / total_trades * 100 if total_trades > 0 else 0

    avg_profit = np.mean([t["profit_pct"] for t in trades])
    avg_win = np.mean([t["profit_pct"] for t in win_trades]) if win_trades else 0
    avg_lose = np.mean([t["profit_pct"] for t in lose_trades]) if lose_trades else 0
    avg_hold = np.mean([t["hold_days"] for t in trades])
    avg_max_profit = np.mean([t.get("max_profit_pct", 0) for t in trades])

    exit_counts = {}
    for t in trades:
        reason = t["exit_reason"]
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

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
        print(f"  {i:>3} {t.get('stock','?'):<10} {t['buy_date']:<12} {t['sell_date']:<12} "
              f"{t['buy_price']:>8.2f} {t['sell_price']:>8.2f} {t['profit_pct']:>+7.2f}% "
              f"{max_p:>+7.2f}% {t['hold_days']:>6} {t['exit_reason']:<8}")
    if len(trades) > 30:
        print(f"  ... 还有 {len(trades) - 30} 笔交易未显示")


def _print_signal_report(title, trades, summaries, key):
    print(f"\n{'─' * 100}")
    print(f"  {title}")
    print(f"{'─' * 100}")

    if not trades:
        print("  无交易记录")
        return

    total_trades = len(trades)
    win_trades = [t for t in trades if t["profit_pct"] > 0]
    lose_trades = [t for t in trades if t["profit_pct"] <= 0]
    win_rate = len(win_trades) / total_trades * 100 if total_trades > 0 else 0

    avg_profit = np.mean([t["profit_pct"] for t in trades])
    avg_win = np.mean([t["profit_pct"] for t in win_trades]) if win_trades else 0
    avg_lose = np.mean([t["profit_pct"] for t in lose_trades]) if lose_trades else 0
    avg_hold = np.mean([t["hold_days"] for t in trades])
    avg_max_profit = np.mean([t.get("max_profit_pct", 0) for t in trades])

    exit_counts = {}
    for t in trades:
        reason = t["exit_reason"]
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    total_profit_pct = sum(t["profit_pct"] for t in trades)

    max_dd_values = [s[f"{key}_max_dd"] for s in summaries if s[f"{key}_trades"] > 0]
    avg_max_dd = np.mean(max_dd_values) if max_dd_values else 0

    print(f"  总交易次数: {total_trades}")
    print(f"  盈利次数: {len(win_trades)} | 亏损次数: {len(lose_trades)}")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  平均每笔收益: {avg_profit:+.2f}%")
    print(f"  平均盈利: {avg_win:+.2f}% | 平均亏损: {avg_lose:+.2f}%")
    print(f"  盈亏比: {abs(avg_win / avg_lose):.2f}" if avg_lose != 0 else "  盈亏比: ∞")
    print(f"  平均持仓天数: {avg_hold:.1f}")
    print(f"  平均最大浮盈: {avg_max_profit:+.2f}%")
    print(f"  累计收益(简单加总): {total_profit_pct:+.2f}%")
    print(f"  平均最大回撤: {avg_max_dd:.2f}%")
    print(f"  退出原因分布: {dict(exit_counts)}")

    print(f"\n  {'#':>3} {'股票':<10} {'买入日':<12} {'卖出日':<12} {'买价':>8} {'卖价':>8} {'收益%':>8} {'最高浮盈':>8} {'持仓天':>6} {'退出原因':<8}")
    print(f"  {'─'*95}")
    for i, t in enumerate(trades[:50], 1):
        max_p = t.get('max_profit_pct', 0)
        print(f"  {i:>3} {t['stock']:<10} {t['buy_date']:<12} {t['sell_date']:<12} "
              f"{t['buy_price']:>8.2f} {t['sell_price']:>8.2f} {t['profit_pct']:>+7.2f}% "
              f"{max_p:>+7.2f}% {t['hold_days']:>6} {t['exit_reason']:<8}")
    if len(trades) > 50:
        print(f"  ... 还有 {len(trades) - 50} 笔交易未显示")

    print(f"\n  各股票收益:")
    print(f"  {'股票':<12} {'交易数':>6} {'收益率':>10} {'最大回撤':>10}")
    print(f"  {'─'*45}")
    for s in summaries:
        trades_key = f"{key}_trades"
        return_key = f"{key}_return"
        dd_key = f"{key}_max_dd"
        if s[trades_key] > 0:
            print(f"  {s['name']:<12} {s[trades_key]:>6} {s[return_key]:>+9.2f}% {s[dd_key]:>9.2f}%")


if __name__ == "__main__":
    mode = "single"
    start = "20200101"
    end = "20251231"
    n_stocks = None

    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1 == "portfolio":
            mode = "portfolio"
        elif arg1 == "oos":
            mode = "oos"
            if len(sys.argv) > 2:
                try:
                    n_stocks = int(sys.argv[2])
                except ValueError:
                    pass
        elif arg1 == "fullmarket":
            mode = "fullmarket"
        else:
            try:
                n_stocks = int(arg1)
            except ValueError:
                pass

    if len(sys.argv) > 3 and mode not in ("oos", "fullmarket"):
        start = sys.argv[2]
        end = sys.argv[3]

    if mode == "oos":
        run_oos_test(start, end, n_stocks=n_stocks or 50)
    elif mode == "fullmarket":
        run_full_market_backtest(start, end)
    else:
        stocks = BACKTEST_STOCKS[:n_stocks] if n_stocks else None
        run_backtest(start_date=start, end_date=end, stocks=stocks, use_market_filter=True, mode=mode)
