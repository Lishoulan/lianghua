"""
潜伏模型V6.3 —— 回测执行脚本
使用tushare数据源（前复权），对标准股票池进行回测
与推送脚本使用完全一致的前复权处理方式
"""
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
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(Path(__file__).parent / ".env")

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")

def _get_pro():
    """延迟初始化tushare pro，避免模块导入时token不存在报错"""
    if TUSHARE_TOKEN is None:
        raise ValueError("TUSHARE_TOKEN环境变量未设置")
    return ts.pro_api(TUSHARE_TOKEN)

from classic_ta.v60_ambush_model import IndicatorCalcBase
from classic_ta.v63_ambush_model import (
    add_micro_confirm_indicators,
    Detect_AmbushSignal_V63,
    StatefulTradeBacktester_V63,
    calc_volatility_parity_shares,
    V63_PARAMS,
)
from ml_strategy.oamv_filter import OAMVHysteresisFilter

# ── 回测股票池 ──────────────────────────────────────────────────────
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


def get_all_a_stocks(min_list_date="20200101"):
    """获取全市场A股列表（排除ST、次新）"""
    try:
        pro = _get_pro()
        stock_basic = pro.stock_basic(
            exchange="", list_status="L",
            fields="ts_code,symbol,name,industry,list_date",
        )
        # 只要沪深A股
        a_stocks = stock_basic[
            stock_basic["ts_code"].str.endswith(".SH")
            | stock_basic["ts_code"].str.endswith(".SZ")
        ].copy()
        # 排除ST
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("*ST")]
        a_stocks = a_stocks[~a_stocks["name"].str.startswith("N")]
        # 排除次新（上市不足1年）
        a_stocks = a_stocks[a_stocks["list_date"] < min_list_date]
        # 排除北交所（8/9开头）
        a_stocks = a_stocks[~a_stocks["ts_code"].str.startswith("8")]
        a_stocks = a_stocks[~a_stocks["ts_code"].str.startswith("9")]
        result = [(row["ts_code"], row["name"]) for _, row in a_stocks.iterrows()]
        print(f"  全市场A股: {len(result)} 只（排除ST/次新/北交所）")
        return result
    except Exception as e:
        print(f"  获取股票列表失败: {e}")
        return BACKTEST_STOCKS


def get_oamv_allow_buy(start_date="20200101", end_date="20260516"):
    """获取OAMV活跃市值择时：允许买入的日期集合（指南针同款参数）"""
    try:
        from ml_strategy.market_amv_cache import get_market_amv_series_for_backtest

        # 优化后参数: SMA(15)平滑 + CostMA(42), 阈值+2.0%/-1.0%
        amv_series = get_market_amv_series_for_backtest(start_date=start_date, end_date=end_date)
        if amv_series is not None and len(amv_series) >= 40:
            oamv = OAMVHysteresisFilter(
                upper_threshold=2.0, lower_threshold=-1.0,
                cost_ma_period=42, roc_period=1,
                weekly_ema_period=5, weekly_use_ema=True,
                smooth_method='sma', smooth_period=15,
                cost_ma_method='sma',
            )
            oamv.fit(amv_series=amv_series)
            data_source = "优化后活筹(SMA15+CostMA42|+2.0/-1.0)"
        else:
            # 回退: 使用沪深300成交额代理
            print("  OAMV: 全市场活跃市值数据不足，回退到成交额代理")
            pro = _get_pro()
            index_df = pro.index_daily(ts_code="000300.SH", start_date=start_date, end_date=end_date)
            if index_df is None or len(index_df) < 40:
                print("  OAMV: 沪深300数据不足，跳过择时")
                return None
            index_df = index_df.sort_values("trade_date").reset_index(drop=True)
            index_df["Date"] = pd.to_datetime(index_df["trade_date"], format="%Y%m%d")
            index_df.set_index("Date", inplace=True)
            index_df["amount"] = index_df["amount"].astype(float)
            oamv = OAMVHysteresisFilter(
                upper_threshold=2.0, lower_threshold=-1.0,
                cost_ma_period=42, smooth_method='sma', smooth_period=15,
            )
            oamv.fit(index_df)
            data_source = "成交额代理(amount)"

        state_df = oamv.get_state_df()
        if state_df is None or len(state_df) == 0:
            return None

        # 构建market_allow_buy Series（按日期索引，True=允许买入）
        market_allow_buy = state_df["oamv_state"] == 1
        allowed_count = market_allow_buy.sum()
        total_count = len(market_allow_buy)
        print(f"  OAMV择时: {data_source} | 允许交易{allowed_count}/{total_count}天 "
              f"({allowed_count/total_count*100:.1f}%)")

        return market_allow_buy
    except Exception as e:
        print(f"  OAMV计算失败: {e}，跳过择时")
        return None


def get_oamv_allow_buy_v2(start_date="20200101", end_date="20260516",
                          buy_threshold=0.0425, sell_threshold=0.0125):
    """获取OAMV择时：基于AMV单日涨跌幅阈值（校准版）

    逻辑(Long-Only):
      - AMV单日涨幅 >= buy_threshold → 允许买入 (True)
      - AMV单日跌幅 <= -sell_threshold → 禁止买入 (False)
      - 其他: 维持上一日状态
    """
    try:
        from ml_strategy.market_amv_cache import get_market_amv_series_for_backtest
        import tushare as ts

        amv_series = get_market_amv_series_for_backtest(start_date=start_date, end_date=end_date)
        if amv_series is None or len(amv_series) < 40:
            print("  AMV校准版: 数据不足，跳过")
            return None

        # 计算AMV单日涨跌幅
        amv_roc = amv_series.pct_change(1)

        # 生成允许买入序列
        allow = pd.Series(False, index=amv_series.index)
        for i in range(1, len(amv_roc)):
            roc = amv_roc.iloc[i]
            if pd.isna(roc):
                allow.iloc[i] = allow.iloc[i-1]
            elif roc >= buy_threshold:
                allow.iloc[i] = True
            elif roc <= -sell_threshold:
                allow.iloc[i] = False
            else:
                allow.iloc[i] = allow.iloc[i-1]

        allowed_count = allow.sum()
        total_count = len(allow)
        print(f"  AMV校准版: buy>={buy_threshold:.2%}, sell<={sell_threshold:.2%} | "
              f"允许交易{allowed_count}/{total_count}天 ({allowed_count/total_count*100:.1f}%)")

        return allow
    except Exception as e:
        print(f"  AMV校准版计算失败: {e}，跳过择时")
        return None


def get_backtest_data(ts_code, start_date="20200101", end_date="20260516"):
    """获取单只股票日线数据（前复权）—— 与推送脚本完全一致的处理方式"""
    try:
        pro = _get_pro()
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or len(df) < 130:
            return None

        # 获取复权因子，手动计算前复权
        try:
            adj = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if adj is not None and len(adj) > 0:
                adj = adj.sort_values("trade_date").reset_index(drop=True)
                latest_adj = adj["adj_factor"].iloc[-1]
                adj_ratio = adj["adj_factor"].astype(float) / float(latest_adj)
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["open"] = df["open"].astype(float) * adj_ratio
                df["high"] = df["high"].astype(float) * adj_ratio
                df["low"] = df["low"].astype(float) * adj_ratio
                df["close"] = df["close"].astype(float) * adj_ratio
            else:
                df = df.sort_values("trade_date").reset_index(drop=True)
        except Exception:
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


def backtest_single_stock(df, initial_cash=100000):
    """单只股票回测（V6.3模型）"""
    try:
        df = IndicatorCalcBase(df)
        df = add_micro_confirm_indicators(df)
        df = Detect_AmbushSignal_V63(df, V63_PARAMS)
        trades = StatefulTradeBacktester_V63(
            df, signal_col="ambush_signal",
            initial_cash=initial_cash, params=V63_PARAMS,
        )
        return trades
    except Exception as e:
        print(f"    回测异常: {e}")
        return []


def _backtest_one_stock(ts_code, name, start_date, end_date):
    """单只股票回测（用于并发）"""
    try:
        df = get_backtest_data(ts_code, start_date, end_date)
        if df is None:
            return ts_code, name, None, 0, []

        trades = backtest_single_stock(df)

        # 信号统计
        df_ind = IndicatorCalcBase(df)
        df_ind = add_micro_confirm_indicators(df_ind)
        df_sig = Detect_AmbushSignal_V63(df_ind, V63_PARAMS)
        sig_count = int(df_sig["ambush_signal"].sum()) if "ambush_signal" in df_sig.columns else 0

        trade_records = []
        for t in trades:
            trade_records.append({
                "stock": name,
                "code": ts_code,
                "buy_date": t.buy_date,
                "sell_date": t.sell_date,
                "buy_price": t.buy_price,
                "sell_price": t.sell_price,
                "shares": t.shares,
                "hold_days": t.hold_days,
                "profit_pct": t.profit_pct,
                "max_profit_pct": t.max_profit_pct,
                "exit_reason": t.exit_reason,
            })

        return ts_code, name, trade_records, sig_count, trades
    except Exception as e:
        return ts_code, name, None, 0, []


def run_backtest(start_date="20200101", end_date="20260516", stocks=None, use_oamv=True, full_market=False, max_workers=3, oamv_mode="v1"):
    """主回测入口

    oamv_mode:
        "v1" - 指南针同款(OAMVHysteresisFilter)
        "v2" - AMV校准版(单日涨跌幅阈值 buy=4.25%/sell=1.25%)
        "off" - 不使用OAMV择时
    """
    if oamv_mode == "off":
        use_oamv = False

    if full_market:
        print("\n📊 获取全市场A股列表...")
        stocks = get_all_a_stocks(min_list_date=start_date)
    elif stocks is None:
        stocks = BACKTEST_STOCKS

    print("=" * 100)
    print("  潜伏模型V6.3 — 回测系统（前复权数据）")
    print(f"  回测区间: {start_date} ~ {end_date}")
    print(f"  回测股票: {len(stocks)} 只{'（全市场）' if full_market else ''}")
    print(f"  数据处理: tushare daily + adj_factor 手动前复权（与推送一致）")
    print(f"  策略逻辑: 威科夫LPS + VPA枯竭 + 微观确认(VWAP/VCP)")
    print(f"  退出机制: 硬止损(-2.5ATR) | 吊灯止盈(3.5ATR) | Buy Climax | 时间止损(10日)")
    print(f"  OAMV择时: {'关闭' if not use_oamv else '指南针同款' if oamv_mode=='v1' else 'AMV校准版(4.25%/1.25%)'}")
    print("=" * 100)

    # ── OAMV择时 ──
    market_allow_buy = None
    if use_oamv:
        print("\n📊 正在获取OAMV活跃市值数据...")
        if oamv_mode == "v2":
            market_allow_buy = get_oamv_allow_buy_v2(start_date, end_date)
        else:
            market_allow_buy = get_oamv_allow_buy(start_date, end_date)

    all_trades = []
    stock_summaries = []
    processed = 0
    errors = 0
    start_time = time.time()

    if full_market and max_workers > 1:
        # ── 并发模式（全市场） ──
        print(f"\n🚀 并发回测 {len(stocks)} 只股票 (workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for ts_code, name in stocks:
                f = executor.submit(_backtest_one_stock, ts_code, name, start_date, end_date)
                futures[f] = (ts_code, name)

            for f in as_completed(futures):
                ts_code, name = futures[f]
                processed += 1
                try:
                    _, _, trade_records, sig_count, trades = f.result()
                except Exception:
                    errors += 1
                    continue

                if trade_records is None:
                    errors += 1
                    if processed % 200 == 0:
                        print(f"  [{processed}/{len(stocks)}] 进度... 错误{errors}")
                    continue

                all_trades.extend(trade_records)

                if trades:
                    stock_summaries.append({
                        "code": ts_code,
                        "name": name,
                        "trades": len(trades),
                        "signals": sig_count,
                        "avg_profit": round(float(np.mean([t.profit_pct for t in trades])), 2) if trades else 0,
                        "win_rate": round(float(len([t for t in trades if t.profit_pct > 0]) / len(trades) * 100), 2) if trades else 0,
                    })

                if processed % 200 == 0:
                    elapsed_so_far = time.time() - start_time
                    speed = processed / elapsed_so_far
                    eta = (len(stocks) - processed) / speed if speed > 0 else 0
                    print(f"  [{processed}/{len(stocks)}] 进度... 交易{len(all_trades)}笔 耗时{elapsed_so_far:.0f}s ETA{eta:.0f}s")
    else:
        # ── 串行模式 ──
        for ts_code, name in stocks:
            processed += 1
            print(f"\n[{processed}/{len(stocks)}] {name} ({ts_code})")

            df = get_backtest_data(ts_code, start_date, end_date)
            if df is None:
                print(f"  数据获取失败")
                errors += 1
                continue

            try:
                trades = backtest_single_stock(df)
            except Exception as e:
                print(f"  回测失败: {e}")
                errors += 1
                continue

            # 信号统计
            df_ind = IndicatorCalcBase(df)
            df_ind = add_micro_confirm_indicators(df_ind)
            df_sig = Detect_AmbushSignal_V63(df_ind, V63_PARAMS)
            sig_count = df_sig["ambush_signal"].sum() if "ambush_signal" in df_sig.columns else 0
            print(f"  潜伏信号={sig_count}")

            for t in trades:
                all_trades.append({
                    "stock": name,
                    "code": ts_code,
                    "buy_date": t.buy_date,
                    "sell_date": t.sell_date,
                    "buy_price": t.buy_price,
                    "sell_price": t.sell_price,
                    "shares": t.shares,
                    "hold_days": t.hold_days,
                    "profit_pct": t.profit_pct,
                    "max_profit_pct": t.max_profit_pct,
                    "exit_reason": t.exit_reason,
                })

            if trades:
                win_trades = [t for t in trades if t.profit_pct > 0]
                lose_trades = [t for t in trades if t.profit_pct <= 0]
                avg_profit = np.mean([t.profit_pct for t in trades])
                avg_hold = np.mean([t.hold_days for t in trades])
                win_rate = len(win_trades) / len(trades) * 100
                print(f"  📈 交易={len(trades)}笔 胜率={win_rate:.0f}% 平均收益={avg_profit:+.2f}% 平均持仓={avg_hold:.1f}天")
            else:
                print(f"  📈 无交易")

            stock_summaries.append({
                "code": ts_code,
                "name": name,
                "trades": len(trades),
                "signals": int(sig_count),
                "avg_profit": round(float(np.mean([t.profit_pct for t in trades])), 2) if trades else 0,
                "win_rate": round(float(len([t for t in trades if t.profit_pct > 0]) / len(trades) * 100), 2) if trades else 0,
            })

    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.1f}s | 成功: {len(stocks)-errors} | 失败: {errors}")

    # ── OAMV择时过滤 ──
    if market_allow_buy is not None and all_trades:
        before_count = len(all_trades)
        filtered_trades = []
        for t in all_trades:
            buy_date = pd.Timestamp(t["buy_date"])
            if buy_date in market_allow_buy.index and market_allow_buy.loc[buy_date]:
                filtered_trades.append(t)
            elif buy_date not in market_allow_buy.index:
                # 日期不在OAMV数据中，保留（可能超出范围）
                filtered_trades.append(t)
        oamv_filtered = before_count - len(filtered_trades)
        all_trades = filtered_trades
        print(f"\n🔒 OAMV择时过滤: 移除{oamv_filtered}笔不允许交易, 保留{len(all_trades)}笔")

    # ── 汇总报告 ──
    print("\n" + "=" * 100)
    print("  潜伏模型V6.3 — 回测结果汇总（前复权数据 + OAMV择时）")
    print("=" * 100)

    if not all_trades:
        print("  无交易记录")
        return

    total = len(all_trades)
    wins = [t for t in all_trades if t["profit_pct"] > 0]
    losses = [t for t in all_trades if t["profit_pct"] <= 0]
    win_rate = len(wins) / total * 100
    avg_profit = np.mean([t["profit_pct"] for t in all_trades])
    avg_win = np.mean([t["profit_pct"] for t in wins]) if wins else 0
    avg_lose = np.mean([t["profit_pct"] for t in losses]) if losses else 0
    avg_hold = np.mean([t["hold_days"] for t in all_trades])
    avg_max_profit = np.mean([t["max_profit_pct"] for t in all_trades])

    exit_counts = {}
    for t in all_trades:
        exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1

    print(f"  总交易: {total}笔 | 盈利: {len(wins)} | 亏损: {len(losses)}")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  平均收益: {avg_profit:+.2f}% | 平均盈利: {avg_win:+.2f}% | 平均亏损: {avg_lose:+.2f}%")
    if avg_lose != 0:
        print(f"  盈亏比: {abs(avg_win / avg_lose):.2f}")
    print(f"  平均持仓: {avg_hold:.1f}天 | 平均最大浮盈: {avg_max_profit:+.2f}%")
    print(f"  退出原因分布: {dict(exit_counts)}")

    # 详细交易列表
    print(f"\n  {'#':>3} {'股票':<10} {'买入日':<12} {'卖出日':<12} {'买价':>8} {'卖价':>8} "
          f"{'收益%':>8} {'最高浮盈':>8} {'持仓天':>6} {'退出原因':<16}")
    print(f"  {'─' * 100}")
    for i, t in enumerate(all_trades[:80], 1):
        print(f"  {i:>3} {t['stock']:<10} {t['buy_date']:<12} {t['sell_date']:<12} "
              f"{t['buy_price']:>8.2f} {t['sell_price']:>8.2f} {t['profit_pct']:>+7.2f}% "
              f"{t['max_profit_pct']:>+7.2f}% {t['hold_days']:>6} {t['exit_reason']:<16}")
    if len(all_trades) > 80:
        print(f"  ... 还有 {len(all_trades) - 80} 笔交易未显示")

    # 各股票汇总
    print(f"\n  各股票收益:")
    print(f"  {'股票':<12} {'信号数':>6} {'交易数':>6} {'胜率':>8} {'平均收益':>10}")
    print(f"  {'─' * 50}")
    for s in stock_summaries:
        if s["trades"] > 0 or s["signals"] > 0:
            print(f"  {s['name']:<12} {s['signals']:>6} {s['trades']:>6} {s['win_rate']:>7.1f}% {s['avg_profit']:>+9.2f}%")

    # 保存结果
    output_dir = Path(__file__).parent / "results" / "ambush_v6_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"ambush_v6_{timestamp}.json"
    result = {
        "version": "6.3",
        "model": "潜伏模型V6.3（前复权 + OAMV择时）",
        "backtest_time": datetime.now().isoformat(),
        "period": f"{start_date}~{end_date}",
        "oamv_filter": use_oamv,
        "total_trades": total,
        "win_rate": round(win_rate, 2),
        "avg_profit": round(avg_profit, 2),
        "avg_win": round(avg_win, 2),
        "avg_lose": round(avg_lose, 2),
        "avg_hold_days": round(avg_hold, 1),
        "exit_counts": exit_counts,
        "trades": all_trades,
        "stock_summaries": stock_summaries,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 回测结果已保存: {report_file}")

    return result


if __name__ == "__main__":
    start = "20200101"
    end = "20260516"
    full = False
    oamv_mode = "v1"
    if len(sys.argv) > 2:
        start = sys.argv[1]
        end = sys.argv[2]
    if "--full" in sys.argv:
        full = True
    if "--oamv" in sys.argv:
        idx = sys.argv.index("--oamv")
        if idx + 1 < len(sys.argv):
            oamv_mode = sys.argv[idx + 1]
    run_backtest(start_date=start, end_date=end, full_market=full, oamv_mode=oamv_mode)
