"""
砖型图选股策略回测 v2（止损防守线版）
策略来源：通达信公式
核心逻辑：生命线趋势 + 止损防守线 + KDJ超卖 + 砖型图第一块砖 + 回踩到位 + 非ST + 市值>=30亿
v2变更：新增坚决止损线(ZX_DK*0.98)，趋势安全改为C>坚决止损线，回踩到位增加距离止损线近条件
"""

import sys
import io
from pathlib import Path
from datetime import datetime
import json
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import tushare as ts
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(TUSHARE_TOKEN)

BACKTEST_START = '20240101'
BACKTEST_END = '20260516'
DATA_START = '20230601'
MIN_MARKET_CAP = 300000


def get_stock_universe():
    print("📊 获取A股股票列表...")
    stock_basic = pro.stock_basic(
        exchange='', list_status='L',
        fields='ts_code,symbol,name,industry,list_date'
    )
    a_stocks = stock_basic[
        stock_basic['ts_code'].str.endswith('.SH') |
        stock_basic['ts_code'].str.endswith('.SZ')
    ]
    a_stocks = a_stocks[~a_stocks['name'].str.contains('ST', na=False)]
    print(f"  非ST股票: {len(a_stocks)} 只")

    print("📊 获取流通市值数据...")
    try:
        trade_cal = pro.trade_cal(
            exchange='SSE', is_open='1',
            start_date='20260501', end_date='20260520'
        )
        latest_date = trade_cal['cal_date'].iloc[-1]
    except Exception:
        latest_date = '20260516'

    try:
        daily_basic = pro.daily_basic(
            trade_date=latest_date,
            fields='ts_code,circ_mv'
        )
        daily_basic = daily_basic[daily_basic['circ_mv'] >= MIN_MARKET_CAP]
        stocks = a_stocks.merge(daily_basic, on='ts_code', how='inner')
        print(f"  流通市值>=30亿: {len(stocks)} 只")
    except Exception as e:
        print(f"  ⚠️ 获取市值数据失败({e})，使用全部非ST股票")
        stocks = a_stocks.copy()
        stocks['circ_mv'] = 0

    return stocks


def calculate_brick_strategy(df):
    df = df.copy()

    for col in ['Close', 'High', 'Low', 'Volume']:
        if col not in df.columns and col.lower() in df.columns:
            df[col] = df[col.lower()]

    # ===== 1. 生命线系统 =====
    df['ma14'] = df['Close'].rolling(window=14).mean()
    df['ma28'] = df['Close'].rolling(window=28).mean()
    df['ma57'] = df['Close'].rolling(window=57).mean()
    df['ma114'] = df['Close'].rolling(window=114).mean()
    df['ZX_DK'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4

    df['ZX_ST'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ZX_ST'] = df['ZX_ST'].ewm(span=10, adjust=False).mean()

    df['yellow_up'] = df['ZX_DK'] > df['ZX_DK'].shift(1)

    df['stop_loss_line'] = df['ZX_DK'] * 0.98
    df['trend_safe'] = (df['Close'] > df['stop_loss_line']) & (df['ZX_ST'] > df['ZX_DK'])

    # ===== 2. KDJ指标 =====
    low_list = df['Low'].rolling(window=9, min_periods=1).min()
    high_list = df['High'].rolling(window=9, min_periods=1).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(50)

    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    df['J_below_13'] = df['J'] < 13
    df['J_freeze'] = (
        df['J_below_13']
        .rolling(window=3, min_periods=1)
        .max()
        .astype(bool)
    )

    # ===== 3. 砖型图核心逻辑 =====
    hhv4 = df['High'].rolling(window=4, min_periods=1).max()
    llv4 = df['Low'].rolling(window=4, min_periods=1).min()

    denom = hhv4 - llv4
    denom = denom.replace(0, np.nan)

    var1a = (hhv4 - df['Close']) / denom * 100 - 90
    var1a = var1a.replace([np.inf, -np.inf], np.nan).fillna(0)

    df['VAR2A'] = var1a.ewm(com=3, adjust=False).mean() + 100

    var3a = (df['Close'] - llv4) / denom * 100
    var3a = var3a.replace([np.inf, -np.inf], np.nan).fillna(50)

    var4a = var3a.ewm(com=5, adjust=False).mean()
    df['VAR5A'] = var4a.ewm(com=5, adjust=False).mean() + 100

    df['VAR6A'] = df['VAR5A'] - df['VAR2A']
    df['brick'] = np.where(df['VAR6A'] > 4, df['VAR6A'] - 4, 0)

    df['AA'] = df['brick'].shift(1) < df['brick']
    df['first_brick'] = (~df['AA'].shift(1).fillna(False)) & df['AA']

    # ===== 4. 空间位置 =====
    df['near_stop_loss'] = df['Close'] < df['stop_loss_line'] * 1.05
    df['pullback_ready'] = (
        (df['Low'] < df['ZX_ST'] * 1.03) | (df['Low'] < df['ZX_DK'] * 1.03)
    ) & df['near_stop_loss']

    # ===== 5. 最终综合选股 =====
    df['XG'] = (
        df['trend_safe'] &
        df['yellow_up'] &
        df['J_freeze'] &
        df['first_brick'] &
        df['pullback_ready']
    )

    return df


def backtest_signals(df, forward_days=None):
    if forward_days is None:
        forward_days = [1, 3, 5, 10, 20]

    signals = df[df['XG'] == True].copy()
    if len(signals) == 0:
        return None

    results = []
    close_prices = df['Close'].values
    open_prices = df['Open'].values if 'Open' in df.columns else close_prices

    for idx_val in signals.index:
        pos = df.index.get_loc(idx_val)
        row = df.loc[idx_val]

        if pos + 1 >= len(close_prices):
            continue

        entry_price = open_prices[pos + 1] if pos + 1 < len(open_prices) else close_prices[pos]

        result = {
            'date': idx_val,
            'close': row['Close'],
            'entry_price': entry_price,
            'J': row.get('J', np.nan),
            'ZX_DK': row.get('ZX_DK', np.nan),
            'ZX_ST': row.get('ZX_ST', np.nan),
            'brick': row.get('brick', np.nan),
        }

        for n in forward_days:
            future_pos = pos + 1 + n
            if future_pos < len(close_prices):
                future_price = close_prices[future_pos]
                ret = (future_price - entry_price) / entry_price * 100
                result[f'return_{n}d'] = ret
            else:
                result[f'return_{n}d'] = np.nan

        results.append(result)

    if not results:
        return None

    return pd.DataFrame(results)


def run_backtest():
    print("=" * 100)
    print("📊 砖型图选股策略回测系统 v2（止损防守线版）")
    print("=" * 100)
    print("策略条件：")
    print("  1. 趋势安全：收盘价>坚决止损线(ZX_DK*0.98) 且 白线>黄线")
    print("  2. 黄线上行：中线生命线斜率向上")
    print("  3. J冰点区：最近3天内J值去过13以下")
    print("  4. 第一块砖：砖型图刚从下降转为上升的第一天")
    print("  5. 回踩到位：低点接近白线或黄线(3%以内) 且 收盘价距止损线<5%")
    print("  6. 非ST 且 流通市值>=30亿")
    print(f"\n回测区间：{BACKTEST_START} ~ {BACKTEST_END}")
    print(f"数据起始：{DATA_START}（需预留MA114计算窗口）")
    print("=" * 100)

    # Step 1
    print("\n[1/4] 获取股票池...")
    stocks = get_stock_universe()
    total = len(stocks)
    print(f"  最终股票池: {total} 只")

    # Step 2
    print(f"\n[2/4] 逐只获取数据并计算信号...")
    all_signal_results = []
    processed = 0
    errors = 0
    start_time = time.time()

    for _, stock_row in stocks.iterrows():
        ts_code = stock_row['ts_code']
        name = stock_row['name']
        industry = stock_row.get('industry', '未知')

        processed += 1

        if processed % 100 == 0:
            elapsed = time.time() - start_time
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / speed if speed > 0 else 0
            print(f"  进度: {processed}/{total} ({processed/total*100:.1f}%) | "
                  f"信号: {len(all_signal_results)} | "
                  f"速度: {speed:.1f}只/秒 | "
                  f"预计剩余: {eta:.0f}秒")

        try:
            for retry in range(3):
                try:
                    df = pro.daily(ts_code=ts_code, start_date=DATA_START, end_date=BACKTEST_END)
                    break
                except Exception:
                    if retry < 2:
                        time.sleep(2)
                    else:
                        raise

            if df is None or len(df) < 130:
                continue

            df = df.sort_values('trade_date').reset_index(drop=True)
            df['Date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df.set_index('Date', inplace=True)
            df['Close'] = df['close']
            df['Open'] = df['open']
            df['High'] = df['high']
            df['Low'] = df['low']
            df['Volume'] = df['vol']

            df = calculate_brick_strategy(df)

            signal_results = backtest_signals(df)

            if signal_results is not None and len(signal_results) > 0:
                signal_results = signal_results[
                    signal_results['date'] >= BACKTEST_START
                ]
                if len(signal_results) > 0:
                    signal_results['ts_code'] = ts_code
                    signal_results['name'] = name
                    signal_results['industry'] = industry
                    all_signal_results.append(signal_results)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ⚠️ {ts_code} {name}: {e}")
            continue

        if processed % 400 == 0:
            time.sleep(1)

    elapsed = time.time() - start_time
    print(f"\n  ✅ 数据处理完成: {processed}只, 错误: {errors}, 耗时: {elapsed:.0f}秒")

    if not all_signal_results:
        print("❌ 未发现任何信号！")
        return None

    # Step 3
    print("\n[3/4] 汇总回测结果...")
    all_signals = pd.concat(all_signal_results, ignore_index=True)
    all_signals = all_signals.sort_values('date').reset_index(drop=True)

    # Step 4
    print("\n[4/4] 生成报告...")
    generate_report(all_signals)

    return all_signals


def generate_report(signals_df):
    print("\n" + "=" * 100)
    print("📋 砖型图选股策略 v2（止损防守线版）— 回测报告")
    print("=" * 100)

    print(f"\n📊 总体统计")
    print(f"  信号总数: {len(signals_df)}")
    print(f"  涉及股票: {signals_df['ts_code'].nunique()} 只")
    print(f"  时间范围: {signals_df['date'].min().strftime('%Y-%m-%d')} ~ "
          f"{signals_df['date'].max().strftime('%Y-%m-%d')}")

    # 行业分布
    print(f"\n📂 行业分布 (Top 10)")
    industry_counts = signals_df['industry'].value_counts().head(10)
    for ind, cnt in industry_counts.items():
        print(f"  {ind:<10} {cnt:>4} 个信号")

    # 前瞻收益分析
    print(f"\n📈 前瞻收益分析（次日开盘买入）")
    print("-" * 80)
    print(f"{'持有期':>8} | {'有效数':>6} | {'胜率':>8} | {'平均收益':>10} | "
          f"{'中位收益':>10} | {'最大收益':>10} | {'最大亏损':>10} | {'盈亏比':>8}")
    print("-" * 80)

    summary = {}
    for n in [1, 3, 5, 10, 20]:
        col = f'return_{n}d'
        valid = signals_df[col].dropna()
        if len(valid) == 0:
            continue

        win_rate = (valid > 0).sum() / len(valid) * 100
        avg_ret = valid.mean()
        median_ret = valid.median()
        max_ret = valid.max()
        min_ret = valid.min()

        wins = valid[valid > 0]
        losses = valid[valid < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        print(f"{n:>6}日 | {len(valid):>6} | {win_rate:>7.1f}% | "
              f"{avg_ret:>+9.2f}% | {median_ret:>+9.2f}% | "
              f"{max_ret:>+9.2f}% | {min_ret:>+9.2f}% | {profit_loss_ratio:>7.2f}")

        summary[f'{n}d'] = {
            'count': int(len(valid)),
            'win_rate': float(win_rate),
            'avg_return': float(avg_ret),
            'median_return': float(median_ret),
            'max_return': float(max_ret),
            'min_return': float(min_ret),
            'profit_loss_ratio': float(profit_loss_ratio),
        }

    # 月度统计
    print(f"\n📅 月度信号统计")
    print("-" * 60)
    signals_df['year_month'] = signals_df['date'].dt.to_period('M')
    monthly = signals_df.groupby('year_month').agg(
        signals=('ts_code', 'count'),
        stocks=('ts_code', 'nunique'),
    )
    if f'return_5d' in signals_df.columns:
        monthly['avg_5d'] = signals_df.groupby('year_month')['return_5d'].mean()
        monthly['win_5d'] = signals_df.groupby('year_month')['return_5d'].apply(
            lambda x: (x > 0).sum() / len(x.dropna()) * 100 if len(x.dropna()) > 0 else 0
        )

    for ym, row in monthly.iterrows():
        ret_str = f" | 5日均收:{row.get('avg_5d', np.nan):+.2f}%" if 'avg_5d' in row and not pd.isna(row.get('avg_5d')) else ""
        win_str = f" 胜率:{row.get('win_5d', np.nan):.0f}%" if 'win_5d' in row and not pd.isna(row.get('win_5d')) else ""
        print(f"  {ym}  信号:{row['signals']:>3}  股票:{row['stocks']:>3}{ret_str}{win_str}")

    # 最近信号
    print(f"\n🎯 最近30个信号详情")
    print("-" * 120)
    print(f"{'日期':>12} {'股票':<10} {'代码':<12} {'行业':<8} {'价格':>8} {'J值':>7} "
          f"{'砖型':>6} {'3日收益':>9} {'5日收益':>9} {'10日收益':>10} {'20日收益':>10}")
    print("-" * 120)

    recent = signals_df.sort_values('date', ascending=False).head(30)
    for _, row in recent.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
        r3 = f"{row.get('return_3d', np.nan):+.2f}%" if not pd.isna(row.get('return_3d', np.nan)) else "N/A"
        r5 = f"{row.get('return_5d', np.nan):+.2f}%" if not pd.isna(row.get('return_5d', np.nan)) else "N/A"
        r10 = f"{row.get('return_10d', np.nan):+.2f}%" if not pd.isna(row.get('return_10d', np.nan)) else "N/A"
        r20 = f"{row.get('return_20d', np.nan):+.2f}%" if not pd.isna(row.get('return_20d', np.nan)) else "N/A"
        j_val = f"{row.get('J', np.nan):.1f}" if not pd.isna(row.get('J', np.nan)) else "N/A"
        brick_val = f"{row.get('brick', np.nan):.1f}" if not pd.isna(row.get('brick', np.nan)) else "N/A"
        print(f"{date_str:>12} {row['name']:<10} {row['ts_code']:<12} {row.get('industry', ''):<8} "
              f"{row['close']:>8.2f} {j_val:>7} {brick_val:>6} "
              f"{r3:>9} {r5:>9} {r10:>10} {r20:>10}")

    # 保存
    output_dir = Path(__file__).parent.parent / "results" / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    signals_file = output_dir / f"brick_strategy_signals_{timestamp}.csv"
    signals_df_save = signals_df.copy()
    signals_df_save['date'] = signals_df_save['date'].dt.strftime('%Y-%m-%d')
    signals_df_save.to_csv(signals_file, encoding='utf-8-sig', index=False)
    print(f"\n💾 信号数据: {signals_file}")

    summary_data = {
        'strategy': '砖型图选股策略 v2（止损防守线版）',
        'conditions': [
            '趋势安全：收盘价>坚决止损线(ZX_DK*0.98) 且 白线>黄线',
            '黄线上行：中线生命线斜率向上',
            'J冰点区：最近3天内J值去过13以下',
            '第一块砖：砖型图刚从下降转为上升',
            '回踩到位：低点接近白线或黄线(3%以内) 且 收盘价距止损线<5%',
            '非ST 且 流通市值>=30亿',
        ],
        'backtest_period': f'{BACKTEST_START} ~ {BACKTEST_END}',
        'total_signals': len(signals_df),
        'unique_stocks': int(signals_df['ts_code'].nunique()),
        'forward_returns': summary,
    }

    summary_file = output_dir / f"brick_strategy_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"💾 回测总结: {summary_file}")

    print("\n" + "=" * 100)
    print("✅ 回测完成！")
    print("=" * 100)


if __name__ == "__main__":
    run_backtest()
