"""
双线战法 + 更多指标 极致优化策略回测
新增策略：
- J: MACD+RSI优化（RSI<70买入，RSI>80止盈）
- K: MACD+布林带（价格触中轨或下轨买入）
- L: MACD+CCI顺势（CCI<-100超卖买入）
- M: MACD+威廉指标（威廉<20超卖买入）
- N: MACD+动量突破（动量加速买入）
- O: MACD+均线多头（价格站上多条均线买入）
- P: MACD+成交量确认（量能放大确认）
- Q: MACD+ATR波动率（高波动回避买入）
- R: MACD+OBV能量潮（OBV上涨确认）
- S: MACD+综合滤（RSI+布林+CCI多维确认）
回测时间：2024-05-17 到 2026-05-17
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import tushare as ts

load_dotenv(Path(__file__).parent.parent / ".env")

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = None
if TUSHARE_TOKEN:
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
        print("✅ Tushare初始化成功")
    except Exception as e:
        print(f"❌ Tushare初始化失败: {e}")
        sys.exit(1)

STRATEGY_NAMES = {
    'B': '趋势持有(MACD白>黄买/MACD下穿0+白<黄卖)',
    'G': '量价配合(MACD+放量上涨买/白<黄卖)',
    'J': 'MACD+RSI(RSI<70买入,RSI>80止盈)',
    'K': 'MACD+布林带(价格触布林中轨买入)',
    'L': 'MACD+CCI顺势(CCI<-100超卖买)',
    'M': 'MACD+威廉(威廉<-20超卖买)',
    'N': 'MACD+动量(动量加速确认买)',
    'O': 'MACD+均线多头(站上5/10/20日线)',
    'P': 'MACD+成交量(量能放大2倍确认)',
    'Q': 'MACD+ATR波动(低波动买入)',
    'R': 'MACD+OBV能量潮(OBV上涨确认)',
    'S': 'MACD+综合滤(RSI+布林+CCI多维)',
}


def get_all_a_stocks():
    print("📊 获取所有A股股票列表...")
    try:
        stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,list_date')
        a_stocks = stock_basic[(stock_basic['ts_code'].str.endswith('.SH')) | (stock_basic['ts_code'].str.endswith('.SZ'))]
        print(f"✅ 共获取 {len(a_stocks)} 只A股股票")
        return a_stocks
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")
        return None


def get_stock_daily(ts_code, start_date='20240517', end_date='20260516'):
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'vol': 'Volume', 'trade_date': 'Date'
        }
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        return None


def calculate_all_indicators(df):
    """计算所有指标"""
    df = df.copy()
    if 'Close' not in df.columns:
        df['Close'] = df['close']
    
    # === 双线战法 ===
    df['white_line'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['white_line'] = df['white_line'].ewm(span=10, adjust=False).mean()
    df['ma14'] = df['Close'].rolling(window=14).mean()
    df['ma28'] = df['Close'].rolling(window=28).mean()
    df['ma57'] = df['Close'].rolling(window=57).mean()
    df['ma114'] = df['Close'].rolling(window=114).mean()
    df['yellow_line'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4
    
    # === MACD ===
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema_fast - ema_slow
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    df['MACD_cross_up'] = [False] + [(df['MACD'].iloc[i] >= 0) and (df['MACD'].iloc[i-1] < 0) for i in range(1, len(df))]
    df['MACD_cross_down'] = [False] + [(df['MACD'].iloc[i] < 0) and (df['MACD'].iloc[i-1] >= 0) for i in range(1, len(df))]
    df['white_above_yellow'] = [False] + [df['white_line'].iloc[i] > df['yellow_line'].iloc[i] for i in range(1, len(df))]
    df['white_cross_down'] = [False] + [(df['white_line'].iloc[i] <= df['yellow_line'].iloc[i]) and (df['white_line'].iloc[i-1] > df['yellow_line'].iloc[i-1]) for i in range(1, len(df))]
    
    # === RSI ===
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_oversold'] = df['RSI'] < 30
    df['RSI_overbought'] = df['RSI'] > 70
    
    # === 布林带 ===
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['BB_touch_lower'] = df['Close'] <= df['BB_lower']
    df['BB_touch_middle'] = (df['Close'] > df['BB_middle']) & (df['Close'].shift(1) <= df['BB_middle'].shift(1))
    
    # === CCI ===
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=14).mean()
    mad = tp.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - sma_tp) / (0.015 * mad)
    df['CCI_oversold'] = df['CCI'] < -100
    df['CCI_overbought'] = df['CCI'] > 100
    
    # === 威廉指标 ===
    df['威廉'] = -100 * (df['High'].rolling(window=14).max() - df['Close']) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())
    df['威廉_oversold'] = df['威廉'] < -80
    df['威廉_overbought'] = df['威廉'] > -20
    
    # === 动量 ===
    df['momentum_5'] = df['Close'].pct_change(5) * 100
    df['momentum_accelerate'] = df['momentum_5'] > df['momentum_5'].shift(1)
    
    # === 均线多头 ===
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma10'] = df['Close'].rolling(window=10).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    df['ma多头'] = (df['Close'] > df['ma5']) & (df['Close'] > df['ma10']) & (df['Close'] > df['ma20'])
    
    # === 成交量 ===
    df['vol_ma5'] = df['Volume'].rolling(window=5).mean()
    df['vol_surge'] = df['Volume'] > df['vol_ma5'] * 2
    df['vol_increase'] = df['Volume'] > df['vol_ma5']
    df['price_up'] = df['Close'] > df['Close'].shift(1)
    df['vol_price_up'] = df['vol_increase'] & df['price_up']
    
    # === ATR ===
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['ATR_ratio'] = df['ATR'] / df['Close'] * 100
    df['low_volatility'] = df['ATR_ratio'] < df['ATR_ratio'].rolling(window=20).mean()
    
    # === OBV ===
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_up'] = df['OBV'] > df['OBV'].shift(1)
    
    # === 量价配合 ===
    df['volume_price_up'] = df['vol_increase'] & df['price_up']
    
    return df


def backtest_strategy(df, strategy='B', initial_cash=1000000, position_size=0.95):
    cash = initial_cash
    position = 0
    entry_price = 0
    peak_price = 0
    
    for _, row in df.iterrows():
        if any(pd.isna(row.get(col)) for col in ['Close', 'MACD', 'white_line', 'yellow_line', 'RSI', 'CCI', '威廉']):
            continue
        
        current_price = row['Close']
        
        # === 策略B：趋势持有（基准） ===
        if strategy == 'B':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow']:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0 and row['MACD_cross_down'] and (not row['white_above_yellow']):
                cash += position * current_price; position = 0
        
        # === 策略G：量价配合 ===
        elif strategy == 'G':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and row['volume_price_up']:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0 and row['MACD_cross_down'] and (not row['white_above_yellow']):
                cash += position * current_price; position = 0
        
        # === 策略J：MACD+RSI ===
        elif strategy == 'J':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and row['RSI'] < 70:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0:
                if current_price > peak_price: peak_price = current_price
                drawdown = (peak_price - current_price) / peak_price * 100 if peak_price > 0 else 0
                # RSI>80止盈 或 白线下穿黄线
                if row['RSI'] > 80 or (row['MACD_cross_down'] and (not row['white_above_yellow'])):
                    cash += position * current_price; position = 0
        
        # === 策略K：MACD+布林带 ===
        elif strategy == 'K':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and (row['BB_touch_lower'] or row['BB_touch_middle']):
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0 and row['MACD_cross_down'] and (not row['white_above_yellow']):
                cash += position * current_price; position = 0
        
        # === 策略L：MACD+CCI ===
        elif strategy == 'L':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and (row['CCI_oversold'] or row['CCI'] < 0):
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0:
                if current_price > peak_price: peak_price = current_price
                drawdown = (peak_price - current_price) / peak_price * 100 if peak_price > 0 else 0
                if row['CCI_overbought'] or drawdown >= 10 or (row['MACD_cross_down'] and (not row['white_above_yellow'])):
                    cash += position * current_price; position = 0
        
        # === 策略M：MACD+威廉 ===
        elif strategy == 'M':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and row['威廉_oversold']:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0:
                if current_price > peak_price: peak_price = current_price
                drawdown = (peak_price - current_price) / peak_price * 100 if peak_price > 0 else 0
                if row['威廉_overbought'] or drawdown >= 10 or (row['MACD_cross_down'] and (not row['white_above_yellow'])):
                    cash += position * current_price; position = 0
        
        # === 策略N：MACD+动量 ===
        elif strategy == 'N':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and row['momentum_accelerate']:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0 and row['MACD_cross_down'] and (not row['white_above_yellow']):
                cash += position * current_price; position = 0
        
        # === 策略O：MACD+均线多头 ===
        elif strategy == 'O':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and row['ma多头']:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0 and row['MACD_cross_down'] and (not row['white_above_yellow']):
                cash += position * current_price; position = 0
        
        # === 策略P：MACD+成交量放大 ===
        elif strategy == 'P':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and row['vol_surge']:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0 and row['MACD_cross_down'] and (not row['white_above_yellow']):
                cash += position * current_price; position = 0
        
        # === 策略Q：MACD+ATR低波动 ===
        elif strategy == 'Q':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and row['low_volatility']:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0 and row['MACD_cross_down'] and (not row['white_above_yellow']):
                cash += position * current_price; position = 0
        
        # === 策略R：MACD+OBV ===
        elif strategy == 'R':
            if position == 0 and row['MACD_cross_up'] and row['white_above_yellow'] and row['OBV_up']:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0 and row['MACD_cross_down'] and (not row['white_above_yellow']):
                cash += position * current_price; position = 0
        
        # === 策略S：MACD+综合过滤 ===
        elif strategy == 'S':
            buy_condition = (
                row['MACD_cross_up'] and 
                row['white_above_yellow'] and 
                row['RSI'] < 70 and 
                row['CCI'] < 100
            )
            sell_condition = row['MACD_cross_down'] and (not row['white_above_yellow'])
            
            if position == 0 and buy_condition:
                shares = int(cash * position_size / current_price)
                if shares > 0:
                    position = shares; entry_price = current_price; peak_price = current_price; cash -= shares * current_price
            elif position > 0 and sell_condition:
                cash += position * current_price; position = 0
    
    if position > 0:
        cash += position * df.iloc[-1]['Close']
        position = 0
    
    return_pct = (cash - initial_cash) / initial_cash * 100
    buy_hold_return = (df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close'] * 100
    
    return return_pct, buy_hold_return


def backtest_single_stock_all_strategies(ts_code, name, industry=None):
    try:
        df = get_stock_daily(ts_code, '20240517', '20260516')
        if df is None or len(df) < 100:
            return None
        
        df = calculate_all_indicators(df)
        
        results = {}
        for strategy in ['B', 'G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']:
            return_pct, buy_hold_return = backtest_strategy(df, strategy)
            results[strategy] = {'return_pct': return_pct, 'buy_hold_return': buy_hold_return}
        
        return {
            'code': ts_code,
            'name': name,
            'industry': industry if industry else '未知',
            'strategies': results,
            'success': True
        }
    except Exception as e:
        return None


def run_multi_strategy_backtest():
    print("=" * 140)
    print("📊 双线战法 + 更多指标 极致优化策略回测")
    print("回测时间：2024-05-17 到 2026-05-17")
    print("=" * 140)
    print("\n策略说明：")
    for k, v in STRATEGY_NAMES.items():
        print(f"  策略{k}: {v}")
    print("=" * 140)
    
    stock_list = get_all_a_stocks()
    if stock_list is None:
        return
    
    all_results = []
    start_time = time.time()
    
    print(f"\n⏳ 开始回测 {len(stock_list)} 只股票 x 11种策略...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {
            executor.submit(backtest_single_stock_all_strategies, row['ts_code'], row['name'], row.get('industry', '未知')): row
            for _, row in stock_list.iterrows()
        }
        
        for i, future in enumerate(as_completed(future_to_stock), 1):
            result = future.result()
            if result is not None and result.get('success', False):
                all_results.append(result)
            
            if i % 500 == 0:
                elapsed = time.time() - start_time
                print(f"进度: {i}/{len(stock_list)} ({i/len(stock_list)*100:.1f}%), "
                      f"成功: {len(all_results)}, 耗时: {elapsed:.1f}秒")
    
    elapsed_time = time.time() - start_time
    
    # 分析结果
    print("\n" + "=" * 140)
    print("📈 多策略对比结果汇总")
    print("=" * 140)
    
    strategy_stats = {}
    for strategy in ['B', 'G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']:
        returns = [r['strategies'][strategy]['return_pct'] for r in all_results]
        buy_hold_returns = [r['strategies'][strategy]['buy_hold_return'] for r in all_results]
        positive_count = sum(1 for r in returns if r > 0)
        outperform_count = sum(1 for r in all_results if r['strategies'][strategy]['return_pct'] > r['strategies'][strategy]['buy_hold_return'])
        total = len(all_results)
        
        strategy_stats[strategy] = {
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'positive_rate': positive_count / total * 100,
            'outperform_rate': outperform_count / total * 100,
            'avg_buy_hold': np.mean(buy_hold_returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'sharpe_like': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        }
    
    print(f"\n{'策略':<6} {'名称':<40} {'平均收益':>10} {'中位收益':>10} {'正收益%':>10} {'跑赢持%':>10} {'夏普比':>10}")
    print("-" * 140)
    for strategy in ['B', 'G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']:
        s = strategy_stats[strategy]
        print(f"{strategy:<6} {STRATEGY_NAMES[strategy][:40]:<40} {s['avg_return']:>+9.2f}% {s['median_return']:>+9.2f}% {s['positive_rate']:>9.1f}% {s['outperform_rate']:>9.1f}% {s['sharpe_like']:>10.3f}")
    
    # 综合排名
    print("\n" + "=" * 140)
    print("🎯 综合评分（加权：平均收益40% + 正收益率30% + 跑赢率20% + 夏普比10%）")
    print("=" * 140)
    
    scores = {}
    for strategy in ['B', 'G', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']:
        s = strategy_stats[strategy]
        all_avg = [strategy_stats[k]['avg_return'] for k in strategy_stats.keys()]
        all_pos = [strategy_stats[k]['positive_rate'] for k in strategy_stats.keys()]
        all_out = [strategy_stats[k]['outperform_rate'] for k in strategy_stats.keys()]
        all_sharpe = [strategy_stats[k]['sharpe_like'] for k in strategy_stats.keys()]
        
        def normalize(val, all_vals):
            min_v, max_v = min(all_vals), max(all_vals)
            if max_v == min_v:
                return 50
            return (val - min_v) / (max_v - min_v) * 100
        
        score = (
            normalize(s['avg_return'], all_avg) * 0.4 +
            normalize(s['positive_rate'], all_pos) * 0.3 +
            normalize(s['outperform_rate'], all_out) * 0.2 +
            normalize(s['sharpe_like'], all_sharpe) * 0.1
        )
        scores[strategy] = score
    
    ranked_final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (k, v) in enumerate(ranked_final, 1):
        marker = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        print(f"  {marker} 第{i}名: 策略{k}({STRATEGY_NAMES[k][:30]}) - 综合得分: {v:.1f}")
    
    # Top3最优策略
    best_strategy = ranked_final[0][0]
    print(f"\n" + "=" * 140)
    print(f"🏆 Top3最优策略 Top20股票")
    print("=" * 140)
    
    for rank_idx, (strat, _) in enumerate(ranked_final[:3], 1):
        print(f"\n--- 第{rank_idx}名: 策略{strat}({STRATEGY_NAMES[strat]}) ---")
        stock_ranked = sorted(all_results, key=lambda x: x['strategies'][strat]['return_pct'], reverse=True)
        print(f"{'排名':<4} {'股票名称':<12} {'代码':<12} {'行业':<10} {'策略收益':>10} {'买入持有':>10}")
        for i, r in enumerate(stock_ranked[:10], 1):
            s = r['strategies'][strat]
            print(f"{i:<4} {r['name']:<12} {r['code']:<12} {r['industry']:<10} {s['return_pct']:>+9.2f}% {s['buy_hold_return']:>+9.2f}%")
    
    # 保存
    save_results(all_results, strategy_stats, scores, elapsed_time)


def save_results(all_results, strategy_stats, scores, elapsed_time):
    output_dir = Path(__file__).parent.parent / "results" / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    data = {
        'strategy_comparison': '双线战法+更多指标极致优化',
        'backtest_period': '2024-05-17到2026-05-17',
        'timestamp': timestamp,
        'strategy_definitions': STRATEGY_NAMES,
        'strategy_stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()} for k, v in strategy_stats.items()},
        'scores': {k: float(v) for k, v in scores.items()},
        'best_strategy': max(scores, key=scores.get),
        'elapsed_seconds': elapsed_time,
        'total_stocks': len(all_results),
    }
    
    report_file = output_dir / f"ultimate_strategy_comparison_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 回测结果已保存: {report_file}")


if __name__ == "__main__":
    run_multi_strategy_backtest()
