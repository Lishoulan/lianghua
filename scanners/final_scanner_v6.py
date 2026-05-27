"""
严格版全A股扫描器 - 最终修正版 v3
时间范围缩短为最近7天！
"""
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import tushare as ts
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(TUSHARE_TOKEN)


def get_all_a_stocks():
    print("=" * 100)
    print("📊 获取所有A股股票列表...")
    stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,list_date')
    a_stocks = stock_basic[(stock_basic['ts_code'].str.endswith('.SH')) | (stock_basic['ts_code'].str.endswith('.SZ'))]
    print(f"✅ 共获取 {len(a_stocks)} 只A股股票")
    return a_stocks


def calculate_corrected_signals(df):
    df = df.copy()
    df['Close'] = df['close']
    df['Open'] = df['open']
    df['High'] = df['high']
    df['Low'] = df['low']
    df['Volume'] = df['vol']
    
    df['white_line'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['white_line'] = df['white_line'].ewm(span=10, adjust=False).mean()
    
    df['ma14'] = df['Close'].rolling(window=14).mean()
    df['ma28'] = df['Close'].rolling(window=28).mean()
    df['ma57'] = df['Close'].rolling(window=57).mean()
    df['ma114'] = df['Close'].rolling(window=114).mean()
    df['yellow_line'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4
    
    low_list = df['Low'].rolling(window=9, min_periods=1).min()
    high_list = df['High'].rolling(window=9, min_periods=1).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    df['J'] = df['J'].clip(0, 100)
    
    df['between_lines'] = (df['Close'] >= df['yellow_line']) & (df['Close'] <= df['white_line'])
    
    df['vol_prev'] = df['Volume'].shift(1)
    df['vol_ratio'] = df['Volume'] / df['vol_prev']
    
    df['prev_Close'] = df['Close'].shift(1)
    df['price_up'] = df['Close'] > df['prev_Close']
    df['price_down'] = df['Close'] < df['prev_Close']
    
    df['kline_yin'] = df['Open'] > df['Close']
    df['kline_yang'] = df['Open'] < df['Close']
    
    df['fake_yin_yang'] = df['price_up'] & df['kline_yin']
    df['real_yin'] = df['price_down'] | df['fake_yin_yang']
    df['real_yang'] = df['price_up'] & df['kline_yang']
    
    df['yin_with_volume_up'] = df['real_yin'] & (df['Volume'] > df['vol_prev'])
    df['yang_double_volume'] = df['real_yang'] & (df['Volume'] >= df['vol_prev'] * 2)
    
    df['limit_up'] = df['Close'] / df['prev_Close'] >= 1.095
    df['limit_up_shrink'] = df['limit_up'] & (df['vol_ratio'] < 1.0)
    df['recent_limit_up_shrink'] = df['limit_up_shrink'].rolling(window=7, min_periods=1).sum()
    
    df['recent_yin_volume_up'] = df['yin_with_volume_up'].rolling(window=7, min_periods=1).max()
    df['recent_yang_double'] = df['yang_double_volume'].rolling(window=30, min_periods=1).sum()
    
    return df


def scan_final_v3():
    print("=" * 100)
    print("📊 严格版全A股双线战法扫描器 - 最终修正版 v3")
    print("扫描时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 100)
    print("【硬条件必须全部满足】：")
    print("  1. 白线在黄线上（右侧交易）")
    print("  2. KDJ-J < 13（超卖）")
    print("  3. 缩量回调（量比<=0.8）")
    print("  4. 在白黄线区间")
    print("  5. 最近7天内无放量阴线（从5天调整）")
    print("  6. 最近7天内无缩量涨停（从5天调整）")
    print("  7. 最近30天有翻倍量阳线（加分）")
    print("=" * 100)
    
    stock_list = get_all_a_stocks()
    buy_candidates = []
    total = len(stock_list)
    processed = 0
    
    start_time = datetime.now()
    
    for _, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        industry = row.get('industry', '未知')
        
        processed += 1
        
        if processed % 200 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n进度: {processed}/{total} ({processed/total*100:.1f}%)")
            print(f"已找到 {len(buy_candidates)} 只符合条件的股票")
        
        df = pro.daily(ts_code=ts_code, start_date='20250101', end_date='20260516')
        
        if df is None or len(df) < 130:
            continue
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        df = calculate_corrected_signals(df)
        
        if len(df) < 130:
            continue
        
        latest = df.iloc[-1]
        
        if pd.isna(latest['yellow_line']) or pd.isna(latest['white_line']):
            continue
        
        has_bad_yin = bool(latest['recent_yin_volume_up'] > 0)
        has_double_yang = bool(latest['recent_yang_double'] >= 2)
        has_limit_up_shrink = bool(latest['recent_limit_up_shrink'] > 0)
        
        condition1 = latest['white_line'] > latest['yellow_line']
        condition2 = latest['J'] < 13
        condition3 = latest['vol_ratio'] <= 0.8
        condition4 = bool(latest['between_lines'])
        condition5 = has_bad_yin == 0
        condition6 = has_limit_up_shrink == 0
        
        if not (condition1 and condition2 and condition3 and condition4 and condition5 and condition6):
            continue
        
        score = 30
        signals = ["白线在上", f"KDJ-J:{latest['J']:.1f}", "缩量回调", "在白黄区间", "7天内无放量阴线", "7天内无缩量涨停"]
        
        if has_double_yang:
            score += 10
            signals.append("30天内有翻倍量阳线")
        
        buy_candidates.append({
            'code': ts_code,
            'name': name,
            'industry': industry,
            'price': float(latest['Close']),
            'white_line': float(latest['white_line']),
            'yellow_line': float(latest['yellow_line']),
            'kdj_j': float(latest['J']),
            'has_double_yang': has_double_yang,
            'score': int(score),
            'signals': signals
        })
        print(f"  发现候选: {name} ({ts_code}) 评分:{score} 价格:{latest['Close']:.2f} J:{latest['J']:.1f}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    buy_candidates.sort(key=lambda x: (-x['score'], x['kdj_j']))
    
    print("\n" + "=" * 100)
    print("🎯 最终修正版v3扫描结果")
    print("=" * 100)
    print(f"扫描股票总数: {total}")
    print(f"成功处理: {processed}")
    print(f"符合所有硬条件: {len(buy_candidates)}")
    print(f"耗时: {elapsed:.2f}秒")
    print("\n" + "=" * 100)
    print(f"{'排名':>4} {'股票':<12} {'代码':<12} {'行业':<10} {'价格':>8} {'KDJ-J':>8} {'评分':>6} {'信号'}")
    print("-" * 100)
    
    for i, stock in enumerate(buy_candidates[:30], 1):
        print(f"{i:>4}. {stock['name']:<12} {stock['code']:<12} {stock['industry']:<10} {stock['price']:>8.2f} {stock['kdj_j']:>8.1f} {stock['score']:>6} {', '.join(stock['signals'])}")
    
    output_dir = Path(__file__).parent.parent / "results" / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {'scan_time': datetime.now().isoformat(), 'buy_candidates': buy_candidates}
    report_file = output_dir / f"final_scan_v3_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n扫描结果已保存: {report_file}")
    
    return buy_candidates


if __name__ == "__main__":
    scan_final_v3()
