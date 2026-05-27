"""
双线战法今日扫描器
扫描A股热门股票，找出符合B1买点或金叉的股票
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

# 关注的股票池（热门A股）
WATCHLIST = [
    ("600519.SH", "贵州茅台"),
    ("601318.SH", "中国平安"),
    ("600036.SH", "招商银行"),
    ("000001.SZ", "平安银行"),
    ("000333.SZ", "美的集团"),
    ("002415.SZ", "海康威视"),
    ("300750.SZ", "宁德时代"),
    ("300059.SZ", "东方财富"),
    ("600570.SH", "恒生电子"),
    ("600362.SH", "江西铜业"),
    ("601899.SH", "紫金矿业"),
    ("601668.SH", "中国建筑"),
    ("000858.SZ", "五粮液"),
    ("000651.SZ", "格力电器"),
    ("601888.SH", "中国中免"),
]


def get_stock_data(ts_code):
    """获取股票数据"""
    try:
        df = pro.daily(ts_code=ts_code, start_date='20230101', end_date='20260516')
        
        if df is None or len(df) < 60:
            return None
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        
        return df
    except Exception as e:
        return None


def calculate_signals(df):
    """计算双线信号"""
    # 复制避免修改原数据
    df = df.copy()
    
    # 列名映射
    df['Close'] = df['close']
    df['High'] = df['high']
    df['Low'] = df['low']
    df['Volume'] = df['vol']
    
    # 白线：EMA(EMA(C,10),10)
    df['white_line'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['white_line'] = df['white_line'].ewm(span=10, adjust=False).mean()
    
    # 黄线：(MA14 + MA28 + MA57 + MA114) / 4
    df['ma14'] = df['Close'].rolling(window=14).mean()
    df['ma28'] = df['Close'].rolling(window=28).mean()
    df['ma57'] = df['Close'].rolling(window=57).mean()
    df['ma114'] = df['Close'].rolling(window=114).mean()
    df['yellow_line'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4
    
    # KDJ
    low_list = df['Low'].rolling(window=9, min_periods=1).min()
    high_list = df['High'].rolling(window=9, min_periods=1).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    df['J'] = df['J'].clip(0, 100)
    
    # 金叉死叉
    df['cross_up'] = (df['white_line'] > df['yellow_line']) & \
                     (df['white_line'].shift(1) <= df['yellow_line'].shift(1))
    df['cross_down'] = (df['white_line'] < df['yellow_line']) & \
                       (df['white_line'].shift(1) >= df['yellow_line'].shift(1))
    
    # 位置判断
    df['above_yellow'] = df['Close'] >= df['yellow_line']
    df['above_white'] = df['Close'] >= df['white_line']
    df['between_lines'] = (df['Close'] >= df['yellow_line']) & (df['Close'] <= df['white_line'])
    
    # 成交量
    df['vol_prev'] = df['Volume'].shift(1)
    df['vol_ratio'] = df['Volume'] / df['vol_prev']
    
    # 简化版B1信号
    df['b1_candidate'] = (
        (df['white_line'] > df['yellow_line']) &  # 白线在上
        (df['J'] < 15) &  # KDJ超卖
        (df['between_lines']) &  # 在白黄线之间
        (df['vol_ratio'] < 0.8)  # 缩量回调
    ).astype(int)
    
    return df


def scan_stocks():
    """扫描所有股票"""
    print("="*100)
    print("📊 双线战法今日扫描器")
    print("扫描时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*100)
    
    results = []
    buy_candidates = []
    
    for ts_code, name in WATCHLIST:
        print(f"\n正在分析: {name} ({ts_code})...")
        
        df = get_stock_data(ts_code)
        
        if df is None or len(df) < 60:
            print(f"  ⚠️ 数据不足，跳过")
            continue
        
        df = calculate_signals(df)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 信号评分
        score = 0
        signals = []
        
        # 今日是否金叉
        if latest['cross_up']:
            score += 10
            signals.append("✅ 今日金叉")
        
        # 白线在黄线之上
        if latest['white_line'] > latest['yellow_line']:
            score += 5
            signals.append("📈 白线在黄线上")
        
        # 在白黄线之间
        if latest['between_lines']:
            score += 3
            signals.append("📍 在白黄线区间")
        
        # KDJ超卖
        if latest['J'] < 15:
            score += 4
            signals.append(f"⚡ KDJ-J: {latest['J']:.1f} (超卖)")
        
        # B1候选
        if latest['b1_candidate'] == 1:
            score += 8
            signals.append("🎯 B1候选信号")
        
        # 今日涨幅
        change_pct = (latest['Close'] - prev['Close']) / prev['Close'] * 100
        
        result = {
            'code': ts_code,
            'name': name,
            'price': latest['Close'],
            'change_pct': change_pct,
            'white_line': latest['white_line'],
            'yellow_line': latest['yellow_line'],
            'J': latest['J'],
            'vol_ratio': latest['vol_ratio'],
            'score': score,
            'signals': signals
        }
        
        results.append(result)
        
        if score >= 5:
            buy_candidates.append(result)
        
        # 显示结果
        if signals:
            print(f"  💎 {name}")
            print(f"    价格: {latest['Close']:.2f} ({change_pct:+.2f}%)")
            print(f"    白线: {latest['white_line']:.2f}, 黄线: {latest['yellow_line']:.2f}")
            print(f"    KDJ-J: {latest['J']:.1f}")
            print(f"    成交量比: {latest['vol_ratio']:.2f}")
            print(f"    信号: {', '.join(signals)}")
        else:
            print(f"  {name}: 无显著信号")
    
    print("\n" + "="*100)
    print("🎯 值得关注的股票（按评分排序）")
    print("="*100)
    
    buy_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    for i, stock in enumerate(buy_candidates, 1):
        print(f"{i:2d}. {stock['name']:12s} ({stock['code']:10s})")
        print(f"    价格: {stock['price']:8.2f}  涨跌: {stock['change_pct']:+6.2f}%")
        print(f"    评分: {stock['score']:3d}  信号: {', '.join(stock['signals'])}")
        print()
    
    # 保存结果
    save_results(results, buy_candidates)
    
    return buy_candidates


def save_results(results, buy_candidates):
    """保存扫描结果"""
    output_dir = Path(__file__).parent.parent / "results" / "double_line_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    data = {
        'scan_time': datetime.now().isoformat(),
        'all_stocks': results,
        'buy_candidates': buy_candidates
    }
    
    report_file = output_dir / f"scan_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 扫描结果已保存: {report_file}")


if __name__ == "__main__":
    scan_stocks()
