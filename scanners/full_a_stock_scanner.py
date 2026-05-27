"""
全A股双线战法扫描器
扫描所有A股股票，找出符合B1买点条件的股票
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
import time

load_dotenv(Path(__file__).parent.parent / ".env")

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(TUSHARE_TOKEN)


def get_all_a_stocks():
    """获取所有A股股票列表"""
    print("📊 获取所有A股股票列表...")
    try:
        # 获取基础股票列表
        stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,list_date')
        # 筛选A股（SH和SZ开头）
        a_stocks = stock_basic[(stock_basic['ts_code'].str.endswith('.SH')) | (stock_basic['ts_code'].str.endswith('.SZ'))]
        print(f"✅ 共获取 {len(a_stocks)} 只A股股票")
        return a_stocks
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")
        return None


def get_stock_data(ts_code):
    """获取单只股票数据"""
    try:
        df = pro.daily(ts_code=ts_code, start_date='20250101', end_date='20260516')
        
        if df is None or len(df) < 100:
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
    df['between_lines'] = (df['Close'] >= df['yellow_line']) & (df['Close'] <= df['white_line'])
    
    # 成交量
    df['vol_prev'] = df['Volume'].shift(1)
    df['vol_ratio'] = df['Volume'] / df['vol_prev']
    df['change_pct'] = df['Close'] / df['close'].shift(1) * 100 - 100
    
    # 放量大跌判断（近期20天内）
    df['big_down'] = (df['change_pct'] < -3) & (df['vol_ratio'] >= 1.5)
    df['recent_big_down'] = df['big_down'].rolling(window=20, min_periods=1).max()
    
    # 成交量收缩判断（近10天）
    df['high_vol_days'] = (df['vol_ratio'] >= 1.5).rolling(window=10, min_periods=1).sum()
    
    return df


def scan_all_a_stocks():
    """扫描全A股"""
    print("="*100)
    print("📊 全A股双线战法扫描器")
    print("扫描时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*100)
    
    # 获取股票列表
    stock_list = get_all_a_stocks()
    if stock_list is None:
        print("❌ 无法获取股票列表")
        return
    
    results = []
    buy_candidates = []
    total_stocks = len(stock_list)
    processed = 0
    skipped = 0
    
    start_time = time.time()
    
    for _, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        industry = row.get('industry', '未知')
        
        processed += 1
        
        if processed % 100 == 0:
            elapsed = time.time() - start_time
            print(f"\n进度: {processed}/{total_stocks} ({processed/total_stocks*100:.1f}%)")
            print(f"已找到 {len(buy_candidates)} 只符合条件的股票")
        
        df = get_stock_data(ts_code)
        
        if df is None:
            skipped += 1
            continue
        
        df = calculate_signals(df)
        
        if len(df) < 100:
            skipped += 1
            continue
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # B1条件检查
        condition1 = latest['white_line'] > latest['yellow_line']  # 白线在上
        condition2 = latest['J'] < 15  # KDJ超卖
        condition3 = latest['vol_ratio'] < 0.8  # 缩量回调
        condition4 = latest['between_lines']  # 在白黄线区间
        condition5 = latest['recent_big_down'] == 0  # 近期无放量大跌
        condition6 = latest['high_vol_days'] <= 3  # 成交量收缩
        
        # 评分
        score = 0
        signals = []
        
        if condition1:
            score += 5
            signals.append("白线在上")
        
        if condition2:
            score += 4
            signals.append(f"KDJ-J:{latest['J']:.1f}")
        
        if condition3:
            score += 3
            signals.append("缩量回调")
        
        if condition4:
            score += 3
            signals.append("在白黄区间")
        
        if condition5:
            score += 2
            signals.append("无放量大跌")
        
        if condition6:
            score += 2
            signals.append("量能收缩")
        
        # 今日涨幅
        change_pct = (latest['Close'] - prev['Close']) / prev['Close'] * 100
        
        result = {
            'code': ts_code,
            'name': name,
            'industry': industry,
            'price': latest['Close'],
            'change_pct': change_pct,
            'white_line': latest['white_line'],
            'yellow_line': latest['yellow_line'],
            'J': latest['J'],
            'vol_ratio': latest['vol_ratio'],
            'score': score,
            'signals': signals,
            'is_b1': condition1 and condition2 and condition3 and condition4 and condition5 and condition6
        }
        
        results.append(result)
        
        if score >= 10:
            buy_candidates.append(result)
            print(f"✨ 发现候选: {name} ({ts_code}) 评分:{score} 价格:{latest['Close']:.2f}")
    
    elapsed_time = time.time() - start_time
    
    # 排序
    buy_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 输出结果
    print("\n" + "="*100)
    print("🎯 全A股扫描结果")
    print("="*100)
    print(f"扫描股票总数: {total_stocks}")
    print(f"成功处理: {processed - skipped}")
    print(f"跳过（数据不足）: {skipped}")
    print(f"符合条件: {len(buy_candidates)}")
    print(f"耗时: {elapsed_time:.2f}秒")
    print("\n" + "="*100)
    print(f"{'排名':>4} {'股票':<12} {'代码':<12} {'行业':<10} {'价格':>8} {'涨跌':>8} {'评分':>6} {'信号'}")
    print("-"*100)
    
    for i, stock in enumerate(buy_candidates[:20], 1):
        print(f"{i:>4}. {stock['name']:<12} {stock['code']:<12} {stock['industry']:<10} "
              f"{stock['price']:>8.2f} {stock['change_pct']:>+7.2f}% {stock['score']:>6} "
              f"{', '.join(stock['signals'])}")
    
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
        'total_stocks': len(results),
        'buy_candidates_count': len(buy_candidates),
        'all_stocks': results,
        'buy_candidates': buy_candidates
    }
    
    report_file = output_dir / f"full_a_stock_scan_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 扫描结果已保存: {report_file}")


if __name__ == "__main__":
    scan_all_a_stocks()
