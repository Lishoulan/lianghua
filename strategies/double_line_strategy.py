"""
双线战法指标计算模块
包含白线、黄线计算，以及5种玩法的信号生成
"""

import pandas as pd
import numpy as np


def calculate_double_line_strategy(df):
    """
    计算双线战法指标
    
    参数:
        df - 包含OHLCV的DataFrame，索引为日期
    
    返回:
        df - 添加了所有指标的DataFrame
    """
    # 确保有OHLC列
    if 'Close' not in df.columns and 'close' in df.columns:
        df['Close'] = df['close']
    if 'High' not in df.columns and 'high' in df.columns:
        df['High'] = df['high']
    if 'Low' not in df.columns and 'low' in df.columns:
        df['Low'] = df['low']
    
    # 1. 计算白线：EMA(EMA(C,10),10)
    df['white_line'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['white_line'] = df['white_line'].ewm(span=10, adjust=False).mean()
    
    # 2. 计算黄线：(MA14 + MA28 + MA57 + MA114) / 4
    df['ma14'] = df['Close'].rolling(window=14).mean()
    df['ma28'] = df['Close'].rolling(window=28).mean()
    df['ma57'] = df['Close'].rolling(window=57).mean()
    df['ma114'] = df['Close'].rolling(window=114).mean()
    df['yellow_line'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4
    
    # 3. 金叉死叉信号
    df['cross_up'] = (df['white_line'] > df['yellow_line']) & \
                     (df['white_line'].shift(1) <= df['yellow_line'].shift(1))
    df['cross_down'] = (df['white_line'] < df['yellow_line']) & \
                       (df['white_line'].shift(1) >= df['yellow_line'].shift(1))
    
    # 4. 白线黄线相对距离
    df['line_distance'] = (df['white_line'] / df['yellow_line'] - 1) * 1000
    
    # 5. 成交量变化
    df['vol_ma5'] = df['Volume'].rolling(window=5).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_ma5']
    
    # 6. 价格变化
    df['price_change'] = df['Close'].pct_change()
    df['price_change_prev'] = df['Close'].shift(1).pct_change()
    
    # 7. KDJ指标
    df = calculate_kdj(df)
    
    # 8. N型结构判断（简化版）
    df = detect_n_structure(df)
    
    # 9. 生成5种玩法的信号
    df = generate_signals(df)
    
    # 10. 生成精确的B1信号
    df = generate_b1_signal(df)
    
    return df


def detect_n_structure(df, lookback=60):
    """
    检测N型结构
    """
    # 检测近期高点和低点
    df['high_20'] = df['High'].rolling(window=20, center=False).max()
    df['low_20'] = df['Low'].rolling(window=20, center=False).min()
    
    # 简化版N型结构：创新高后回踩再创新高
    df['new_high'] = df['High'] == df['high_20']
    df['new_low'] = df['Low'] == df['low_20']
    
    # 标记N型结构
    df['n_structure'] = False
    
    # 这里可以用更复杂的模式匹配，先简化处理
    df['price_above_yellow'] = df['Close'] > df['yellow_line']
    
    return df


def generate_signals(df):
    """
    生成5种玩法的交易信号
    """
    
    # ============================================
    # 玩法1: 金叉入场，死叉离场（无脑玩法）
    # ============================================
    df['play1_signal'] = 0
    df.loc[df['cross_up'], 'play1_signal'] = 1  # 买入
    df.loc[df['cross_down'], 'play1_signal'] = -1  # 卖出
    
    # ============================================
    # 玩法2: 死叉时是最后的离场时机
    # ============================================
    df['play2_signal'] = 0
    df.loc[df['cross_down'], 'play2_signal'] = -1  # 强制离场
    
    # ============================================
    # 玩法3: 死叉多时反而可能是极限买点（欲死叉未死叉）
    # TX1: 昨日白线在黄线上
    # TX2: 白线贴近黄线（距离 < 3）
    # TX3: 今日白线仍在黄线上
    # TX4: 昨日不是放量下跌
    # TX5: 今日放量上涨
    # ============================================
    df['tx1'] = (df['white_line'].shift(1) > df['yellow_line'].shift(1)).fillna(False)
    df['tx2'] = (df['line_distance'] < 3).fillna(False)
    df['tx3'] = (df['white_line'] > df['yellow_line']).fillna(False)
    df['tx4'] = ~((df['price_change_prev'] < 0) & (df['vol_ratio'].shift(1) > 1.2)).fillna(False)
    df['tx5'] = ((df['vol_ratio'] > 1.2) & (df['price_change'] > 0)).fillna(False)
    
    df['play3_signal'] = 0
    df.loc[df['tx1'] & df['tx2'] & df['tx3'] & df['tx4'] & df['tx5'], 'play3_signal'] = 1
    
    # ============================================
    # 玩法4: 白线黄线区间都是容错率高的买入区
    # 需要: 1. 异动后回到B1  2. 在白黄区间
    # ============================================
    df['between_lines'] = ((df['Close'] <= df['white_line']) & \
                          (df['Close'] >= df['yellow_line'])).fillna(False)
    
    # 简化版：之前有大涨异动，现在回到区间
    df['recent_spike'] = (df['price_change'].rolling(window=10).max() > 0.05).fillna(False)
    df['price_above_yellow'] = (df['price_above_yellow']).fillna(False)
    
    df['play4_signal'] = 0
    df.loc[df['between_lines'] & df['recent_spike'] & df['price_above_yellow'], 'play4_signal'] = 1
    
    # ============================================
    # 玩法5: 放量金叉后缩量回踩黄线
    # 条件: 金叉放量 -> 缩量回踩到黄线
    # ============================================
    df['golden_cross_volume'] = (df['cross_up']) & (df['vol_ratio'] > 1.5)
    df['retrace_to_yellow'] = (df['Close'] <= df['yellow_line'] * 1.02) & \
                              (df['Close'] >= df['yellow_line'] * 0.98) & \
                              (df['vol_ratio'] < 0.8)
    
    # 检测近期有放量金叉，现在回踩
    df['recent_golden_cross'] = df['golden_cross_volume'].rolling(window=20, min_periods=1).max()
    # 确保是布尔值
    df['recent_golden_cross'] = df['recent_golden_cross'].fillna(0).astype(bool)
    
    df['play5_signal'] = 0
    df.loc[df['recent_golden_cross'] & df['retrace_to_yellow'], 'play5_signal'] = 1
    
    return df


def backtest_strategy(df, signal_col, initial_cash=1000000):
    """
    回测单个策略
    
    参数:
        df - 数据
        signal_col - 信号列名
        initial_cash - 初始资金
    """
    cash = initial_cash
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = []
    
    for i, row in df.iterrows():
        current_price = row['Close']
        
        if pd.isna(current_price) or pd.isna(row[signal_col]):
            portfolio_values.append(cash + position * current_price if position > 0 else cash)
            continue
        
        # 买入信号
        if row[signal_col] == 1 and position == 0:
            # 全仓买入（简化）
            shares = int(cash / current_price)
            if shares > 0:
                position = shares
                entry_price = current_price
                cash -= shares * current_price
                trades.append({
                    'date': i,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'cash_used': shares * current_price
                })
        
        # 卖出信号
        elif row[signal_col] == -1 and position > 0:
            cash += position * current_price
            trades.append({
                'date': i,
                'type': 'SELL',
                'price': current_price,
                'shares': position,
                'cash_gained': position * current_price,
                'profit': (current_price - entry_price) * position
            })
            position = 0
            entry_price = 0
        
        # 更新组合价值
        portfolio_values.append(cash + position * current_price if position > 0 else cash)
    
    # 最后清仓
    if position > 0:
        final_price = df.iloc[-1]['Close']
        cash += position * final_price
        trades.append({
            'date': df.index[-1],
            'type': 'FINAL_SELL',
            'price': final_price,
            'shares': position,
            'cash_gained': position * final_price,
            'profit': (final_price - entry_price) * position
        })
    
    df['portfolio_value'] = portfolio_values
    
    return df, trades, cash


def calculate_kdj(df, n=9, m1=3, m2=3):
    """
    计算KDJ指标
    
    参数:
        df - 包含High, Low, Close的DataFrame
        n - RSV周期，默认9
        m1 - K值平滑周期，默认3
        m2 - D值平滑周期，默认3
    
    返回:
        df - 添加了KDJ指标的DataFrame
    """
    # 确保有High和Low列
    if 'High' not in df.columns and 'high' in df.columns:
        df['High'] = df['high']
    if 'Low' not in df.columns and 'low' in df.columns:
        df['Low'] = df['low']
    
    # 计算RSV (未成熟随机值)
    low_list = df['Low'].rolling(window=n, min_periods=1).min()
    high_list = df['High'].rolling(window=n, min_periods=1).max()
    
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)  # 填充NaN为中间值50
    
    # 计算K值（EMA平滑）
    df['K'] = rsv.ewm(com=m1-1, adjust=False).mean()
    
    # 计算D值（EMA平滑）
    df['D'] = df['K'].ewm(com=m2-1, adjust=False).mean()
    
    # 计算J值
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # J值限制在0-100之间
    df['J'] = df['J'].clip(0, 100)
    
    return df


def generate_b1_signal(df):
    """
    生成精确的B1信号
    
    B1定义:
    1. 放量上涨（前期上涨阶段放量，比昨日成交量大2倍以上）
    2. 顶部无放量阴线（确认不是出货）
    3. 缩量回调（回调时缩量，说明惜售）
    4. KDJ小于13（超卖区间，性价比高）
    
    参数:
        df - 包含所有指标的DataFrame
    
    返回:
        df - 添加了B1信号的DataFrame
    """
    # 初始化信号
    df['b1_signal'] = 0
    
    # 条件1: 白线在黄线之上（右侧交易）
    df['condition_1'] = df['white_line'] > df['yellow_line']
    
    # 条件2: J值小于13（超卖）
    df['condition_2'] = df['J'] < 13
    
    # 条件3: 缩量回调（成交量比昨日少一半以上）
    # 缩量 = 今日成交量 <= 昨日成交量 * 0.5
    df['volume_prev'] = df['Volume'].shift(1)
    df['condition_3'] = df['Volume'] <= df['volume_prev'] * 0.5
    
    # 条件4: 前期有放量上涨（过去10天内有放量上涨）
    # 放量上涨定义：涨幅>3% 且 成交量 >= 昨日成交量 * 2
    df['volume_2x'] = df['Volume'] >= df['volume_prev'] * 2
    df['big_up'] = (df['price_change'] > 0.03) & (df['volume_2x'])
    df['recent_big_up'] = df['big_up'].rolling(window=10, min_periods=1).max()
    df['condition_4'] = df['recent_big_up'] == 1
    
    # 条件5: 前期顶部无放量阴线（过去5天内没有放量大跌）
    # 放量大跌定义：跌幅>3% 且 成交量 >= 昨日成交量 * 2
    df['big_down'] = (df['price_change'] < -0.03) & (df['volume_2x'])
    df['recent_big_down'] = df['big_down'].rolling(window=5, min_periods=1).max()
    df['condition_5'] = df['recent_big_down'] == 0
    
    # 条件6: 价格在白黄线区间内
    df['between_lines'] = (df['Close'] >= df['yellow_line']) & (df['Close'] <= df['white_line'])
    df['condition_6'] = df['between_lines']
    
    # 综合所有条件生成B1信号
    df['b1_signal'] = (
        df['condition_1'] & 
        df['condition_2'] & 
        df['condition_3'] & 
        df['condition_4'] & 
        df['condition_5'] & 
        df['condition_6']
    ).astype(int)
    
    return df
