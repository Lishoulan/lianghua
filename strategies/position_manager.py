"""
双线战法仓位管理系统
实现分批建仓、减仓、止损、止盈的完整仓位管理策略
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class PositionManager:
    """
    仓位管理器
    
    策略逻辑（循环操作法）：
    1. B1买入（第一笔底仓）
    2. 横盘或温和上涨持有
    3. 白黄区间加仓（可以把仓位买满）
    4. 加速后放飞（开始减仓）
    5. 跌破白线卖出
    6. 重新观察白黄区间建仓价值
    7. 循环...
    """
    
    def __init__(self, initial_cash=1000000):
        self.initial_cash = initial_cash
        self.reset()
    
    def reset(self):
        """重置状态"""
        self.cash = self.initial_cash
        self.position = 0  # 当前持仓股数
        self.position_value = 0  # 当前持仓市值
        self.avg_cost = 0  # 平均成本
        self.trades = []  # 交易记录
        self.positions = []  # 仓位变化记录
        self.portfolio_values = []  # 组合价值记录
        
        # 仓位管理参数
        self.max_position_ratio = 1.0  # 最大仓位比例（100%）
        self.position_ratio = 0  # 当前仓位比例
        self.has_position = False  # 是否有持仓
        
        # 建仓参数
        self.first_position_ratio = 0.3  # B1买入比例
        self.second_position_ratio = 0.3  # 白黄区间加仓比例
        self.third_position_ratio = 0.4  # 第三笔加仓比例
        self.position_level = 0  # 建仓层级 0=空仓, 1=B1, 2=区间, 3=满仓
        
        # 止损参数
        self.stop_loss_ratio = 0.05  # 5%止损
        self.trailing_stop_ratio = 0.08  # 8%移动止损
        
    def can_buy(self, price, ratio):
        """检查是否可以买入"""
        needed_cash = self.cash * ratio
        shares = int(needed_cash / price)
        return shares > 0
    
    def buy(self, price, ratio, signal_type, date):
        """买入"""
        if self.position_ratio >= self.max_position_ratio:
            return False
        
        needed_cash = self.cash * ratio
        shares = int(needed_cash / price)
        
        if shares <= 0:
            return False
        
        cost = shares * price
        self.cash -= cost
        self.position += shares
        self.position_value = self.position * price
        
        # 更新平均成本
        total_cost = self.position * self.avg_cost + cost
        self.position += shares
        self.avg_cost = total_cost / self.position
        
        self.has_position = True
        self.position_ratio = self.position * price / self.initial_cash
        
        # 记录交易
        self.trades.append({
            'date': date,
            'type': 'BUY',
            'signal': signal_type,
            'price': price,
            'shares': shares,
            'cash': cost,
            'position_ratio': self.position_ratio,
            'avg_cost': self.avg_cost
        })
        
        return True
    
    def sell(self, price, ratio, signal_type, date):
        """卖出"""
        if self.position == 0:
            return False
        
        shares = int(self.position * ratio)
        if shares <= 0:
            return False
        
        revenue = shares * price
        profit = (price - self.avg_cost) * shares
        profit_pct = (price - self.avg_cost) / self.avg_cost * 100
        
        self.cash += revenue
        self.position -= shares
        self.position_value = self.position * price
        
        if self.position == 0:
            self.avg_cost = 0
            self.has_position = False
            self.position_ratio = 0
            self.position_level = 0
        
        # 记录交易
        self.trades.append({
            'date': date,
            'type': 'SELL',
            'signal': signal_type,
            'price': price,
            'shares': shares,
            'cash': revenue,
            'profit': profit,
            'profit_pct': profit_pct,
            'position_ratio': self.position_ratio if self.position > 0 else 0
        })
        
        return True
    
    def update_stop_loss(self, current_price):
        """更新止损位"""
        if not self.has_position:
            return None
        
        # 成本止损
        cost_stop = self.avg_cost * (1 - self.stop_loss_ratio)
        
        # 移动止损（从最高点回撤）
        if len(self.trades) > 0:
            buy_trades = [t for t in self.trades if t['type'] == 'BUY']
            if buy_trades:
                max_price = max([t['price'] for t in buy_trades])
                trailing_stop = max_price * (1 - self.trailing_stop_ratio)
                return max(cost_stop, trailing_stop)
        
        return cost_stop
    
    def get_portfolio_value(self, current_price):
        """获取组合价值"""
        return self.cash + self.position * current_price


def backtest_with_position_management(df, strategy_col, initial_cash=1000000):
    """
    使用仓位管理的回测
    
    策略逻辑：
    1. 买入信号 -> 分批建仓（B1 30% + 区间 30% + 确认 40%）
    2. 持有过程中根据信号加减仓
    3. 跌破止损位 -> 全部清仓
    4. 卖出信号 -> 分批减仓
    """
    pm = PositionManager(initial_cash)
    df['portfolio_value'] = 0.0
    
    for i, row in df.iterrows():
        price = row['Close']
        signal = row.get(strategy_col, 0)
        
        # 更新止损位
        stop_loss_price = pm.update_stop_loss(price)
        
        # 1. 检查止损
        if stop_loss_price and price <= stop_loss_price and pm.has_position:
            pm.sell(price, 1.0, 'STOP_LOSS', i)
            continue
        
        # 2. 买入信号
        if signal == 1 and not pm.has_position:
            # 买入B1（第一笔底仓）
            if pm.can_buy(price, pm.first_position_ratio):
                pm.buy(price, pm.first_position_ratio, 'B1_BUY', i)
            
            # 如果还有信号，继续加仓到区间
            if signal == 1 and pm.can_buy(price, pm.second_position_ratio):
                pm.buy(price, pm.second_position_ratio, 'ZONE_BUY', i)
            
            # 如果还有信号，满仓
            if signal == 1 and pm.can_buy(price, pm.third_position_ratio):
                pm.buy(price, pm.third_position_ratio, 'FULL_BUY', i)
        
        # 3. 持有过程中，根据白黄区间继续加仓
        elif signal == 1 and pm.has_position and pm.position_ratio < 0.8:
            # 每次加仓20%
            if pm.can_buy(price, 0.2):
                pm.buy(price, 0.2, 'ADD_BUY', i)
        
        # 4. 卖出信号
        elif signal == -1 and pm.has_position:
            # 分批减仓
            if pm.position_ratio > 0.5:
                pm.sell(price, 0.5, 'PART_SELL', i)
            else:
                pm.sell(price, 1.0, 'FULL_SELL', i)
        
        # 更新组合价值
        df.at[i, 'portfolio_value'] = pm.get_portfolio_value(price)
        pm.portfolio_values.append(pm.get_portfolio_value(price))
    
    # 最后清仓
    if pm.has_position:
        final_price = df.iloc[-1]['Close']
        pm.sell(final_price, 1.0, 'FINAL_SELL', df.index[-1])
    
    return df, pm.trades, pm.cash, pm


def backtest_with_double_line_position(df, initial_cash=1000000):
    """
    使用双线战法专属仓位管理的回测
    
    循环操作法完整实现：
    - B1买入（30%）
    - 横盘/温和上涨持有
    - 白黄区间加仓（30%）
    - 加速后放飞（开始减仓）
    - 跌破白线卖出
    """
    pm = PositionManager(initial_cash)
    
    # 白线、黄线
    white_line = df['white_line'].values
    yellow_line = df['yellow_line'].values
    ma60 = df['ma60'].values
    close_prices = df['Close'].values
    
    for i in range(len(df)):
        price = close_prices[i]
        wl = white_line[i] if not np.isnan(white_line[i]) else price
        yl = yellow_line[i] if not np.isnan(yellow_line[i]) else price
        m60 = ma60[i] if not np.isnan(ma60[i]) else price
        
        # 金叉死叉
        cross_up = row_get(df, 'cross_up', i)
        cross_down = row_get(df, 'cross_down', i)
        
        # 价格在白黄区间
        in_zone = (price <= wl) and (price >= yl)
        
        # 趋势判断
        bullish_trend = price > m60 and wl > m60
        
        # 更新止损
        stop_loss = pm.update_stop_loss(price)
        
        # 1. 止损检查
        if stop_loss and price <= stop_loss and pm.has_position:
            pm.sell(price, 1.0, 'STOP_LOSS', df.index[i])
            continue
        
        # 2. B1买入条件：趋势多头 + 价格在白黄区间 + 金叉准备
        if not pm.has_position and bullish_trend and in_zone:
            # B1买入30%
            if pm.can_buy(price, 0.3):
                pm.buy(price, 0.3, 'B1_ENTRY', df.index[i])
                pm.position_level = 1
        
        # 3. 区间加仓条件：已有B1 + 再次回踩 + 趋势仍多头
        elif pm.has_position and pm.position_level == 1 and in_zone and bullish_trend:
            # 区间加仓30%
            if pm.can_buy(price, 0.3):
                pm.buy(price, 0.3, 'ZONE_ADD', df.index[i])
                pm.position_level = 2
        
        # 4. 满仓确认：趋势加速 + 价格突破
        elif pm.has_position and pm.position_level == 2 and price > wl:
            # 满仓40%
            if pm.can_buy(price, 0.4):
                pm.buy(price, 0.4, 'FULL_POSITION', df.index[i])
                pm.position_level = 3
        
        # 5. 减仓条件：加速后跌破白线
        elif pm.has_position and pm.position_level >= 2 and price < wl:
            # 分批减仓
            if pm.position_ratio > 0.5:
                pm.sell(price, 0.5, 'REDUCE_50', df.index[i])
                pm.position_level = max(1, pm.position_level - 1)
            else:
                pm.sell(price, 1.0, 'EXIT_ALL', df.index[i])
                pm.position_level = 0
        
        # 6. 全部清仓条件：死叉
        if cross_down and pm.has_position:
            pm.sell(price, 1.0, 'CROSS_DOWN_EXIT', df.index[i])
            pm.position_level = 0
    
    # 最后清仓
    if pm.has_position:
        final_price = close_prices[-1]
        pm.sell(final_price, 1.0, 'FINAL_SELL', df.index[-1])
    
    return pm.trades, pm.cash, pm


def row_get(df, col, idx):
    """安全获取行数据"""
    if col not in df.columns:
        return False
    val = df[col].iloc[idx] if idx < len(df) else False
    if pd.isna(val):
        return False
    return bool(val)


def analyze_position_trades(trades):
    """分析交易记录"""
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'total_profit': 0
        }
    
    sell_trades = [t for t in trades if t['type'] in ('SELL', 'FULL_SELL', 'EXIT_ALL', 'FINAL_SELL', 'STOP_LOSS')]
    
    total_profit = sum([t.get('profit', 0) for t in sell_trades])
    winning_trades = len([t for t in sell_trades if t.get('profit', 0) > 0])
    losing_trades = len([t for t in sell_trades if t.get('profit', 0) < 0])
    
    return {
        'total_trades': len(sell_trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / len(sell_trades) * 100 if sell_trades else 0,
        'avg_profit': total_profit / len(sell_trades) if sell_trades else 0,
        'total_profit': total_profit
    }
