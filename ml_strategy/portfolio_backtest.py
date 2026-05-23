import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Position:
    ts_code: str
    name: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    entry_prob: float = 0.0
    peak_price: float = 0.0
    entry_atr: float = 0.0

    @property
    def value(self):
        return self.shares * self.current_price

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price


@dataclass
class Trade:
    ts_code: str
    name: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    exit_reason: str = ''
    entry_prob: float = 0.0

    @property
    def profit_pct(self):
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def profit_abs(self):
        return self.shares * (self.exit_price - self.entry_price)

    @property
    def hold_days(self):
        return (self.exit_date - self.entry_date).days


class PortfolioBacktester:
    def __init__(self, initial_cash: float = 10000000, max_stocks: int = 3,
                 commission_rate: float = 0.0003, stamp_duty_rate: float = 0.001,
                 slippage_rate: float = 0.0005, position_size_pct: float = 0.25,
                 catboost_threshold: float = 0.65,
                 impact_model: str = 'sqrt', impact_coefficient: float = 0.4,
                 spread_half: float = 0.001):
        self.initial_cash = initial_cash
        self.max_stocks = max_stocks
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage_rate = slippage_rate
        self.position_size_pct = position_size_pct
        self.catboost_threshold = catboost_threshold
        self.impact_model = impact_model
        self.impact_coefficient = impact_coefficient
        self.spread_half = spread_half

        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

    def calculate_impact_slippage(self, shares: int, price: float,
                                  daily_volume: float, daily_volatility: float) -> float:
        if self.impact_model == 'sqrt':
            participation_rate = shares / daily_volume if daily_volume > 0 else 1.0
            impact = self.impact_coefficient * daily_volatility * math.sqrt(participation_rate)
            total_slippage = impact + self.spread_half
            total_slippage = min(total_slippage, 0.02)
            return total_slippage
        elif self.impact_model == 'fixed':
            return self.slippage_rate
        return self.slippage_rate

    def calculate_costs(self, amount: float, is_buy: bool = True,
                        include_slippage: bool = True,
                        shares: int = 0, price: float = 0.0,
                        daily_volume: float = 0, daily_volatility: float = 0.0) -> float:
        commission = amount * self.commission_rate
        commission = max(commission, 5.0)

        stamp_duty = 0.0
        if not is_buy:
            stamp_duty = amount * self.stamp_duty_rate

        slippage = 0.0
        if include_slippage:
            if daily_volume > 0 and daily_volatility > 0 and shares > 0 and price > 0:
                slippage_fraction = self.calculate_impact_slippage(
                    shares, price, daily_volume, daily_volatility)
                slippage = amount * slippage_fraction
            else:
                slippage = amount * self.slippage_rate

        total_cost = commission + stamp_duty + slippage
        return total_cost

    def get_available_slot_count(self) -> int:
        return self.max_stocks - len(self.positions)

    def can_buy(self) -> bool:
        return self.get_available_slot_count() > 0 and self.cash > 1000

    def buy(self, ts_code: str, name: str, price: float, date: pd.Timestamp,
            prob: float = 0.0, atr: float = 0.0,
            daily_volume: float = 0, daily_volatility: float = 0.02) -> Optional[Position]:
        if not self.can_buy():
            return None

        target_value = self.initial_cash * self.position_size_pct
        target_value = min(target_value, self.cash * 0.9)

        shares = int(target_value / price / 100) * 100
        if shares <= 0:
            return None

        use_impact = (self.impact_model == 'sqrt'
                      and daily_volume > 0
                      and daily_volatility > 0)

        if use_impact:
            slippage_fraction = self.calculate_impact_slippage(
                shares, price, daily_volume, daily_volatility)
            effective_price = price * (1 + slippage_fraction)
            amount = shares * effective_price
            cost = self.calculate_costs(amount, is_buy=True, include_slippage=False)
        else:
            amount = shares * price
            cost = self.calculate_costs(amount, is_buy=True)

        total_spend = amount + cost
        if total_spend > self.cash:
            return None

        self.cash -= total_spend

        pos = Position(
            ts_code=ts_code,
            name=name,
            entry_date=date,
            entry_price=price,
            shares=shares,
            entry_prob=prob,
            peak_price=price,
            entry_atr=atr,
        )

        self.positions[ts_code] = pos
        return pos

    def sell(self, ts_code: str, price: float, date: pd.Timestamp,
             reason: str = '',
             daily_volume: float = 0, daily_volatility: float = 0.02) -> Optional[Trade]:
        if ts_code not in self.positions:
            return None

        pos = self.positions[ts_code]

        use_impact = (self.impact_model == 'sqrt'
                      and daily_volume > 0
                      and daily_volatility > 0)

        if use_impact:
            slippage_fraction = self.calculate_impact_slippage(
                pos.shares, price, daily_volume, daily_volatility)
            effective_price = price * (1 - slippage_fraction)
            amount = pos.shares * effective_price
            cost = self.calculate_costs(amount, is_buy=False, include_slippage=False)
        else:
            amount = pos.shares * price
            cost = self.calculate_costs(amount, is_buy=False)

        self.cash += (amount - cost)

        trade = Trade(
            ts_code=ts_code,
            name=pos.name,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=price,
            shares=pos.shares,
            exit_reason=reason,
            entry_prob=pos.entry_prob,
        )

        self.trades.append(trade)
        del self.positions[ts_code]
        return trade

    def update_equity(self, date: pd.Timestamp, stock_prices: Dict[str, float]):
        total_equity = self.cash

        for ts_code, pos in self.positions.items():
            current_price = stock_prices.get(ts_code, pos.entry_price)
            if current_price > pos.peak_price:
                pos.peak_price = current_price
            total_equity += pos.shares * current_price

        self.equity_curve.append({
            'date': date,
            'cash': self.cash,
            'position_value': total_equity - self.cash,
            'total_equity': total_equity,
            'position_count': len(self.positions),
        })

    def get_portfolio_value(self, stock_prices: Dict[str, float]) -> float:
        total = self.cash
        for ts_code, pos in self.positions.items():
            current_price = stock_prices.get(ts_code, pos.entry_price)
            total += pos.shares * current_price
        return total

    def finalize(self, last_date: pd.Timestamp, stock_prices: Dict[str, float]):
        remaining_ts_codes = list(self.positions.keys())
        for ts_code in remaining_ts_codes:
            price = stock_prices.get(ts_code, 0.0)
            if price > 0:
                self.sell(ts_code, price, last_date, reason='force_close')

        self.update_equity(last_date, stock_prices)

    def get_summary(self) -> Dict:
        if not self.equity_curve:
            return {}

        df_equity = pd.DataFrame(self.equity_curve).set_index('date')
        total_equity = df_equity['total_equity']

        total_return = (total_equity.iloc[-1] - self.initial_cash) / self.initial_cash * 100

        daily_returns = total_equity.pct_change().dropna()

        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.0

        if volatility > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        drawdown = (total_equity.cummax() - total_equity) / total_equity.cummax()
        max_drawdown = drawdown.max() * 100 if len(drawdown) > 0 else 0.0

        if len(self.trades) > 0:
            win_trades = [t for t in self.trades if t.profit_pct > 0]
            win_rate = len(win_trades) / len(self.trades) * 100
            avg_profit = np.mean([t.profit_pct for t in self.trades])
            avg_win = np.mean([t.profit_pct for t in win_trades]) if win_trades else 0.0
            loss_trades = [t for t in self.trades if t.profit_pct <= 0]
            avg_loss = np.mean([t.profit_pct for t in loss_trades]) if loss_trades else 0.0
            profit_factor = (sum([t.profit_abs for t in win_trades]) /
                             abs(sum([t.profit_abs for t in loss_trades]))) if loss_trades else float('inf')
            avg_hold = np.mean([t.hold_days for t in self.trades])
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            avg_hold = 0.0

        return {
            'total_return': float(total_return),
            'annual_return': float((1 + total_return / 100) ** (252 / len(df_equity)) - 1) * 100,
            'volatility': float(volatility * 100),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'trade_count': len(self.trades),
            'win_rate': float(win_rate),
            'avg_trade_profit': float(avg_profit),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'avg_hold_days': float(avg_hold),
            'final_equity': float(total_equity.iloc[-1]),
        }

    def get_trade_details(self) -> List[Dict]:
        return [
            {
                'code': t.ts_code,
                'name': t.name,
                'entry_date': str(t.entry_date.date()),
                'exit_date': str(t.exit_date.date()),
                'entry_price': float(t.entry_price),
                'exit_price': float(t.exit_price),
                'shares': t.shares,
                'profit_pct': float(t.profit_pct),
                'profit_abs': float(t.profit_abs),
                'hold_days': t.hold_days,
                'exit_reason': t.exit_reason,
                'entry_prob': float(t.entry_prob),
            }
            for t in self.trades
        ]
