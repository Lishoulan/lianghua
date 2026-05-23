import sys
from pathlib import Path
import os
import json
import time
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "pip_libs"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(
            Path(__file__).parent / 'auto_trade.log', encoding='utf-8'
        ),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('AutoTrader')

POSITIONS_FILE = Path(__file__).parent / 'auto_positions.json'
TRADE_LOG_FILE = Path(__file__).parent / 'auto_trade_log.json'


class AutoTrader:

    BROKER_MAP = {
        'yh': '银河证券',
        'ht': '华泰证券',
        'gj': '国金证券',
        'gf': '广发证券',
        'th': '同花顺通用',
    }

    def __init__(self, broker='yh', exe_path=None, dry_run=True):
        self.broker = broker
        self.exe_path = exe_path
        self.dry_run = dry_run
        self.trader = None
        self.positions = self._load_positions()
        self.trade_log = self._load_trade_log()

        if not dry_run:
            self._connect()

    def _connect(self):
        try:
            import easytrader
            if self.exe_path:
                self.trader = easytrader.use(self.broker)
                self.trader.connect(self.exe_path)
            else:
                self.trader = easytrader.use(self.broker)
                self.trader.connect()
            logger.info(f"easytrader连接成功: {self.BROKER_MAP.get(self.broker, self.broker)}")
        except Exception as e:
            logger.error(f"easytrader连接失败: {e}")
            self.trader = None

    def _load_positions(self):
        if POSITIONS_FILE.exists():
            with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_positions(self):
        with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.positions, f, ensure_ascii=False, indent=2)

    def _load_trade_log(self):
        if TRADE_LOG_FILE.exists():
            with open(TRADE_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_trade_log(self):
        with open(TRADE_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.trade_log, f, ensure_ascii=False, indent=2)

    def _normalize_code(self, code):
        if '.' in code:
            return code.split('.')[0]
        return code

    def _get_market_prefix(self, code):
        pure_code = self._normalize_code(code)
        if pure_code.startswith('6'):
            return ''
        elif pure_code.startswith('0') or pure_code.startswith('3'):
            return ''
        return ''

    def buy(self, code, name, price, amount, reason=''):
        pure_code = self._normalize_code(code)
        amount = int(amount / 100) * 100
        if amount <= 0:
            logger.warning(f"买入数量为0，跳过: {name}({pure_code})")
            return False

        trade_record = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'BUY',
            'code': pure_code,
            'name': name,
            'price': price,
            'amount': amount,
            'reason': reason,
            'status': 'pending',
        }

        if self.dry_run:
            logger.info(f"[模拟买入] {name}({pure_code}) 价格:{price} 数量:{amount} 原因:{reason}")
            trade_record['status'] = 'dry_run'
            self.trade_log.append(trade_record)
            self._save_trade_log()
            self.positions[pure_code] = {
                'name': name,
                'entry_price': price,
                'amount': amount,
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
            }
            self._save_positions()
            return True

        if self.trader is None:
            logger.error("easytrader未连接，无法下单")
            trade_record['status'] = 'failed_no_connection'
            self.trade_log.append(trade_record)
            self._save_trade_log()
            return False

        try:
            result = self.trader.buy(security=pure_code, price=price, amount=amount)
            logger.info(f"[实盘买入] {name}({pure_code}) 价格:{price} 数量:{amount} 结果:{result}")
            trade_record['status'] = 'submitted'
            trade_record['result'] = str(result)
            self.trade_log.append(trade_record)
            self._save_trade_log()
            self.positions[pure_code] = {
                'name': name,
                'entry_price': price,
                'amount': amount,
                'entry_date': datetime.now().strftime('%Y-%m-%d'),
            }
            self._save_positions()
            return True
        except Exception as e:
            logger.error(f"[买入失败] {name}({pure_code}): {e}")
            trade_record['status'] = 'failed'
            trade_record['error'] = str(e)
            self.trade_log.append(trade_record)
            self._save_trade_log()
            return False

    def sell(self, code, name, price, amount, reason=''):
        pure_code = self._normalize_code(code)
        amount = int(amount / 100) * 100
        if amount <= 0:
            logger.warning(f"卖出数量为0，跳过: {name}({pure_code})")
            return False

        trade_record = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'SELL',
            'code': pure_code,
            'name': name,
            'price': price,
            'amount': amount,
            'reason': reason,
            'status': 'pending',
        }

        if self.dry_run:
            logger.info(f"[模拟卖出] {name}({pure_code}) 价格:{price} 数量:{amount} 原因:{reason}")
            trade_record['status'] = 'dry_run'
            self.trade_log.append(trade_record)
            self._save_trade_log()
            if pure_code in self.positions:
                del self.positions[pure_code]
                self._save_positions()
            return True

        if self.trader is None:
            logger.error("easytrader未连接，无法下单")
            trade_record['status'] = 'failed_no_connection'
            self.trade_log.append(trade_record)
            self._save_trade_log()
            return False

        try:
            result = self.trader.sell(security=pure_code, price=price, amount=amount)
            logger.info(f"[实盘卖出] {name}({pure_code}) 价格:{price} 数量:{amount} 结果:{result}")
            trade_record['status'] = 'submitted'
            trade_record['result'] = str(result)
            self.trade_log.append(trade_record)
            self._save_trade_log()
            if pure_code in self.positions:
                del self.positions[pure_code]
                self._save_positions()
            return True
        except Exception as e:
            logger.error(f"[卖出失败] {name}({pure_code}): {e}")
            trade_record['status'] = 'failed'
            trade_record['error'] = str(e)
            self.trade_log.append(trade_record)
            self._save_trade_log()
            return False

    def sell_all(self, code, name, price, reason=''):
        pure_code = self._normalize_code(code)
        if pure_code in self.positions:
            amount = self.positions[pure_code].get('amount', 0)
            return self.sell(code, name, price, amount, reason)
        else:
            if not self.dry_run and self.trader is not None:
                try:
                    balance = self.get_position(pure_code)
                    if balance and balance.get('enable_amount', 0) > 0:
                        return self.sell(code, name, price, int(balance['enable_amount']), reason)
                except Exception:
                    pass
            logger.warning(f"未找到持仓记录: {name}({pure_code})")
            return False

    def get_balance(self):
        if self.dry_run:
            logger.info("[模拟] 查询资金")
            return {'asset_balance': 500000, 'enable_balance': 250000}
        if self.trader is None:
            return None
        try:
            return self.trader.balance
        except Exception as e:
            logger.error(f"查询资金失败: {e}")
            return None

    def get_position(self, code=None):
        if self.dry_run:
            logger.info(f"[模拟] 查询持仓 code={code}")
            if code:
                pure_code = self._normalize_code(code)
                return self.positions.get(pure_code, None)
            return self.positions
        if self.trader is None:
            return None
        try:
            positions = self.trader.position
            if code:
                pure_code = self._normalize_code(code)
                for pos in positions:
                    if pos.get('证券代码') == pure_code:
                        return pos
                return None
            return positions
        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
            return None

    def get_today_trades(self):
        if self.dry_run:
            return [t for t in self.trade_log if t['time'].startswith(datetime.now().strftime('%Y-%m-%d'))]
        if self.trader is None:
            return []
        try:
            return self.trader.today_trades
        except Exception as e:
            logger.error(f"查询当日成交失败: {e}")
            return []

    def summary(self):
        print(f"\n{'='*60}")
        print(f"  AutoTrader 状态")
        print(f"{'='*60}")
        print(f"  券商: {self.BROKER_MAP.get(self.broker, self.broker)}")
        print(f"  模式: {'模拟(Dry Run)' if self.dry_run else '实盘(Live)'}")
        print(f"  连接: {'已连接' if self.trader else '未连接'}")
        print(f"  当前持仓: {len(self.positions)}只")
        for code, pos in self.positions.items():
            print(f"    {pos['name']}({code}) 入场:{pos['entry_price']} 数量:{pos['amount']}")
        print(f"  今日交易: {len(self.get_today_trades())}笔")
        print(f"{'='*60}")


if __name__ == '__main__':
    trader = AutoTrader(dry_run=True)
    trader.summary()
