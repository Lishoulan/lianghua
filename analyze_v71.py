
import json

file_path = r"D:\Solo\TradingAgents\paper_trade_results\paper_trade_v71_20260522_224446.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"版本: {data['version']}")
print(f"回测周期: {data['sim_period']}")

print("\n=== 所有平仓交易记录 ===")
all_trades = data.get('all_trades', [])
for i, trade in enumerate(all_trades):
    print(f"{i+1}. {trade['code']} {trade['name']}")
    print(f"   买入日: {trade['entry_date']} 价格: {trade['entry_price']:.2f}")
    print(f"   卖出日: {trade['exit_date']} 价格: {trade['exit_price']:.2f}")
    print(f"   收益: {trade['profit_pct']:.2f}% 持仓: {trade['hold_days']}天 原因: {trade['exit_reason']}")
    print()

print("\n=== 交易统计 ===")
win_trades = [t for t in all_trades if t['profit_pct'] > 0]
loss_trades = [t for t in all_trades if t['profit_pct'] <= 0]
print(f"总交易次数: {len(all_trades)}")
print(f"盈利交易: {len(win_trades)} ({len(win_trades)/len(all_trades)*100:.1f}%)")
print(f"亏损交易: {len(loss_trades)} ({len(loss_trades)/len(all_trades)*100:.1f}%)")

avg_win = sum(t['profit_pct'] for t in win_trades)/len(win_trades) if win_trades else 0
avg_loss = sum(t['profit_pct'] for t in loss_trades)/len(loss_trades) if loss_trades else 0
print(f"平均盈利: {avg_win:.2f}%")
print(f"平均亏损: {avg_loss:.2f}%")

print("\n=== 模拟收益计算 (初始资金50万, 每只最大25%仓位) ===")
initial_cash = 500000
cash = initial_cash
positions = {}
daily_log = data.get('daily_log', [])

print("\n模拟交易过程:")
for day in daily_log:
    date = day['date']
    for sell in day.get('sells', []):
        code = sell['code']
        if code in positions:
            entry_price = positions[code]['entry_price']
            exit_price = sell['price']
            position_value = positions[code]['value']
            profit_pct = sell['profit_pct'] / 100
            profit = position_value * profit_pct
            cash += position_value + profit
            print(f"{date} 卖出 {sell['name']}({code}): 入账 {cash:.0f} 收益 {profit:.0f} ({profit_pct*100:.1f}%)")
            del positions[code]
    
    for buy in day.get('buys', []):
        code = buy['code']
        if code not in positions:
            position_weight = buy.get('mvo_weight', 0.25)
            buy_value = cash * position_weight
            entry_price = buy['price']
            positions[code] = {
                'name': buy['name'],
                'entry_price': entry_price,
                'value': buy_value,
                'entry_date': date
            }
            cash -= buy_value
            print(f"{date} 买入 {buy['name']}({code}): 支出 {buy_value:.0f} 剩余 {cash:.0f}")

print(f"\n=== 模拟最终结果 ===")
print(f"剩余现金: {cash:.0f}")
print(f"持仓市值:")
final_holdings_value = 0
for code, pos in positions.items():
    # 用最后一天的价格来估值
    last_day = daily_log[-1]
    final_price = 0
    if 'positions' in last_day and code in last_day['positions']:
        final_price = last_day['positions'][code]['current_price']
    if final_price > 0:
        profit_pct = (final_price - pos['entry_price']) / pos['entry_price']
        final_value = pos['value'] * (1 + profit_pct)
        final_holdings_value += final_value
        print(f"  {pos['name']}({code}): 当前值 {final_value:.0f} (浮盈 {profit_pct*100:.1f}%)")
    else:
        print(f"  {pos['name']}({code}): 当前值 {pos['value']:.0f} (无最新价格)")

final_total = cash + final_holdings_value
total_return = (final_total - initial_cash) / initial_cash * 100
print(f"总资金: {final_total:.0f}")
print(f"总收益率: {total_return:.2f}%")
print(f"年化收益率: {total_return/0.5:.2f}%")
