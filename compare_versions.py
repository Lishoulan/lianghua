
import json

for ver, fname in [
    ("v7.1", r"D:\Solo\TradingAgents\paper_trade_results\paper_trade_v71_20260522_224446.json"),
    ("v7.2", r"D:\Solo\TradingAgents\paper_trade_results\paper_trade_v72_20260523_094824.json"),
]:
    with open(fname, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_trades = data.get('all_trades', [])
    win_trades = [t for t in all_trades if t['profit_pct'] > 0]
    loss_trades = [t for t in all_trades if t['profit_pct'] <= 0]

    initial_cash = 500000
    cash = initial_cash
    positions = {}
    daily_log = data.get('daily_log', [])

    for day in daily_log:
        for sell in day.get('sells', []):
            code = sell['code']
            if code in positions:
                position_value = positions[code]['value']
                profit_pct = sell['profit_pct'] / 100
                profit = position_value * profit_pct
                cash += position_value + profit
                del positions[code]
        for buy in day.get('buys', []):
            code = buy['code']
            if code not in positions:
                position_weight = buy.get('mvo_weight', 0.25)
                buy_value = initial_cash * position_weight
                if buy_value > cash:
                    buy_value = cash * 0.95
                positions[code] = {'name': buy['name'], 'entry_price': buy['price'], 'value': buy_value}
                cash -= buy_value

    final_holdings_value = 0
    last_day = daily_log[-1]
    for code, pos in positions.items():
        if 'positions' in last_day and code in last_day['positions']:
            final_price = last_day['positions'][code]['current_price']
            profit_pct = (final_price - pos['entry_price']) / pos['entry_price']
            final_holdings_value += pos['value'] * (1 + profit_pct)
        else:
            final_holdings_value += pos['value']

    final_total = cash + final_holdings_value
    total_return = (final_total - initial_cash) / initial_cash * 100

    print(f"\n{'='*60}")
    print(f"  {ver} 回测结果")
    print(f"{'='*60}")
    print(f"  总交易: {len(all_trades)}笔  胜率: {len(win_trades)/len(all_trades)*100:.1f}%")
    if win_trades:
        print(f"  平均盈利: {sum(t['profit_pct'] for t in win_trades)/len(win_trades):.2f}%")
    if loss_trades:
        print(f"  平均亏损: {sum(t['profit_pct'] for t in loss_trades)/len(loss_trades):.2f}%")
    print(f"  最大单笔盈利: {max(t['profit_pct'] for t in all_trades):.2f}%")
    print(f"  最大单笔亏损: {min(t['profit_pct'] for t in all_trades):.2f}%")
    print(f"  剩余现金: {cash:,.0f}")
    print(f"  持仓市值: {final_holdings_value:,.0f}")
    print(f"  最终资金: {final_total:,.0f}")
    print(f"  总收益率: {total_return:.2f}%")
    print(f"  年化收益率: {total_return/0.5:.2f}%")

    print(f"\n  逐笔明细:")
    for i, t in enumerate(all_trades):
        sign = "+" if t['profit_pct'] > 0 else ""
        print(f"    {i+1}. {t['name']:6s} {t['entry_date']}→{t['exit_date']}  {sign}{t['profit_pct']:.2f}%  {t['hold_days']}天  {t['exit_reason']}")
