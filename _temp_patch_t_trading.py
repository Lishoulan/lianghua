"""临时脚本：为 message_builder.py 追加做T建议展示逻辑"""
import sys

FILE = r'd:\Solo\TradingAgents\classic_ta\common\message_builder.py'

with open(FILE, 'rb') as f:
    text = f.read().decode('utf-8')

lines = text.split('\n')

# === Step 1: Admin version - insert after support/resistance, before trade reference ===
support_line_idx = None
trade_ref_idx = None
for i, line in enumerate(lines):
    if '支撑:' in line and '阻力:' in line and support_line_idx is None:
        support_line_idx = i
    if '交易参考' in line and trade_ref_idx is None:
        trade_ref_idx = i

print(f'Support/resistance: L{support_line_idx+1}')
print(f'  {lines[support_line_idx]}')
print(f'Trade reference: L{trade_ref_idx+1}')
print(f'  {lines[trade_ref_idx]}')

# Insert position: right after support/resistance line
insert_idx = support_line_idx + 1

# Build the t_trading display block (admin version)
t_trading_block = '''    # 做T建议（知行合一双轨趋势系统）
    t_info = s.get("t_trading")
    if t_info:
        t_mode = t_info.get("mode", "观望")
        if t_mode == "观望":
            lines.append(f"")
            lines.append(f"🔄 做T: 观望（{t_info.get('risk_alert', '振幅不足') or '振幅不足'}）")
        else:
            mode_icon = "📈" if t_mode == "正T" else "📉"
            slope = t_info.get("yellow_slope", "走平")
            amp = t_info.get("amplitude", 0)
            lines.append(f"")
            lines.append(f"🔄 做T建议 [{mode_icon} {t_mode}]")
            lines.append(f"   黄线斜率: {slope} | 振幅: {amp:.1f}%")
            buy_sig = t_info.get("buy_signal")
            if buy_sig:
                lines.append(f"   买入: {buy_sig}")
            sell_sig = t_info.get("sell_signal")
            if sell_sig:
                lines.append(f"   卖出: {sell_sig}")
            risk_alert = t_info.get("risk_alert")
            if risk_alert:
                lines.append(f"   ⚠️ 风控: {risk_alert}")
'''

t_trading_lines = t_trading_block.rstrip('\n').split('\n')

# Insert admin version
new_lines = lines[:insert_idx] + t_trading_lines + lines[insert_idx:]

# === Step 2: Beta version - add t_trading line after "止损" line in priority and normal signals ===
# Find the priority signals "止损" line and normal signals "止损" line
# Priority: lines.append(f"   止损:{s['hard_stop']:.2f} | 持仓10天")
# Normal: lines.append(f"   止损:{s['hard_stop']:.2f} | 持仓10天")
# We need to insert after each of these (but only once per block)

# Re-find positions in new_lines since indices shifted
priority_stop_loss_idx = None
normal_stop_loss_idx = None

# Track which block we're in
in_priority_block = False
in_normal_block = False

for i, line in enumerate(new_lines):
    stripped = line.strip()
    if '重点推荐' in line:
        in_priority_block = True
        in_normal_block = False
    elif '关注标的' in line:
        in_priority_block = False
        in_normal_block = True
    elif '仓位管理' in line:
        in_priority_block = False
        in_normal_block = False

    if '止损:' in line and '持仓10天' in line:
        if in_priority_block and priority_stop_loss_idx is None:
            priority_stop_loss_idx = i
        elif in_normal_block and normal_stop_loss_idx is None:
            normal_stop_loss_idx = i

print(f'\nPriority stop_loss: L{priority_stop_loss_idx+1}')
print(f'  {new_lines[priority_stop_loss_idx]}')
print(f'Normal stop_loss: L{normal_stop_loss_idx+1}')
print(f'  {new_lines[normal_stop_loss_idx]}')

# Beta version t_trading line (insert after stop_loss line, before the empty line)
beta_t_trading_line = '                lines.append(f"   🔄 做T: {t_mode} | 买:{buy_short} | 卖:{sell_short}")'

# We need to insert a conditional block. Let's build it properly.
# The beta version uses 16-space indent (inside for loop inside if block)
beta_block = '''                # 做T精简提示
                t_info = s.get("t_trading")
                if t_info and t_info.get("mode") != "观望":
                    t_mode = t_info["mode"]
                    buy_short = (t_info.get("buy_signal") or "—").split("|")[0].strip()[:20]
                    sell_short = (t_info.get("sell_signal") or "—").split("|")[0].strip()[:20]
                    lines.append(f"   🔄 做T: {t_mode} | 买:{buy_short} | 卖:{sell_short}")
'''

beta_lines = beta_block.rstrip('\n').split('\n')

# Insert into normal block first (higher index) to preserve priority block index
# Normal block insertion
new_lines = new_lines[:normal_stop_loss_idx+1] + beta_lines + new_lines[normal_stop_loss_idx+1:]

# Priority block insertion (index unchanged since normal is after priority)
new_lines = new_lines[:priority_stop_loss_idx+1] + beta_lines + new_lines[priority_stop_loss_idx+1:]

# Write back
new_text = '\n'.join(new_lines)
with open(FILE, 'wb') as f:
    f.write(new_text.encode('utf-8'))

print(f'\n✅ Inserted admin t_trading block ({len(t_trading_lines)} lines) at L{insert_idx+1}')
print(f'✅ Inserted beta t_trading block ({len(beta_lines)} lines) at L{priority_stop_loss_idx+2} (priority)')
print(f'✅ Inserted beta t_trading block ({len(beta_lines)} lines) at L{normal_stop_loss_idx+2+len(beta_lines)} (normal)')
print(f'   File now has {len(new_lines)} lines (was {len(lines)})')
