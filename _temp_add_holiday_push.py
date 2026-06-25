"""临时脚本：为 daily_push.py 添加节假日提示推送"""
import re

FILE = r'd:\Solo\TradingAgents\classic_ta\daily_push.py'

with open(FILE, 'rb') as f:
    text = f.read().decode('utf-8')

lines = text.split('\n')

# === 定位 L242-244 的交易日检查代码块 ===
# L242:     # 交易日检查
# L243:     if not _is_trading_day():
# L244:         print("📅 今日非交易日，跳过推送", flush=True)
# L245:         return

# 找到交易日检查块
target_idx = None
for i, line in enumerate(lines):
    if 'if not _is_trading_day()' in line:
        target_idx = i
        break

print(f'Found target at L{target_idx+1}: {lines[target_idx]}')
print(f'  L{target_idx+2}: {lines[target_idx+1]}')
print(f'  L{target_idx+3}: {lines[target_idx+2]}')

# 构建新的节假日提示推送代码块
# 替换原有的 3 行（if + print + return）为新的逻辑
new_block = '''    # 交易日检查
    if not _is_trading_day():
        print("📅 今日非交易日，跳过常规推送", flush=True)
        # 节假日提示推送（让用户知道系统正常运行，只是休市）
        try:
            today_cn = datetime.now(_BJT).strftime("%Y-%m-%d")
            weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][datetime.now(_BJT).weekday()]
            holiday_title = f"📅 量化潜伏 {today_cn} | 休市日"
            holiday_desp = f"""### 📅 今日休市提示

- 📅 {today_cn} {weekday_cn}
- 📊 状态: A股休市日（节假日或周末）
- ⏸️ 操作: 今日无信号扫描，无推送内容

### 💡 说明
- 量化系统正常运行，仅因休市跳过扫描
- 下一个交易日将自动恢复推送
- 如需手动触发，可在 GitHub Actions 使用 workflow_dispatch

> ⚡ 量化潜伏系统 V6.4"""
            print(f"  📢 推送节假日提示...", flush=True)
            send_group_push(holiday_title, holiday_desp, holiday_title, holiday_desp)
        except Exception as e:
            print(f"  ⚠️ 节假日提示推送失败（非致命）: {e}", flush=True)
        return
'''

new_block_lines = new_block.rstrip('\n').split('\n')

# 替换原有的 3 行（target_idx, target_idx+1, target_idx+2）
# 原代码：
#   target_idx:     if not _is_trading_day():
#   target_idx+1:         print("📅 今日非交易日，跳过推送", flush=True)
#   target_idx+2:         return
new_lines = lines[:target_idx-1] + new_block_lines + lines[target_idx+3:]

# 写回
new_text = '\n'.join(new_lines)
with open(FILE, 'wb') as f:
    f.write(new_text.encode('utf-8'))

print(f'\n✅ Replaced 3 lines at L{target_idx+1} with {len(new_block_lines)} lines')
print(f'   File now has {len(new_lines)} lines (was {len(lines)})')
print(f'\n=== New block preview ===')
for i, line in enumerate(new_block_lines):
    print(f'  L{target_idx+i}: {line}')
