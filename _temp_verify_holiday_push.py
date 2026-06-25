"""验证节假日提示推送功能"""
import ast

# 1. AST 语法检查
with open(r'd:\Solo\TradingAgents\classic_ta\daily_push.py', 'rb') as f:
    ast.parse(f.read())
print('AST OK: daily_push.py')

# 2. 导入测试
from classic_ta.daily_push import daily_push, _is_trading_day
print('Import OK: daily_push, _is_trading_day')

# 3. 交易日检查
is_trading = _is_trading_day()
print(f'_is_trading_day() = {is_trading}')
if is_trading:
    print('  -> 今天是交易日（不会触发节假日提示）')
else:
    print('  -> 今天是休市日（将触发节假日提示推送）')

# 4. 模拟节假日提示推送（mock send_group_push）
print()
print('=== 模拟节假日提示推送 ===')
from classic_ta.common import push_channels as _pc

original_send = _pc.send_group_push

def mock_send_group_push(admin_title, admin_desp, beta_title, beta_desp, scheduled=None):
    print(f'[MOCK] 管理员组标题: {admin_title}')
    print(f'[MOCK] 内测组标题: {beta_title}')
    print(f'[MOCK] 内容长度: {len(admin_desp)} 字符')
    print()
    print('=' * 60)
    print(admin_desp)
    print('=' * 60)
    return {'admin': True, 'beta': True}

_pc.send_group_push = mock_send_group_push

# 5. 执行 daily_push（今天休市，应触发节假日提示）
try:
    daily_push()
except Exception as e:
    print(f'EXCEPTION: {type(e).__name__}: {e}')
finally:
    _pc.send_group_push = original_send

print()
print('=== 验证完成 ===')
