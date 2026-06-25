# 做T双轨策略集成（task-097）

> **任务编号**：task-097
> **任务类型**：功能集成 / 策略扩展
> **关联模块**：`classic_ta/common/{t_trading, scanner, message_builder}.py`
> **参考文档**：[CODE_WIKI_NEW.md](file:///d:/Solo/TradingAgents/CODE_WIKI_NEW.md) §4.2.1 / §4.2.7
> **最后更新**：2026-06-19

---

## 0. 任务概览（结构精简版）

### 0.1 目标

将"知行合一双轨趋势系统"集成到每日推送，对每个潜伏信号自动附加做T建议（正T / 倒T / 观望）。复用现有 `white_line` / `yellow_line` / `J` / `atr14` 指标，零新增数据源依赖。

### 0.2 实施状态盘点（2026-06-19）

| Task | 文件 | 状态 | 实际行号 | 备注 |
|------|------|------|---------|------|
| Task 1 | `classic_ta/common/t_trading.py` | ✅ 已完成 | 251 行 | 5 个函数 + 11 个参数常量 |
| Task 2 | `classic_ta/common/scanner.py` | ✅ 已完成 | L108 / L131 / L170 | import + 调用 + 字段 |
| Task 3 | `classic_ta/common/message_builder.py` | ❌ 未完成 | `_append_signal_detail` 在 L325 | 缺做T展示逻辑 |
| Task 4 | 语法验证 | ❌ 未执行 | — | 待 Task 3 完成后执行 |

> **关键差异**：原任务文档引用的行号（scanner L129/L168、message_builder L422/L258）与当前代码不符。本优化版已按实际代码校准。

### 0.3 数据契约

**输入**（来自 `scanner._extract_signal_info`）：

```python
df: pd.DataFrame  # 已计算 IndicatorCalcBase + add_micro_confirm_indicators + Detect_AmbushSignal_V64
# 必需列: Close, High, Low, Volume, white_line, yellow_line, J, volume_ma
signal_idx: int   # 通常 = len(df) - 1
```

**输出**（`analyze_t_trading` 返回结构）：

```python
{
    "mode": "正T" | "倒T" | "观望",      # 主模式
    "yellow_slope": "向上" | "向下" | "走平",  # 黄线5日斜率方向
    "buy_signal": str | None,             # 买入建议文本
    "sell_signal": str | None,            # 卖出建议文本
    "risk_alert": str | None,             # 风控提示文本
    "amplitude": float,                   # 日内振幅%
}
```

### 0.4 调用链路

```
daily_push.py
  └─ SyncScanner.scan()
       └─ _fetch_and_process_one_core()           # 数据获取 + 指标计算
            └─ _extract_signal_info()              # scanner.py L94
                 ├─ analyze_signal_detail()        # 威科夫/VPA/蜡烛图解读
                 └─ analyze_t_trading(df, len-1)   # scanner.py L131 ← 做T分析
            └─ signal_info["t_trading"] = t_info   # scanner.py L170
  └─ message_builder.build_push_message()
       └─ _append_signal_detail()                  # message_builder.py L325
            └─ ❌ 缺失: t_trading 展示逻辑          # Task 3 待补
```

---

## 1. 运维导向版

### 1.1 部署影响评估

| 维度 | 影响 | 说明 |
|------|------|------|
| 数据源 | 无 | 复用现有 `white_line` / `yellow_line` / `J` / `atr14` |
| 缓存 | 无 | 不新增缓存表，不修改 DuckDB Schema |
| 性能 | 极低 | 单股 O(1) 计算，全市场扫描增量 < 1s |
| 推送消息 | 中 | 管理员版每信号 +3~5 行，内测版每信号 +1 行 |
| 配置 | 无 | 参数硬编码在 `t_trading.py` 模块顶部常量 |
| 兼容性 | 向后兼容 | `t_info` 字段缺失时 message_builder 应优雅降级 |

### 1.2 实施步骤（按 Task 顺序）

#### Step 1: 验证 t_trading.py（已完成，仅验证）

```bash
# 验证模块可导入
python -c "from classic_ta.common.t_trading import analyze_t_trading; print('OK')"

# 验证函数签名
python -c "
from classic_ta.common.t_trading import analyze_t_trading
import inspect
print(inspect.signature(analyze_t_trading))
# 期望: (df: pandas.DataFrame, signal_idx: int) -> Dict
"
```

#### Step 2: 验证 scanner.py 集成（已完成，仅验证）

```bash
# 验证 import 和调用存在
python -c "
import ast
with open('classic_ta/common/scanner.py','rb') as f:
    tree = ast.parse(f.read())
src = open('classic_ta/common/scanner.py','rb').read().decode('utf-8')
assert 'from classic_ta.common.t_trading import analyze_t_trading' in src, 'import 缺失'
assert 'analyze_t_trading(df, len(df) - 1)' in src, '调用缺失'
assert '\"t_trading\": t_info' in src, '字段缺失'
print('scanner.py 集成验证通过')
"
```

#### Step 3: 实施 message_builder.py 集成（待执行）

**目标位置**：`_append_signal_detail()` 函数内，支撑阻力展示之后、交易参考之前。

**管理员版（完整展示）**：

```python
# 在 _append_signal_detail() 中，支撑阻力之后追加：
t_info = s.get("t_trading")
if t_info and t_info.get("mode") != "观望":
    mode_icon = {"正T": "📈", "倒T": "📉"}.get(t_info["mode"], "➡️")
    lines.append(f"")
    lines.append(f"🔄 做T建议 [{mode_icon} {t_info['mode']}]")
    lines.append(f"   黄线斜率: {t_info['yellow_slope']} | 振幅: {t_info['amplitude']:.1f}%")
    if t_info.get("buy_signal"):
        lines.append(f"   买入: {t_info['buy_signal']}")
    if t_info.get("sell_signal"):
        lines.append(f"   卖出: {t_info['sell_signal']}")
    if t_info.get("risk_alert"):
        lines.append(f"   ⚠️ 风控: {t_info['risk_alert']}")
elif t_info and t_info.get("mode") == "观望":
    lines.append(f"")
    lines.append(f"🔄 做T: 观望（{t_info.get('risk_alert','振幅不足')}）")
```

**内测精简版（单行）**：

```python
# 在 build_beta_push_message() 对应信号循环中追加一行：
t_info = s.get("t_trading")
if t_info and t_info.get("mode") != "观望":
    parts = [f"🔄 {t_info['mode']}"]
    if t_info.get("buy_signal"):
        parts.append(f"买:{t_info['buy_signal'].split('|')[0].strip()}")
    if t_info.get("sell_signal"):
        parts.append(f"卖:{t_info['sell_signal'].split('|')[0].strip()}")
    lines.append(f"   {' | '.join(parts)}")
```

#### Step 4: 语法验证 + Dry-run

```bash
# 1. AST 语法检查（全部修改文件）
python -c "
import ast
for f in ['classic_ta/common/t_trading.py','classic_ta/common/scanner.py','classic_ta/common/message_builder.py']:
    with open(f,'rb') as fp:
        ast.parse(fp.read())
    print(f'{f}: AST OK')
"

# 2. 导入冒烟测试
python -c "
from classic_ta.common.t_trading import analyze_t_trading
from classic_ta.common.scanner import SyncScanner, _extract_signal_info
from classic_ta.common.message_builder import build_push_message, _append_signal_detail
print('All imports OK')
"

# 3. Dry-run 验证做T建议出现在推送消息中
python classic_ta/daily_push.py --dry-run 2>&1 | grep -A 5 "做T建议"
```

### 1.3 回滚方案

| 场景 | 回滚动作 |
|------|---------|
| Task 3 引发推送异常 | 注释 `_append_signal_detail()` 中做T展示代码块 |
| t_trading 计算异常 | scanner.py L131 改为 `t_info = None`，L170 保留字段但值为 None |
| 完整回滚 | `git revert` 对应 commit；t_trading.py 可保留不删除（无副作用） |

### 1.4 监控指标

实施后关注以下指标（来自 [CODE_WIKI_NEW.md](file:///d:/Solo/TradingAgents/CODE_WIKI_NEW.md) §4.4.4）：

- `scan.duration_seconds`：扫描耗时（预期增量 < 5%）
- `push.wechat.success`：推送成功率（应保持 ≥95%）
- 推送消息长度：管理员版单信号段长度（预期 +150 字符以内）

---

## 2. 技术深度版

### 2.1 t_trading.py 实现详解

#### 2.1.1 模块常量

```python
SLOPE_LOOKBACK = 5          # 黄线斜率回看天数
SLOPE_UP_THRESHOLD = 0.3    # 黄线斜率 >0.3% 判定向上
SLOPE_DOWN_THRESHOLD = -0.3 # 黄线斜率 <-0.3% 判定向下
MIN_AMPLITUDE_PCT = 1.5     # 最小振幅%，低于此观望
TOUCH_TOLERANCE_PCT = 0.5   # 触碰黄线容差%
J_OVERSOLD = 20             # KDJ J值超卖阈值（正T超跌买）
J_OVERBOUGHT = 80           # KDJ J值超买阈值（正T卖）
J_EXTREME_OVERBOUGHT = 100  # KDJ J值极度超买（倒T卖）
STOP_LOSS_PCT = 1.5         # 纠错止损%
SHORT_VOLUME_RATIO = 0.8    # 缩量判定（量比<0.8为缩量）
```

#### 2.1.2 函数签名与职责

| 函数 | 签名 | 职责 |
|------|------|------|
| `analyze_t_trading` | `(df: pd.DataFrame, signal_idx: int) -> Dict` | 主入口，判定模式并分发 |
| `_analyze_long_t` | `(close, white, yellow, j_val, amplitude, ...) -> Dict` | 正T模式（黄线向上，先买后卖） |
| `_analyze_short_t` | `(close, white, yellow, j_val, amplitude, ...) -> Dict` | 倒T模式（黄线向下/走平，先卖后买） |
| `_get_slope_label` | `(df: pd.DataFrame, signal_idx: int) -> str` | 黄线斜率方向标签 |
| `_default_result` | `() -> Dict` | 数据不足时的默认返回 |

#### 2.1.3 核心判定逻辑

```
┌─────────────────────────────────────────────────────────────┐
│  analyze_t_trading(df, signal_idx)                          │
├─────────────────────────────────────────────────────────────┤
│  1. 边界检查: signal_idx < 5 或越界 → _default_result()      │
│  2. 提取: close, high, low, white, yellow, J, volume, vol_ma │
│  3. 振幅过滤: (high-low)/close*100 < 1.5% → 观望             │
│  4. 黄线斜率: (yellow[now] - yellow[-5]) / yellow[-5] * 100  │
│     ├─ > 0.3%  → "向上" → _analyze_long_t()                 │
│     ├─ < -0.3% → "向下" → _analyze_short_t()                │
│     └─ 其他    → "走平" → _analyze_short_t()                │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1.4 正T信号规则（`_analyze_long_t`）

| 信号类型 | 触发条件 | 文本模板 |
|---------|---------|---------|
| 买入1 | 触碰黄线（±0.5%）且未跌破 | `回踩黄线{yellow}(偏离{pct}%)` |
| 买入2 | 跌破双线 且 J<20 | `超跌买(J={j}<{J_OVERSOLD},跌破双线)` |
| 卖出1 | 白线上方 且 J>80 | `白线{white}上方+J={j}>{J_OVERBOUGHT}` |
| 卖出2 | 正乖离>3% | `正乖离过大(偏离白线{pct}%)` |
| 风控 | 始终提示 | `跌破黄线1.5%({stop_price})严格短线止损` |

#### 2.1.5 倒T信号规则（`_analyze_short_t`）

| 信号类型 | 触发条件 | 文本模板 |
|---------|---------|---------|
| 卖出1 | 触黄线（下方反弹） | `触黄线阻力{yellow}(偏离{pct}%)` |
| 卖出2 | 跌破白线 | `跌破白线{white}` |
| 卖出3 | J>100 且 白线上方 | `超买卖(J={j}>{J_EXTREME_OVERBOUGHT})` |
| 卖出4 | 正乖离>3% 且 白线上方 | `正乖离过大(偏离白线{pct}%)` |
| 买回1 | 回落至白线下方 | `回落至白线{white}下方` |
| 买回2 | 黄线附近缩量止跌 | `黄线{yellow}附近缩量止跌(量比{vol_ratio})` |
| 风控 | 始终提示 | `突破{sell_ref}的1.5%({buyback_price})立即买回防卖飞` |

### 2.2 scanner.py 集成点详解

**位置**：`_extract_signal_info()` 函数（[scanner.py](file:///d:/Solo/TradingAgents/classic_ta/common/scanner.py#L94-L172)）

```python
# L108: 导入（函数内延迟导入，避免循环依赖）
from classic_ta.common.t_trading import analyze_t_trading

# L131: 调用（与 analyze_signal_detail 并列，使用同一 df 和 signal_idx）
detail = analyze_signal_detail(df, len(df) - 1, best_params)
t_info = analyze_t_trading(df, len(df) - 1)

# L170: 字段追加（signal_info dict 末尾）
"t_trading": t_info,
```

**设计要点**：
- 延迟导入（函数内 import）符合项目惯例，避免模块加载顺序问题
- `t_info` 与 `detail` 并列存储，message_builder 可独立访问
- `t_info` 为 None 时（数据不足），下游应优雅降级

### 2.3 message_builder.py 集成点详解

**目标函数**：`_append_signal_detail(lines, s, index, icon, is_priority=True)`

**当前函数结构**（[message_builder.py](file:///d:/Solo/TradingAgents/classic_ta/common/message_builder.py#L325)）：

```
L325: def _append_signal_detail(lines, s, index, icon, is_priority=True):
L327-330: 评分提取
L331-339: 优先挡标签
L341-349: 评分星级
L351-355: 信号头部
L357-376: 评分拆解
L378-382: 价格涨跌
L384-??? : 均线结构
...      : SOS锚定 / 威科夫 / VPA / 蜡烛图 / 支撑阻力
...      : 交易参考（买入价/止损/吊灯线/仓位）
```

**插入位置**：支撑阻力展示之后、交易参考之前（约 L400 附近，需运行时确认）。

**降级策略**：

```python
t_info = s.get("t_trading")
if not t_info:
    return  # 老数据兼容，t_trading 字段可能不存在

if t_info.get("mode") == "观望":
    # 观望模式仅显示一行提示
    ...
else:
    # 正T/倒T 显示完整建议
    ...
```

### 2.4 与现有指标体系的依赖关系

```
v60_ambush_model.IndicatorCalcBase(df)
  ├─ white_line = Close.ewm(span=10).ewm(span=10)   ← t_trading 使用
  ├─ yellow_line = (ma14+ma28+ma57+ma114)/4          ← t_trading 使用
  ├─ J = 3K - 2D (clip 0~100)                        ← t_trading 使用
  ├─ atr14                                           ← (未使用)
  └─ volume_ma                                       ← t_trading 使用（量比计算）

v64_ambush_model.Detect_AmbushSignal_V64(df, params)
  └─ ambush_signal                                   ← scanner 据此筛选信号日
```

**关键约束**：
- `t_trading` 必须在 `IndicatorCalcBase` 之后调用（依赖 white_line/yellow_line/J/volume_ma）
- `t_trading` 不依赖 `add_micro_confirm_indicators` 或 `add_entry_quality_indicators`
- `signal_idx` 必须是 `ambush_signal==True` 的交易日（由 scanner 保证）

### 2.5 边界情况处理

| 场景 | 处理 | 代码位置 |
|------|------|---------|
| `signal_idx < 5` | 返回 `_default_result()` | `analyze_t_trading` 开头 |
| `len(df) <= signal_idx` | 返回 `_default_result()` | `analyze_t_trading` 开头 |
| `yellow_line` 或 `white_line` 为 0/NaN | 返回 `_default_result()` | `analyze_t_trading` 中段 |
| `prev_yellow <= 0`（5日前黄线异常） | `slope_label = "走平"` | `_get_slope_label` |
| `volume_ma <= 0` | `vol_ratio = 1.0`（视为正常量） | `analyze_t_trading` |
| `t_info` 为 None（scanner 异常） | message_builder 跳过展示 | Task 3 降级逻辑 |

---

## 3. 任务清单（按当前状态校准）

### Task 1: t_trading.py — ✅ 已完成

- **文件**：`classic_ta/common/t_trading.py`（251 行）
- **状态**：已实现 `analyze_t_trading` + 4 个辅助函数 + 11 个参数常量
- **验证**：`python -c "from classic_ta.common.t_trading import analyze_t_trading; print('OK')"`

### Task 2: scanner.py 集成 — ✅ 已完成

- **文件**：`classic_ta/common/scanner.py`
- **修改点**：
  - L108: `from classic_ta.common.t_trading import analyze_t_trading`
  - L131: `t_info = analyze_t_trading(df, len(df) - 1)`
  - L170: `"t_trading": t_info,`
- **验证**：AST 检查 + 导入测试

### Task 3: message_builder.py 集成 — ❌ 待执行

- **文件**：`classic_ta/common/message_builder.py`
- **目标函数**：`_append_signal_detail()`（L325）
- **插入位置**：支撑阻力之后、交易参考之前
- **管理员版**：追加 5 行做T建议展示（模式/买入/卖出/风控）
- **内测版**：追加 1 行精简做T提示
- **降级**：`t_info` 为 None 或 mode="观望" 时优雅跳过

### Task 4: 语法验证 + Dry-run — ❌ 待执行

```bash
# 1. AST 语法检查
python -c "
import ast
for f in ['classic_ta/common/t_trading.py','classic_ta/common/scanner.py','classic_ta/common/message_builder.py']:
    with open(f,'rb') as fp:
        ast.parse(fp.read())
    print(f'{f}: OK')
"

# 2. 导入冒烟测试
python -c "
from classic_ta.common.t_trading import analyze_t_trading
from classic_ta.common.scanner import SyncScanner, _extract_signal_info
from classic_ta.common.message_builder import build_push_message, _append_signal_detail
print('imports OK')
"

# 3. Dry-run 验证
python classic_ta/daily_push.py --dry-run
```

---

## 4. 修改文件清单

| 文件 | 状态 | 操作 |
|------|------|------|
| `classic_ta/common/t_trading.py` | ✅ 已存在 | 无需操作（仅验证） |
| `classic_ta/common/scanner.py` | ✅ 已集成 | 无需操作（仅验证） |
| `classic_ta/common/message_builder.py` | ❌ 待修改 | 追加做T展示逻辑 |

---

## 5. 验收标准

1. **功能验收**：
   - `python classic_ta/daily_push.py --dry-run` 输出包含 `🔄 做T建议` 字样
   - 正T信号显示"买入/卖出/风控"三段
   - 倒T信号显示"卖出/买回/风控"三段
   - 观望信号显示单行提示
   - 无 t_trading 字段的老信号不报错

2. **性能验收**：
   - 全市场扫描耗时增量 < 5%
   - 推送消息长度增量 < 10%

3. **兼容性验收**：
   - `t_info = None` 时不抛异常
   - `t_info["mode"] = "观望"` 时仅显示单行
   - 现有信号展示逻辑不受影响

---

## 6. 风险与注意事项

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| 行号漂移 | 低 | 本文档行号基于 2026-06-19 代码快照，修改前应重新定位 |
| 推送消息超长 | 中 | `_truncate_content()` 已有 60000 字符限制（[push_channels.py](file:///d:/Solo/TradingAgents/classic_ta/common/push_channels.py)） |
| 做T建议误导用户 | 中 | 推送末尾保留免责声明；风控提示强制显示 |
| 参数硬编码 | 低 | 11 个常量集中在 `t_trading.py` 顶部，未来可迁移至 `V64_PARAMS` |

---

**文档版本**：v2.0（基于 CODE_WIKI_NEW.md 重构）
**最后更新**：2026-06-19
