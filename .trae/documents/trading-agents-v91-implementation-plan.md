# TradingAgents v9.1 优化实施计划（续）

> P0 已完成（.env绝对路径加载 + LLM API健壮性 + live_trader.py同步）
> 本文档覆盖 P1-P6 的详细实施步骤

---

## 当前状态确认

- ✅ P0-1: `daily_scan_push.py` 的 `load_dotenv` 已改为绝对路径 + override
- ✅ P0-2: `llm_analyzer.py` 已增强（property延迟读取 + .env兜底 + 3次重试 + 60秒超时 + 事件分析prompt）
- ✅ P0-3: `live_trader.py` 的 `load_dotenv` 已同步修复
- ✅ `holdings.json` 已不存在（无需删除）
- ⬜ `ml_strategy/archive/` 目录尚未创建
- ⬜ `live_trader.py` 尚未集成 LLM 分析
- ⬜ `check_sell_conditions()` 中 MIN_HOLD_DAYS 阻断风控卖出（daily_scan_push.py:L408-409）
- ⬜ `live_trader.py` 中同样存在 MIN_HOLD_DAYS 阻断风控卖出（L546-548）
- ⬜ `panic_breaker.is_panic()` 返回 bool，需改为3级
- ⬜ `cost_aware_optimizer` 无行业集中度约束
- ⬜ `daily_scan_push.py` 无 `--mode` 参数

---

## P1：代码清理与架构统一

### P1-1: 归档旧版模块

**目标**：将不再使用的模块移入 `ml_strategy/archive/`，保持核心层纯净

**步骤**：
1. 创建 `D:\Solo\TradingAgents\ml_strategy\archive\` 目录
2. 在 `archive/` 下创建 `__init__.py`（空文件）
3. 移动以下7个文件到 `archive/`：
   - `pipeline.py` — 旧版推理管线，已被 daily_scan_push.py 的8步流水线替代
   - `portfolio_optimizer.py` — 旧版组合优化，已被 CostAwarePortfolioOptimizer 替代
   - `kan_predictor.py` — 旧版KAN预测器，已被 ChebyKANTrainer 替代
   - `lgb_predictor.py` — LightGBM，未使用
   - `isotonic_calibrator.py` — v10.0备用，当前未使用
   - `dynamic_spread.py` — v10.0备用，当前未使用
   - `fractional_kelly.py` — v10.0备用，当前未使用
4. 全局搜索确认无其他文件 import 这些模块（如有则更新 import 路径）

**验证**：运行 `python daily_scan_push.py` 确认无 ImportError

### P1-2: 同步 live_trader.py 的 LLM 分析

**目标**：让 `live_trader.py` 也具备 DeepSeek 中长期分析能力

**步骤**：
1. 在 `live_trader.py` 顶部添加 `from ml_strategy.llm_analyzer import LLMStockAnalyzer`
2. 在 `run_live_trader()` 的 Step 4（交易信号汇总）之后，新增 Step 4.5：LLM分析
   - 对当前持仓股 + 买入信号股调用 `llm_analyzer.analyze_stock()`
   - 将分析结果附加到推送消息中
3. 在半自动模式的推送消息中增加 AI 分析板块
4. 注意：`live_trader.py` 使用 `LivePortfolio`，而 `daily_scan_push.py` 使用 `ScanPortfolio`，两者共享 `portfolio.json`，推送逻辑需适配

**验证**：手动运行 `live_trader.py`，确认 LLM 分析结果出现在推送中

---

## P2：修改"最小持仓3天"逻辑 — 风控止损优先

**目标**：风控类卖出（熔断/0AMV翻空）不受 MIN_HOLD_DAYS 限制，T+1即可卖出；止盈/信号类卖出仍需满足3天

### 修改文件1: `daily_scan_push.py` — `check_sell_conditions()` 函数

**当前逻辑**（L408-409）：
```python
if hold_days < MIN_HOLD_DAYS:
    return None, profit_pct, dd_pct
```
这会导致所有卖出（包括风控类）被3天规则阻断。

**修改为**：
```python
# 风控类卖出：不受持仓天数限制
if is_panic:
    sell_reason = '熔断器触发'
elif not oamv_daily:
    sell_reason = '0AMV日线BEAR'

# 止盈/信号类卖出：需满足最小持仓天数
if sell_reason is None and hold_days >= MIN_HOLD_DAYS:
    yellow_line = row.get('yellow_line')
    if yellow_line is not None and not pd.isna(yellow_line):
        if current_price < yellow_line:
            sell_reason = f'收盘<{yellow_line:.2f}(黄线)'

    if sell_reason is None and dd_pct >= TRAILING_STOP_PCT:
        sell_reason = f'峰值回撤{dd_pct:.1f}%≥{TRAILING_STOP_PCT}%'

    if sell_reason is None:
        entry_atr = pos.get('entry_atr', 0)
        if entry_atr > 0 and peak_price > entry_price:
            dd_atr = (peak_price - current_price) / entry_atr
            if dd_atr >= ATR_STOP_MULT:
                sell_reason = f'ATR止损{dd_atr:.1f}x≥{ATR_STOP_MULT}x'
```

### 修改文件2: `live_trader.py` — Step 2 持仓风控检查

**当前逻辑**（L546-548）：
```python
if hold_days < MIN_HOLD_DAYS:
    print(f"  {ts_code} {pos['name']}: 持仓{hold_days}天 < {MIN_HOLD_DAYS}天最小持仓，继续持有")
    continue
```

**修改为**：同样的分层逻辑——先判断风控类（不受天数限制），再判断止盈/信号类（需满足3天）

**验证**：
- 确认 T+1 日熔断触发时可以正常卖出
- 确认 T+1 日止盈类卖出仍被3天规则限制
- 确认 T+2 日 0AMV翻空时可以正常卖出

---

## P3：尾盘买入策略 — 信号时机优化

**目标**：将买入信号判定时间移动到尾盘14:45，彻底发挥A股T+1交易优势

### 步骤

1. **修改 `daily_scan_push.py` 增加 `--mode` 参数**
   - 在 `run_daily_scan()` 入口处解析命令行参数
   - 三种模式：
     - `morning`（早盘08:00）：只推送大势判断 + 持仓风控检查，**不执行买入**
     - `afternoon`（尾盘14:45）：完整8步流水线（卖出+买入+LLM+推送）
     - `evening`（盘后18:30）：推送当日交易总结 + 更新持仓状态（不执行买卖）
   - 默认模式为 `afternoon`（向后兼容）

2. **morning 模式逻辑**
   - 执行 Step 1（持仓风控检查）和 Step 2（执行卖出）——卖出不受模式限制
   - 跳过 Step 3（扫描买入）和 Step 4（执行买入）
   - 推送消息标题加 `[早盘]` 前缀，内容只含大势+持仓风控
   - 如果有风控卖出，仍然执行并推送

3. **afternoon 模式逻辑**
   - 完整8步流水线（与当前逻辑一致）
   - 推送消息标题加 `[尾盘]` 前缀

4. **evening 模式逻辑**
   - 只读取当前持仓状态，计算组合概况
   - 调用 LLM 分析持仓股
   - 推送消息标题加 `[盘后]` 前缀，内容为当日总结 + AI分析

5. **更新定时任务**
   - 08:00 → `python daily_scan_push.py --mode morning`
   - 14:45 → `python daily_scan_push.py --mode afternoon`
   - 18:30 → `python daily_scan_push.py --mode evening`

**验证**：
- 分别用三种模式运行，确认行为正确
- 确认 morning 模式不执行买入
- 确认 afternoon 模式完整执行

---

## P4：MVO 行业集中度约束

**目标**：在 `CostAwarePortfolioOptimizer` 的凸优化约束中增加行业集中度上限，同行业权重之和 ≤ 40%

### 步骤

1. **修改 `cost_aware_optimizer.py`**
   - 在 `_cost_aware_optimize()` 方法中增加 `industry_map` 参数：`Dict[str, str]`（ts_code → industry_name）
   - 在约束条件中增加行业集中度约束：
     ```python
     # 行业集中度约束：同行业权重之和 <= 40%
     if industry_map:
         industry_groups = {}
         for i, code in enumerate(candidate_codes):
             ind = industry_map.get(code, 'unknown')
             if ind not in industry_groups:
                 industry_groups[ind] = []
             industry_groups[ind].append(i)

         for ind_name, indices in industry_groups.items():
             if len(indices) > 1:
                 constraints.append(
                     cp.sum(w[indices]) <= 0.4
                 )
     ```
   - 在 `optimize()` 方法中透传 `industry_map` 参数

2. **修改 `daily_scan_push.py` 调用处**
   - 在调用 `portfolio_optimizer.optimize()` 时，从 `all_stock_data` 中提取行业信息
   - 构建 `industry_map = {code: all_stock_data[code].get('industry', '') for code in candidate_codes}`
   - 传入 `industry_map` 参数

3. **修改 `live_trader.py` 调用处**
   - 同样传入 `industry_map`

**验证**：
- 确认同行业两只股票（如广州发展+中山公用）的权重之和不超过40%
- 确认无行业冲突时权重分配正常

---

## P5：Panic Breaker 增加领先指标

**目标**：将 `is_panic()` 从 bool 改为3级返回值（normal/warning/panic），增加跌停增速和市场宽度恶化速率

### 步骤

1. **修改 `panic_breaker.py`**

   a. **增加领先指标计算**（在 `compute_market_breadth` 中）：
   - 跌停家数增速：`limit_down_acceleration`
     - 逻辑：如果前日跌停 N 家，当日跌停 > N×3，标记为 `early_warning`
   - 市场宽度恶化速率：`breadth_deterioration_rate`
     - 逻辑：5日内 breadth 从 <50% 飙升到 >70%，标记为加速恶化
   - 存储为 `self.warning_state`：`pd.Series`，值为 `'normal'`/`'warning'`/`'panic'`

   b. **修改 `is_panic()` → `get_market_state()`**：
   - 返回值从 `bool` 改为 `str`：`'normal'` / `'warning'` / `'panic'`
   - `panic`：原条件（85%跌破MA20 或 跌停>150）
   - `warning`：跌停3倍增速 或 breadth 5日恶化超20个百分点
   - `normal`：其他

   c. **保留 `is_panic()` 兼容方法**：
   - `is_panic(date)` → `return self.get_market_state(date) == 'panic'`
   - 避免破坏现有代码

2. **修改 `daily_scan_push.py`**
   - 将 `is_panic = panic_breaker.is_panic(last_date)` 改为 `market_state = panic_breaker.get_market_state(last_date)`
   - 适配3级状态：
     - `panic`：强制清仓（与当前逻辑一致）
     - `warning`：禁止新买入，但不清仓；推送消息中标注⚠️预警
     - `normal`：正常交易
   - 在推送消息的"大势判断"板块中增加市场状态显示

3. **修改 `live_trader.py`**
   - 同样适配3级市场状态

4. **修改 `check_sell_conditions()`**
   - `is_panic` 参数改为 `market_state`
   - 风控类卖出条件：`market_state == 'panic'` → 熔断清仓
   - `warning` 状态下不禁售（只是不买），但可以正常止盈/止损

**验证**：
- 确认 `is_panic()` 兼容方法仍返回 bool
- 确认 `get_market_state()` 返回正确的3级状态
- 确认 warning 状态下禁止买入但允许卖出

---

## P6：LLM 事件驱动与文本情感分析

**目标**：让 LLM 做它最擅长的事——事件驱动与文本情感分析，而非看技术指标

### 步骤

1. **增加新闻获取模块**
   - 在 `daily_scan_push.py` 中新增 `fetch_stock_news()` 函数
   - 使用 AKShare 的 `stock_news_em` 接口（东方财富个股新闻）
   - 获取候选股最近5条核心新闻标题
   - 增加 try-except 和超时处理，新闻获取失败不影响主流程

2. **修改 `daily_scan_push.py` 的 Step 6（LLM分析）**
   - 对每只分析股票：
     a. 先调用技术面分析（现有 `analyze_stock()` 逻辑）
     b. 再获取新闻并调用事件面分析（`analyze_event()`）
   - 两者合并推送到 Server 酱
   - 事件分析结果格式：
     ```
     风险：[无/低/中/高] - [一句话说明]
     催化：[无/弱/强] - [一句话说明]
     ```

3. **增加事件风险否决逻辑**
   - 如果 `analyze_event()` 返回"风险：高"，则：
     - 如果是买入候选股：自动取消该股的买入计划
     - 如果是持仓股：在推送中标注⚠️高风险提醒
   - 这是定性防雷辅助，不自动卖出持仓

4. **修改推送消息格式**
   - AI分析板块拆分为两部分：
     - 📊 技术面分析（现有）
     - 📰 事件面分析（新增）

**验证**：
- 确认新闻获取正常（AKShare接口可用）
- 确认事件分析结果出现在推送中
- 确认高风险股票的买入被自动取消
- 确认新闻获取失败时不影响主流程

---

## 实施顺序与依赖关系

```
P1 (清理) ─→ P2 (止损优先) ─→ P3 (尾盘策略)
                            ─→ P4 (行业约束)
                            ─→ P5 (领先指标) ─→ P6 (事件分析)
```

- P1 是代码卫生，为后续修改打基础
- P2 是核心风控修复，优先级最高
- P3/P4/P5 可并行，但建议按顺序实施
- P5 的3级状态变更会影响 P3/P4 的代码，建议先做 P5 再做 P3
- P6 依赖 AKShare 接口可用性，独立于其他改动

**建议执行顺序**：P1 → P2 → P5 → P4 → P3 → P6

---

## 风险与注意事项

1. **P2 修改后需充分测试**：风控止损优先是核心逻辑变更，必须确保：
   - 风控类卖出（熔断/0AMV翻空）不受天数限制
   - 止盈/信号类卖出仍受3天限制
   - 不引入新的逻辑漏洞

2. **P5 的3级状态需要向下兼容**：`is_panic()` 必须保留为 bool 返回，避免破坏 `paper_trade_v90.py` 等其他调用方

3. **P3 的定时任务修改**：需要同时修改 Windows Task Scheduler 的2个任务配置，改为3个任务

4. **P6 的 AKShare 接口**：`stock_news_em` 可能有频率限制，需要增加缓存和去重逻辑

5. **所有修改完成后**：运行一次完整的 `daily_scan_push.py`，确认8步流水线正常
