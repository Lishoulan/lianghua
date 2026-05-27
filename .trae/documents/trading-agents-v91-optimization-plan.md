# TradingAgents v9.1 优化实施计划

基于用户提供的四大维度优化建议，按优先级排序实施。

---

## P0：修复定时任务中 .env 加载 + LLM API 健壮性

### 问题
2026-05-27 早盘定时运行时 `llm_analyses={}`，DeepSeek API 调用失败。
根因：定时任务 CWD 不是项目路径，`load_dotenv(Path(__file__).parent / ".env")` 可能找不到 `.env`。

### 实施步骤

1. **修改 `daily_scan_push.py` 的 load_dotenv 调用**
   - 将 `load_dotenv(Path(__file__).parent / ".env")` 改为使用 `dotenv_path` 参数 + `override=True`
   - 添加绝对路径 fallback：
   ```python
   env_path = Path(__file__).resolve().parent / ".env"
   load_dotenv(dotenv_path=env_path, override=True)
   ```

2. **修改 `llm_analyzer.py` 的 API 调用健壮性**
   - 增加重试机制（3次重试，指数退避）
   - 增加超时时间到 60 秒
   - 增加错误日志输出（打印 HTTP 状态码和错误信息）
   - 在 `_call_api` 中增加 `.env` 兜底加载：
   ```python
   if not self.api_key:
       from dotenv import load_dotenv
       from pathlib import Path
       env_path = Path(__file__).resolve().parent.parent / ".env"
       load_dotenv(dotenv_path=env_path, override=True)
   ```

3. **修改 `live_trader.py` 的 load_dotenv 同步**
   - 同样改为 `dotenv_path + override=True`

4. **验证**
   - 手动运行 `daily_scan_push.py`，确认 `llm_analyses` 非空
   - 检查 Server 酱推送中包含 AI 分析板块

---

## P1：代码清理与架构统一

### 实施步骤

1. **删除废弃的 `holdings.json`**
   - 删除 `portfolio_state/holdings.json`（已被 `portfolio.json` 替代）

2. **归档旧版模块**
   - 在 `ml_strategy/` 下创建 `archive/` 子目录
   - 移入以下文件：
     - `pipeline.py`（旧版推理管线）
     - `portfolio_optimizer.py`（旧版组合优化）
     - `kan_predictor.py`（旧版KAN预测器）
     - `lgb_predictor.py`（LightGBM，未使用）
     - `isotonic_calibrator.py`（v10.0备用）
     - `dynamic_spread.py`（v10.0备用）
     - `fractional_kelly.py`（v10.0备用）

3. **同步 `live_trader.py`**
   - 添加 `llm_analyzer.py` 导入和 LLM 分析步骤
   - 确保与 `daily_scan_push.py` 的持仓管理逻辑一致

---

## P2：修改"最小持仓3天"逻辑 — 风控止损优先

### 问题
当前 `check_sell_conditions()` 中，`hold_days < MIN_HOLD_DAYS` 直接 return None，
导致熔断/0AMV翻空等风控卖出也被3天规则阻断。T+1日突发利空无法逃顶。

### 实施步骤

1. **修改 `check_sell_conditions()` 函数**
   - 将风控类卖出（熔断/0AMV翻空）与止盈/信号类卖出分开判断
   - 风控类卖出（is_panic / not oamv_daily）：不受 MIN_HOLD_DAYS 限制，T+1即可
   - 止盈/信号类卖出（跌破黄线/峰值回撤/ATR止损）：仍需满足 MIN_HOLD_DAYS
   - 逻辑改为：
   ```python
   # 风控类卖出：不受持仓天数限制
   if is_panic:
       sell_reason = '熔断器触发'
   elif not oamv_daily:
       sell_reason = '0AMV日线BEAR'
   
   # 止盈/信号类卖出：需满足最小持仓天数
   if sell_reason is None and hold_days >= MIN_HOLD_DAYS:
       # 跌破黄线 / 峰值回撤 / ATR止损
       ...
   ```

2. **同步修改 `live_trader.py` 中的卖出逻辑**（如果有类似代码）

3. **验证**
   - 确认 T+1 日熔断触发时可以正常卖出
   - 确认 T+1 日止盈类卖出仍被3天规则限制

---

## P3：尾盘买入策略 — 信号时机优化

### 问题
当前 08:00 扫描基于昨日收盘数据生成信号，用户 9:30 开盘买入承担跳空滑点风险。
A股 T+1 规则下，尾盘 14:45 买入更优：当日K线定型，次日即可卖出。

### 实施步骤

1. **调整定时任务时间**
   - 早盘扫描 08:00 → 保留，但仅做**信息推送**（大势/持仓风控），不执行买入
   - 新增尾盘扫描 14:45 → 执行买入信号计算 + 自动建仓
   - 晚间扫描 18:30 → 保留，盘后确认

2. **修改 `daily_scan_push.py` 增加模式参数**
   - 添加 `--mode` 参数：`morning`（早盘信息）/ `afternoon`（尾盘交易）/ `evening`（盘后确认）
   - `morning` 模式：只推送大势判断 + 持仓风控，不执行买入
   - `afternoon` 模式：完整8步流水线（卖出+买入+LLM+推送）
   - `evening` 模式：推送当日交易总结 + 更新持仓状态

3. **更新定时任务**
   - 08:00 → `daily_scan_push.py --mode morning`
   - 14:45 → `daily_scan_push.py --mode afternoon`
   - 18:30 → `daily_scan_push.py --mode evening`

---

## P4：MVO 行业集中度约束

### 问题
当前持仓广州发展+中山公用同属公用事业板块，组合存在行业集中度风险。
`CostAwarePortfolioOptimizer` 的凸优化约束中没有行业限制。

### 实施步骤

1. **修改 `CostAwarePortfolioOptimizer.optimize()` 方法**
   - 增加 `industry_map` 参数：`{ts_code: industry_name}`
   - 在约束条件中增加：同行业权重之和 ≤ 40%
   - 实现方式：遍历行业分组，为每个行业添加 `cp.sum(w[同行业索引]) <= 0.4` 约束

2. **修改 `daily_scan_push.py` 调用处**
   - 传入 `industry_map`：从 `all_stock_data` 中提取每只候选股的行业

3. **验证**
   - 确认同行业两只股票的权重之和不超过40%

---

## P5：Panic Breaker 增加领先指标

### 问题
当前熔断条件（85%跌破MA20 / 跌停>150）是滞后指标，触发时往往已是千股跌停中后期。

### 实施步骤

1. **增加跌停家数增速指标**
   - 在 `panic_breaker.py` 中增加 `limit_down_acceleration` 检测
   - 逻辑：如果昨日跌停 N 家，今日跌停 > N×3（3倍增速），提前触发预警
   - 预警级别：`early_warning`（不强制清仓，但禁止新买入）

2. **增加市场宽度恶化速率**
   - 计算 breadth 的5日变化率
   - 如果5日内 breadth 从 <50% 飙升到 >70%，标记为加速恶化

3. **修改 `is_panic()` 返回值**
   - 从 `bool` 改为 `str`：`'normal'` / `'warning'` / `'panic'`
   - `warning`：禁止新买入，但不清仓
   - `panic`：强制清仓

4. **修改 `daily_scan_push.py` 中的熔断判断逻辑**
   - 适配新的3级返回值

---

## P6：LLM 分析升级 — 事件驱动与文本情感

### 问题
当前 LLM 输入纯量价技术指标，输出往往是模棱两可的废话。
LLM 真正擅长的是文本理解和事件分析。

### 实施步骤

1. **增加新闻/公告获取模块**
   - 使用 AKShare 的 `stock_news_em`（东方财富个股新闻）接口
   - 获取候选股最近3条核心新闻标题+摘要

2. **修改 `llm_analyzer.py` 的 prompt 构建**
   - 新增 `_build_event_prompt()` 方法
   - 输入：股票代码 + 名称 + 最近新闻标题 + 公告摘要
   - 输出：是否存在"潜在雷区"或"强政策催化"
   - Prompt 示例：
   ```
   你是A股风控分析师。以下是{name}({code})的近期新闻：
   1. [新闻标题1]
   2. [新闻标题2]
   3. [新闻标题3]
   
   请判断：
   1. 是否存在重大利空风险（减持/立案/业绩暴雷/退市风险）？
   2. 是否存在强政策催化？
   回答格式：
   风险：[无/低/中/高] - [一句话说明]
   催化：[无/弱/强] - [一句话说明]
   ```

3. **修改 `daily_scan_push.py` 的 LLM 分析步骤**
   - 先调用技术面分析（现有逻辑）
   - 再调用事件面分析（新增逻辑）
   - 两者合并推送到 Server 酱

---

## 实施优先级与依赖关系

```
P0 (立即) ─→ P1 (清理) ─→ P2 (止损优先) ─→ P3 (尾盘策略)
                                        ─→ P4 (行业约束)
                                        ─→ P5 (领先指标)
                                        ─→ P6 (事件分析)
```

- P0 是基础，必须先修复
- P1 是代码卫生，为后续修改打基础
- P2/P3/P4/P5/P6 可以并行，但建议按顺序实施
- P3（尾盘策略）需要同时修改定时任务配置
- P5（领先指标）改动较大，建议在 P2 验证后再做
- P6（事件分析）依赖 AKShare 接口可用性
