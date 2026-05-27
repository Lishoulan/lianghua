# v9.1 策略回测计划

## 目标
基于 v9.1 最新策略逻辑，回测最近半年（约120个交易日）的模拟交易结果，生成可用于实盘参考的回测报告。

## 现有基础设施分析

### 可复用
- `ml_strategy/portfolio_backtest.py` — `PortfolioBacktester` 类，支持完整的持仓管理、成本计算、权益曲线
- `paper_trade_v90.py` — 数据加载、特征工程、模型训练的完整流程
- `daily_scan_push.py` — v9.1 最新的 `check_sell_conditions()` 逻辑（风控止损优先 + 3级市场状态）

### 需要修改/新增
- `paper_trade_v90.py` 的卖出逻辑仍使用旧版（MIN_HOLD_DAYS 阻断所有卖出），需同步 v9.1
- `paper_trade_v90.py` 未使用 3 级市场状态（warning/panic/normal）
- `paper_trade_v90.py` 未传入 MVO 行业集中度约束
- 缺少 A 股真实交易约束：T+1（买入当天不能卖出）、涨跌停无法成交

## 实施方案

创建新脚本 `backtest_v91.py`，基于 `paper_trade_v90.py` 的数据加载和模型训练流程，但使用 v9.1 的完整策略逻辑。

### Step 1: 创建 `backtest_v91.py` 脚本骨架

复用 `paper_trade_v90.py` 的以下部分：
- 数据加载（行业数据、股票数据、指数数据）
- 特征工程（SterileCleaner + DisagreementBuilder + SSADenoiser + PathSignature）
- 0AMV 过滤器
- 熔断器（使用 v9.1 的 3 级状态）
- RADE 集成模型训练
- CostAware MVO（传入 industry_map）

### Step 2: 使用 PortfolioBacktester 替代 SimPosition

用 `ml_strategy/portfolio_backtest.py` 的 `PortfolioBacktester` 类替代自定义的 `SimPosition`：
- 支持完整的成本计算（佣金+印花税+冲击滑点）
- 支持权益曲线追踪
- 支持最终汇总统计（Sharpe、MaxDD、胜率等）

### Step 3: 实现 v9.1 卖出逻辑

核心改动——风控止损优先：
```python
# 风控类卖出：不受持仓天数限制
if market_state == 'panic':
    sell_reason = '熔断器触发'
elif not oamv_daily:
    sell_reason = '0AMV日线BEAR'

# 止盈/信号类卖出：需满足最小持仓天数
if sell_reason is None and hold_days >= MIN_HOLD_DAYS:
    # 跌破黄线 / 峰值回撤 / ATR止损
```

### Step 4: 实现 A 股真实交易约束

1. **T+1 约束**：买入当天不能卖出（entry_date == current_date 时跳过卖出检查）
2. **涨停无法买入**：如果当日涨幅 ≥ 9.5%（非ST）或 ≥ 4.5%（ST），跳过该股买入
3. **跌停无法卖出**：如果当日跌幅 ≤ -9.5%（非ST）或 ≤ -4.5%（ST），跳过该股卖出
4. **冷却期**：卖出后 COOLDOWN_DAYS 天内不再买入同一只股票

### Step 5: 实现 warning 状态逻辑

- `market_state == 'warning'`：禁止新买入，但不清仓
- `market_state == 'panic'`：强制清仓
- `market_state == 'normal'`：正常交易

### Step 6: MVO 行业集中度约束

在调用 `portfolio_optimizer.optimize()` 时传入 `industry_map`：
```python
industry_map = {code: all_stock_data.get(code, {}).get('industry', '') for code in candidate_codes}
weights, valid_codes = portfolio_optimizer.optimize(
    candidate_codes, all_stock_data, oamv_state_df, current_date, ml_probs,
    industry_map=industry_map
)
```

### Step 7: 生成回测报告

输出内容包括：
1. **策略概览**：回测区间、初始资金、最终净值、总收益率
2. **风险指标**：年化波动率、Sharpe比率、最大回撤、最大回撤持续期
3. **交易统计**：交易笔数、胜率、平均盈利/亏损、盈亏比、平均持仓天数
4. **逐笔明细**：每笔交易的买入/卖出日期、价格、收益、持仓天数、卖出原因
5. **月度收益表**：按月统计收益率
6. **权益曲线数据**：每日净值变化（可用于绘图）
7. **与沪深300对比**：同期基准收益率

保存为 JSON 文件到 `backtest_results/` 目录。

### Step 8: .env 加载修复

同步 v9.1 的 .env 绝对路径加载逻辑：
```python
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
```

## 关键参数（与 daily_scan_push.py 保持一致）

| 参数 | 值 |
|------|-----|
| INITIAL_CASH | 1,000,000 |
| MAX_PORTFOLIO_STOCKS | 3 |
| POSITION_SIZE_PCT | 0.25 |
| MIN_HOLD_DAYS | 3 |
| COOLDOWN_DAYS | 5 |
| TRAILING_STOP_PCT | 8.0 |
| ATR_STOP_MULT | 1.5 |
| CATBOOST_BUY_THRESHOLD | 0.65 |
| OAMV_UPPER | 4.0 |
| OAMV_LOWER | -2.3 |
| PANIC_BREADTH_THRESHOLD | 0.85 |
| PANIC_LIMIT_DOWN_THRESHOLD | 150 |
| LIMIT_DOWN_ACCEL_FACTOR | 3.0 |
| BREADTH_DETERIORATION_PCT | 0.20 |
| INDUSTRY_MAX_WEIGHT | 0.40 |
| STOCK_LIMIT | 500 |
| SIM_DAYS | 120 |
| TRAIN_WINDOW_MONTHS | 12 |

## 文件结构

```
D:\Solo\TradingAgents\
├── backtest_v91.py          ← 新建（v9.1回测脚本）
├── backtest_results/        ← 新建（回测结果输出目录）
├── ml_strategy/
│   ├── portfolio_backtest.py ← 复用（PortfolioBacktester类）
│   ├── panic_breaker.py     ← 复用（v9.1 3级状态）
│   ├── cost_aware_optimizer.py ← 复用（v9.1 行业约束）
│   └── ...
```

## 注意事项

1. **避免未来函数**：模型训练数据截止日期必须早于回测起始日期，或者使用滚动训练窗口
2. **Tushare API 限频**：500只股票 × 数据加载，需注意 API 调用频率
3. **运行时间**：预计 15-30 分钟（数据加载 + 特征工程 + 模型训练 + 逐日回放）
4. **回测结果仅供参考**：历史表现不代表未来收益
