# lianghua Code Wiki

> A股潜伏策略量化交易系统 —— 基于威科夫+VPA理论的情绪冰点潜伏框架

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [核心模块详解](#4-核心模块详解)
   - 4.1 [classic_ta — 经典技术分析策略包](#41-classic_ta--经典技术分析策略包)
   - 4.2 [classic_ta/common — 公共模块（重构新增）](#42-classic_tacommon--公共模块重构新增)
   - 4.3 [ml_strategy — 机器学习策略包](#43-ml_strategy--机器学习策略包)
   - 4.4 [wechat_push — 微信公众号推送包](#44-wechat_push--微信公众号推送包)
   - 4.5 [根目录脚本](#45-根目录脚本)
5. [策略版本演进](#5-策略版本演进)
6. [依赖关系图](#6-依赖关系图)
7. [数据流与运行方式](#7-数据流与运行方式)
8. [配置与环境变量](#8-配置与环境变量)
9. [GitHub Actions CI/CD](#9-github-actions-cicd)
10. [微信公众号订阅分销方案](#10-微信公众号订阅分销方案)
11. [附录：GitHub Actions 准时性分析](#11-附录github-actions-准时性分析)

---

## 1. 项目概述

lianghua 是一个面向A股市场的**量化潜伏信号自动筛选与推送系统**，核心策略为 **"潜伏模型"（Ambush Model）**，基于威科夫（Wyckoff）理论与VPA（量价分析）方法，通过识别"需求大阳线（SOS）"后的"情绪冰点"来低吸买入，而非追涨。

**核心策略流程：**

```
SOS锚定（确认主力入场）→ 等待情绪冰点（J值超卖+缩量+小实体）→ 潜伏买入 → 多级退出机制
```

**关键特性：**
- 多版本迭代策略（V6.0 → V6.4），每版叠加新维度优化
- 行业横截面相对强度过滤
- 波动率平价仓位管理
- OAMV（活跃市值）择时系统
- DuckDB列式缓存，增量数据获取
- Server酱微信推送实盘信号（管理员组 + 内测组）
- 微信公众号群发图文推送（订阅用户）
- 腾讯云函数网关（微信事件回调 + 信号推送）
- GitHub Actions 自动化定时运行（四个cron触发点 + 备用兜底）

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GitHub Actions (daily_push.yml)                   │
│     UTC 05:30 / 09:30 (主触发) + 05:45 / 09:45 (备用兜底)           │
└──────────┬──────────────────────────────────────────────────────────┘
           │ 定时触发 / workflow_dispatch
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│              classic_ta/daily_push.py  （统一版每日推送）             │
│  盘中模式 (09:00-15:00) | 盘后模式                                   │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ├── classic_ta/common/oamv_status.py    ← OAMV择时状态
           ├── classic_ta/common/industry_analysis.py ← 行业热度分析
           ├── classic_ta/common/stock_pool.py     ← 股票池 + 实时行情
           ├── classic_ta/common/scanner.py        ← 全市场扫描引擎
           ├── classic_ta/common/signal_analyzer.py ← 信号详情分析
           └── classic_ta/common/message_builder.py ← 推送消息构建
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    V6.4 潜伏信号引擎                                   │
│  Detect_AmbushSignal_V64 → 精细动态评分过滤 → 信号输出               │
│  继承链: V60 → V61 → V62 → V63 → V64                               │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│  行业动量过滤  │ │  OAMV择时     │ │  精细动态评分过滤  │
│  (V62/V63)   │ │  (ml_strategy)│ │  (scanner.py)    │
└──────────────┘ └──────────────┘ └──────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────────┐  ┌────────────────────────────────────┐
│  推送层: Server酱         │  │  推送层: 微信公众号                  │
│  管理员组 + 内测组         │  │  (订阅用户, 图文HTML群发)            │
│  定时投递 (14:15/18:15)  │  │  wechat_push + 腾讯云函数           │
└──────────────────────────┘  └────────────────────────────────────┘
```

---

## 3. 目录结构

```
TradingAgents/
├── classic_ta/                    # 核心策略包
│   ├── common/                    # 公共模块（从daily_push重构出）
│   │   ├── __init__.py
│   │   ├── push_channels.py      # Server酱推送通道（分组+定时投递）
│   │   ├── oamv_status.py        # OAMV择时状态获取
│   │   ├── industry_analysis.py  # 行业热度分析
│   │   ├── stock_pool.py         # 股票池管理（全A获取、预筛选、实时行情）
│   │   ├── scanner.py            # 全市场扫描引擎（SyncScanner + 动态评分）
│   │   ├── signal_analyzer.py    # 信号详情分析（威科夫+VPA解读）
│   │   └── message_builder.py    # 推送消息构建器（管理员版+内测版）
│   ├── __init__.py
│   ├── v60_ambush_model.py        # V6.0 基础潜伏模型
│   ├── v61_ambush_model.py        # V6.1 Spring Test + 吊灯止盈
│   ├── v62_ambush_model.py        # V6.2 行业热度过滤
│   ├── v63_ambush_model.py        # V6.3 四维度深度优化
│   ├── v64_ambush_model.py        # V6.4 入场质量评分
│   ├── daily_push.py              # 统一版每日推送主入口（V6.4精细动态评分）
│   ├── v63_mootdx_push.py         # 盘前Mootdx扫描推送
│   ├── stock_data_cache.py        # CSV缓存（旧版，已弃用）
│   └── stock_data_duckdb.py       # DuckDB缓存（当前版本）
├── ml_strategy/                   # 机器学习策略包
│   ├── __init__.py
│   ├── market_amv_cache.py        # 全市场活跃市值缓存
│   └── oamv_filter.py             # OAMV滞后阈值择时过滤器
├── wechat_push/                   # 微信公众号推送包
│   ├── __init__.py                # 公众号群发核心模块
│   ├── cloud_function.py          # 腾讯云函数入口
│   ├── subscription.py            # 订阅墙中间件（计划管理、权限校验）
│   └── monitoring.py              # 监控埋点（推送日志、系统指标）
├── tests/                         # 单元测试
│   ├── test_indicator_calc.py
│   ├── test_scanner.py
│   ├── test_signal_detection.py
│   └── test_stock_data_duckdb.py
├── .github/workflows/
│   └── daily_push.yml             # 统一定时调度工作流（含重试+告警+自动关闭）
├── docs/specs/                    # 设计文档
│   └── 2026-06-16-wechat-subscription-design.md
├── results/                       # 运行结果输出目录
│   ├── stock_cache.duckdb         # DuckDB股票数据缓存
│   ├── oamv_cache/                # OAMV缓存CSV
│   └── daily/                     # 每日推送结果
├── .env.example                   # 环境变量模板
├── .gitignore
├── LICENSE                        # MIT License
├── README.md
├── CODE_WIKI.md                   # 本文档
├── requirements.txt               # Python依赖
├── trigger_push.py                # GitHub API触发脚本
├── backtest_ambush_v6.py          # 回测入口
├── compare_v63_v64.py             # V63 vs V64 对比回测
├── compare_v63_v64_large.py       # 大规模对比回测
├── full_backtest_v64.py           # V64 全量回测
├── v64_live_backtest.py           # V64 实盘回测
├── v64_optimized_backtest.py      # V64 优化回测
├── dynamic_score_backtest.py      # 动态评分回测
├── fine_dynamic_backtest.py       # 精细动态评分回测
├── oamv_threshold_optimizer.py    # OAMV阈值优化器
├── param_sensitivity.py           # 参数敏感性分析
├── score_filter_compare.py        # 评分过滤对比
├── run_all_models_backtest.py     # 全模型回测运行
└── analyze_*.py / backtest_*.py / check_*.py / compare_*.py  # 各类分析/回测脚本
```

---

## 4. 核心模块详解

### 4.1 classic_ta — 经典技术分析策略包

#### 4.1.1 v60_ambush_model.py — V6.0 基础潜伏模型

**职责：** 整个策略体系的基础层，定义了指标计算工厂、SOS锚定逻辑、情绪冰点检测和7级退出机制。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `IndicatorCalcBase` | `(df: pd.DataFrame) -> pd.DataFrame` | 指标计算工厂，输入OHLCV输出全部技术指标 |
| `Detect_AmbushSignal` | `(df: pd.DataFrame, params: Dict) -> pd.DataFrame` | V6.0潜伏信号检测引擎 |

**IndicatorCalcBase 追加列：**
- `white_line` — 双重EMA(10)白线（短期趋势）
- `yellow_line` — MA(14,28,57,114)均线黄线（中期趋势）
- `atr14` — 14日ATR
- `volume_ma` — 20日成交量均线
- `K, D, J` — KDJ指标

**DEFAULT_PARAMS 关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sos_body_ratio` | 0.50 | SOS实体占比阈值 |
| `sos_vol_relative` | 1.5 | SOS量能相对阈值 |
| `ambush_window` | 5 | SOS后潜伏等待窗口(天) |
| `ambush_j_oversold` | 13 | 潜伏J值超卖阈值 |
| `ambush_vol_shrink` | 0.70 | 潜伏缩量阈值 |
| `hard_stop_atr` | 2.0 | 硬止损ATR倍数 |
| `time_stop_days` | 8 | 时间止损天数 |

**7级退出机制：** 硬止损 → 追踪止盈 → 保本 → Buy Climax → 时间止损 → 死叉 → 超时平仓

---

#### 4.1.2 v61_ambush_model.py — V6.1 Spring Test + 吊灯止盈

**职责：** 在V6.0基础上实施两大P0级优化：弹簧试探微确认和退出机制精简。

**核心函数与类：**

| 名称 | 类型 | 说明 |
|------|------|------|
| `Detect_AmbushSignal_V61` | 函数 | 追加Spring Test微确认的信号检测 |
| `detect_buy_climax_v61` | 函数 | Buy Climax派发信号检测 |
| `StatefulTradeBacktester_V61` | 类 | V6.1状态机回测器 |
| `compute_v61_metrics` | 函数 | V6.1绩效指标计算 |
| `Position` | dataclass | 持仓数据结构 |
| `TradeRecord` | dataclass | 交易记录数据结构 |
| `ExitReason` | enum | 退出原因枚举 |

**V61_PARAMS 关键新增参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `spring_test_enabled` | True | 是否启用弹簧试探 |
| `spring_body_ratio` | 1.0 | 下影线>实体×N |
| `chandelier_atr_mult` | 3.0 | 吊灯止盈ATR倍数 |
| `chandelier_min_days` | 2 | 吊灯止盈最少持仓天数 |

**Spring Test 三选一条件（OR逻辑）：**
- a) 下影线 > 实体长度（探底回升）
- b) 收盘价 > 开盘价（收阳线）
- c) J值拐头（当日J > 前日J）

**退出机制精简：** 7级 → 4级（硬止损 → 吊灯止盈 → Buy Climax → 时间止损）

---

#### 4.1.3 v62_ambush_model.py — V6.2 行业热度过滤

**职责：** 引入行业动量过滤维度，只买入处于上涨趋势行业中的股票。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `compute_industry_momentum` | `(signals_cache, industry_map, momentum_days) -> pd.DataFrame` | 计算每个行业每天的动量值 |
| `build_industry_allow_matrix` | `(mom_df, threshold) -> pd.DataFrame` | 根据行业动量构建允许买入矩阵 |
| `StatefulTradeBacktester_V62` | 函数 | V6.2状态机回测器（含行业过滤） |

**V62_PARAMS 关键新增参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `industry_filter_enabled` | True | 是否启用行业过滤 |
| `industry_momentum_days` | 20 | 行业动量回看天数 |
| `industry_momentum_threshold` | 0.0 | 行业动量阈值（0=正收益才买） |

**行业动量计算逻辑：** 110个行业分类 → 每行业等权平均涨幅 → 近N日累计涨幅 → 阈值过滤

---

#### 4.1.4 v63_ambush_model.py — V6.3 四维度深度优化

**职责：** 在V6.2基础上实施4个维度的量化升级，是策略的核心版本。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `add_micro_confirm_indicators` | `(df, params) -> pd.DataFrame` | 添加VWAP/VCP微观确认指标 |
| `Detect_AmbushSignal_V63` | `(df, params) -> pd.DataFrame` | V6.3信号检测（含微观确认） |
| `StatefulTradeBacktester_V63` | 函数 | V6.3状态机回测器 |
| `calc_volatility_parity_shares` | `(cash, atr, hard_stop_atr, ...) -> int` | 波动率平价仓位计算 |
| `calc_dynamic_stop_params` | `(df, idx, params) -> tuple` | 基于ATR百分位的动态止损参数 |
| `calc_limit_price` | `(df, idx, params) -> float` | 限价单价格计算 |
| `compute_industry_rs_matrix` | `(signals_cache, industry_map, ...) -> tuple` | 横截面相对强度矩阵 |
| `build_industry_allow_matrix_v63` | `(rs_df, rotation_df, ...) -> pd.DataFrame` | V6.3行业允许买入矩阵 |

**四维度优化：**

| 维度 | 名称 | 核心改进 |
|------|------|----------|
| 一 | 横截面相对强度 | 绝对阈值 → 百分位排名Top 30% + 轮入加速 |
| 二 | 波动率平价仓位 | 固定30% → shares = (总资金×1%) / (ATR×止损倍数) |
| 三 | 智能微观止跌确认 | Spring Test → VWAP/VCP右侧微确认 |
| 四 | 限价单执行 | 市价买入 → 限价 = 收盘价×(1-折扣) |

**V63_PARAMS 关键新增参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `industry_rs_top_pct` | 0.30 | 行业RS排名前30% |
| `industry_rs_or_rotation` | True | 轮入加速条件也可买入 |
| `volatility_parity_enabled` | True | 启用波动率平价 |
| `risk_per_trade` | 0.01 | 单笔最大风险=总资金1% |
| `max_position_pct` | 0.15 | 单只仓位上限15% |
| `micro_confirm_enabled` | True | 启用微观止跌确认 |
| `dynamic_stop_enabled` | True | 启用动态止损 |
| `utad_exit_enabled` | True | UT/AD退出保护 |

**PositionV63 dataclass：** 扩展了Position，增加 `support_score`、`entry_quality_score`、`volatility_pctile` 等字段。

---

#### 4.1.5 v64_ambush_model.py — V6.4 入场质量评分

**职责：** 在V6.3基础上新增入场质量评分系统（Entry Quality Score），量化"情绪冰点+主力托底"的组合特征。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `calc_obv` | `(df: pd.DataFrame) -> pd.Series` | 计算OBV能量潮指标 |
| `add_entry_quality_indicators` | `(df, params) -> pd.DataFrame` | 计算入场质量评分(0~8分) |
| `Detect_AmbushSignal_V64` | `(df, params) -> pd.DataFrame` | V6.4信号检测（含入场质量评分） |
| `add_inst_support_indicators` | `(df, params) -> pd.DataFrame` | 主力托底评分指标（旧因子体系） |

**入场质量评分4维度（0~8分）：**

| 维度 | 评分 | 说明 |
|------|------|------|
| E1: J值深度 | 0/1/2 | J<0=极度超卖(2分), J<5=非常超卖(1分) |
| E2: 量能枯竭度 | 0/1/2 | 量<30%均量=极度枯竭(2分), <50%=非常萎缩(1分) |
| E3: 盘面形态 | 0/1/2 | 下影线>实体×2 或 实体<1.5% |
| E4: 黄白线关系 | 0/1/2 | 金叉判断/白线接近黄线 |

**V64_PARAMS 关键新增参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `entry_quality_enabled` | True | 启用入场质量评分 |
| `entry_quality_min_score` | 3 | 最低入场质量分 |
| `score_position_enabled` | True | 差异化仓位（高分加大仓位） |
| `score_position_mult` | {0:0.7...8:1.5} | 评分→仓位倍数映射 |
| `score_time_stop_enabled` | True | 差异化持仓时间 |
| `breakeven_stop_enabled` | True | 保本止损 |

---

#### 4.1.6 daily_push.py — 统一版每日推送主入口

**职责：** V6.4精细动态评分版每日实盘信号推送统一入口，整合 `classic_ta/common/` 下所有子模块，支持盘中/盘后双模式，通过Server酱分组推送到微信，盘后还触发公众号群发。

**主流程（10步）：**

```
1. 数据预热（检查缓存、验证数据源可用性）
2. 判断盘中/盘后模式（09:00-15:00为盘中）
3. OAMV活跃市值择时
4. 获取行业分类（Tushare/Akshare）
5. 盘中模式：获取 akshare 实时行情
6. 全市场扫描（SyncScanner，默认8线程）
7. 行业热度分析 + 行业过滤
8. 精细动态评分过滤（apply_dynamic_score_filter）
9. 构建推送消息（管理员版 + 内测版）
10. Server酱分组推送（定时投递 14:15/18:15）；盘后触发公众号群发
```

**命令行参数：** `--dry-run` 仅扫描不推送

---

#### 4.1.7 stock_data_duckdb.py — DuckDB股票数据缓存

**职责：** 将4855个CSV碎文件合并为单个DuckDB文件，大幅提升缓存恢复速度和存储效率。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `load_stock_cache` | `(ts_code: str) -> pd.DataFrame or None` | 加载单只股票缓存数据 |
| `save_stock_cache` | `(ts_code: str, df: pd.DataFrame)` | 保存单只股票数据到缓存 |
| `get_stock_data_cached` | `(ts_code, min_rows=130) -> pd.DataFrame or None` | 带增量缓存的股票数据获取（核心入口） |
| `get_cache_stats` | `() -> dict` | 获取缓存统计信息 |
| `migrate_csv_to_duckdb` | `() -> dict` | 一键CSV迁移到DuckDB |
| `_fetch_raw_stock_data` | `(ts_code, start_date, end_date)` | 从akshare/tushare获取原始数据 |
| `_clean_dataframe` | `(df: pd.DataFrame) -> pd.DataFrame` | 数据清洗（停牌/复权异常/NaN） |

**DuckDB表结构：** `daily_data (ts_code, date, open, high, low, close, volume)`

**数据获取优先级：** akshare（无限流）→ tushare（手动前复权）

**增量缓存流程：**
1. 加载本地缓存（DuckDB优先，CSV回退）
2. 缓存已是最新 → 直接返回（零API调用）
3. 否则只获取增量数据（缓存最后日期+1 ~ 今天）
4. 合并并保存

**线程安全：** 使用 `threading.Lock`（`_duckdb_write_lock`）防止多线程并发写入冲突。

**数据清洗规则：**
- 排除Volume=0的停牌日
- 排除OHLC为NaN或0的行
- 前复权突变检测（单日涨跌幅>50%）
- fillna + dropna

---

#### 4.1.8 v63_mootdx_push.py — 盘前扫描推送

**职责：** 使用Mootdx数据源进行盘前快速扫描，适合盘前时间窗口。

**与daily_push的区别：**
- 数据源：Mootdx（通达信）vs Tushare/Akshare
- 时效性：盘前扫描 vs 收盘后推送
- 并发：ThreadPoolExecutor并发获取

---

### 4.2 classic_ta/common — 公共模块（重构新增）

此目录是从旧版 `v63_daily_push.py` / `v64_daily_push.py` 重构而来，将各职责拆分为独立模块，提升可维护性与可测性。

#### 4.2.1 push_channels.py — Server酱推送通道

**职责：** 封装所有Server酱推送逻辑，支持多Key分组推送和Server酱定时投递。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `send_serverchan` | `(title, desp, keys=None, scheduled=None)` | 向指定Key列表推送，可设定北京时间定时投递 |
| `send_group_push` | `(admin_title, admin_desp, beta_title, beta_desp, scheduled=None)` | 分组推送：管理员组+内测组，各自消息 |

**环境变量依赖：**
- `SERVERCHAN_KEYS_ADMIN` — 管理员组Key（逗号分隔多Key）
- `SERVERCHAN_KEYS_BETA` — 内测组Key

**定时投递原理：** 利用Server酱 `scheduled` 参数（Unix时间戳），在扫描完成后立即提交，由Server酱服务端在指定时间（14:15/18:15北京时间）投递到微信，从而解耦扫描时间与到达时间。

---

#### 4.2.2 oamv_status.py — OAMV择时状态

**职责：** 获取当前市场OAMV择时状态，供主流程判断牛熊模式。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_oamv_status` | `() -> dict or None` | 获取OAMV择时状态；优先读AMV缓存，降级为沪深300成交额代理 |

**降级逻辑：** AMV缓存不存在或过期 → 用沪深300成交额作为代理指标计算OAMV偏离度

---

#### 4.2.3 industry_analysis.py — 行业热度分析

**职责：** 基于全市场扫描结果计算行业动量排名、冷热分布、轮动信号。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `compute_industry_analysis` | `(all_signals_data, industry_map, params) -> dict` | 计算行业动量排名、冷热分布、轮动信号；返回含 `hot_industries` / `cold_industries` / `rotation_signals` 的字典 |

---

#### 4.2.4 stock_pool.py — 股票池管理

**职责：** 全A股列表获取、批量预筛选、实时行情拉取、盘中K线拼接。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_all_a_stocks` | `() -> list` | 获取全A股列表（排除ST/北交所） |
| `batch_prefilter_stocks` | `() -> list` | akshare批量预筛选，减少无效扫描范围 |
| `get_realtime_quotes` | `() -> pd.DataFrame` | 获取 akshare 全市场实时行情 |
| `append_realtime_bar` | `(df, quote) -> pd.DataFrame` | 盘中将实时行情拼接为最新一根K线 |

---

#### 4.2.5 scanner.py — 全市场扫描引擎

**职责：** 基于ThreadPoolExecutor的同步并发扫描器，支持断点续传、重试机制、精细动态评分过滤。

**核心类与函数：**

| 名称 | 类型 | 说明 |
|------|------|------|
| `SyncScanner` | 类 | 基于ThreadPoolExecutor的同步扫描器，默认8线程 |
| `apply_dynamic_score_filter` | 函数 | `(signals, oamv_status, params) -> list`，精细动态评分过滤 |

**精细动态评分规则（`DYNAMIC_SCORE_PARAMS`）：**

```python
DYNAMIC_SCORE_PARAMS = {
    "bull_min_score": 4,               # 牛市允许4分信号
    "bull_score4_j_max": 8,            # 4分信号J值上限
    "bull_score4_vol_ratio_max": 0.70, # 4分信号量比上限
    "bear_min_score": 5,               # 熊市允许5分信号
    "j_hard_cap": 8,                   # J值硬上限（无论评分）
}
```

**评分档位逻辑：**
- 牛市（OAMV=bull）：评分≥4 可入场，但4分需同时满足 J<8 且量比<0.70
- 熊市（OAMV=bear）：评分≥5 才可入场
- J值硬上限：无论何种市场环境，J>8 的信号一律过滤

**断点续传：** 扫描过程中维护已处理股票集合，异常中断后可从断点继续

---

#### 4.2.6 signal_analyzer.py — 信号详情分析

**职责：** 对每个潜伏信号进行深度解读，生成威科夫理论解读、VPA量价分析、蜡烛图形态、支撑/阻力位。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `analyze_signal_detail` | `(df, signal_idx) -> dict` | 分析信号详情：威科夫解读+VPA量价+蜡烛图+支撑阻力 |

---

#### 4.2.7 message_builder.py — 推送消息构建器

**职责：** 将扫描结果、OAMV状态、行业分析组装为推送消息。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `build_push_message` | `(oamv_status, signals, industry_stats, ...) -> str` | 构建管理员组推送消息（完整技术版 Markdown） |
| `build_beta_push_message` | `(oamv_status, signals, industry_stats, ...) -> str` | 构建内测组推送消息（精简版） |

**管理员版消息结构：**
- 策略参数摘要
- 市场环境（OAMV择时 + 趋势切换历史）
- 行业风向（强势行业Top8 + 弱势行业）
- 潜伏信号（每只含价格/白黄线/J值/量比/ATR/SOS锚定日/威科夫/VPA/蜡烛图/支撑阻力/T+1参考买入/退出参数/评分）
- 页脚

---

### 4.3 ml_strategy — 机器学习策略包

#### 4.3.1 oamv_filter.py — OAMV滞后阈值择时过滤器

**职责：** 实现OAMV（活跃市值）择时系统，通过滞后阈值（Hysteresis）机制判断市场牛熊状态。

**核心类：** `OAMVHysteresisFilter`

**构造参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `upper_threshold` | 2.0 | 牛市阈值（OAMV偏离度>2%判定牛市） |
| `lower_threshold` | -1.0 | 熊市阈值（OAMV偏离度<-1%判定熊市） |
| `cost_ma_period` | 42 | 成本均线周期 |
| `smooth_method` | 'sma' | 平滑方式(sma/ema/hybrid/none) |
| `smooth_period` | 15 | 平滑周期 |

**核心方法：**

| 方法 | 说明 |
|------|------|
| `compute_oamv_proxy(index_df)` | 用成交额代理计算OAMV |
| `compute_oamv_live_chips(daily_basic_df)` | 用活跃筹码计算OAMV |
| `compute_oamv_universe(all_stock_data, ...)` | 全市场股票池计算OAMV |
| `fit(amv_series / index_df)` | 拟合OAMV数据，计算状态 |
| `get_state_df()` | 获取状态DataFrame（含oamv_state列） |

**滞后阈值逻辑：**
- 当前牛市 → 只有OAMV偏离度 < lower_threshold 才切换为熊市
- 当前熊市 → 只有OAMV偏离度 > upper_threshold 才切换为牛市
- 避免在阈值附近频繁切换

---

#### 4.3.2 market_amv_cache.py — 全市场活跃市值缓存

**职责：** 通过tushare daily_basic逐日聚合全市场活跃市值，结果缓存到本地CSV。

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `load_cache` | `() -> pd.DataFrame or None` | 加载缓存数据 |
| `save_cache` | `(df)` | 保存缓存数据 |
| `fetch_market_amv` | `(start_date, end_date, existing_cache, max_age_days)` | 获取全市场活跃市值时间序列 |
| `get_market_amv_series` | `() -> pd.Series` | 获取全市场活跃市值时间序列（带缓存） |
| `get_market_amv_series_for_backtest` | `(start_date, end_date)` | 回测专用（保留全部历史） |

**活跃市值计算公式：** `AMV = Σ(每只股票的 circ_mv × turnover_rate_f / 100)`

**缓存策略：** 增量获取，只获取缓存中缺失的交易日数据。限速0.3秒/请求。

**缓存文件：** `results/oamv_cache/market_amv_cache.csv`

---

### 4.4 wechat_push — 微信公众号推送包

#### 4.4.1 \_\_init\_\_.py — 公众号推送核心模块

**职责：** 通过微信公众号（订阅号）群发图文推文，将量化潜伏信号推送给所有关注者。

**核心功能：**
1. 微信事件回调处理（关注/取关/消息）
2. 信号接收 → 生成图文HTML → 上传素材 → 群发推文
3. access_token自动缓存刷新（2小时有效期，提前5分钟刷新）

**核心函数：**

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_access_token` | `() -> str` | 获取微信access_token，自动缓存刷新 |
| `check_signature` | `(signature, timestamp, nonce) -> bool` | 验证微信服务器签名 |
| `upsert_user` | `(open_id, nickname)` | 新增或更新用户（关注时调用） |
| `unsubscribe_user` | `(open_id)` | 用户取关时更新状态 |
| `get_subscriber_count` | `() -> int` | 获取有效订阅用户数 |
| `build_article_html` | `(oamv_status, signals, industry_stats, is_intraday) -> str` | 生成公众号图文HTML（精简卡片式） |
| `upload_news_article` | `(title, content_html, thumb_media_id) -> str` | 上传图文素材到微信服务器 |
| `mass_send_news` | `(media_id) -> dict` | 群发图文消息给所有关注者 |
| `push_signals_to_wechat` | `(oamv_status, signals, industry_stats, is_intraday) -> dict` | 主入口：接收信号数据 → 生成图文 → 上传 → 群发 |
| `handle_wechat_event` | `(xml_data) -> str` | 处理微信事件推送（subscribe/unsubscribe/文本消息） |

**信号分级逻辑：**
- **重点推荐（🔴）：** 评分≥8 / 量比<0.3(极度缩量) / 价格在10~20元最佳区间 → 仓位15%
- **关注标的（⚪）：** 其余信号 → 仓位5%

**HTML图文模板：** 卡片式排版，包含头部（日期+模式）、市场情绪（OAMV）、季节性提示、行业风向、信号卡片（分级）、仓位管理、底部免责声明。

**微信事件处理：**
- `subscribe` → 自动回复欢迎语 + 7天免费试用 + Supabase记录用户
- `unsubscribe` → 更新Supabase用户状态
- 文本消息"信号" → 回复信号获取方式
- 文本消息"帮助" → 回复使用说明
- 文本消息"订阅" → 回复订阅方案

---

#### 4.4.2 cloud_function.py — 腾讯云函数入口

**职责：** 腾讯云函数入口，提供两个处理函数，通过API网关暴露HTTP接口。

**核心函数：**

| 函数 | 说明 | API网关配置 |
|------|------|-------------|
| `wechat_handler` | 微信事件回调处理 | GET /wechat（验证签名）/ POST /wechat（接收事件） |
| `push_handler` | 接收GitHub Actions信号推送 | POST /push（需Bearer PUSH_API_KEY认证） |

**push_handler 请求格式：**
```json
{
  "oamv_status": {...},
  "signals": [...],
  "industry_stats": [...],
  "is_intraday": false
}
```

**部署方式：** 腾讯云函数 → API网关 → 绑定自定义域名（可选）

---

#### 4.4.3 subscription.py — 订阅墙中间件

**职责：** 管理用户订阅计划，校验权限，处理计划升级与过期。

**订阅计划：**

| 计划 | 说明 | 价格 |
|------|------|------|
| `trial` | 7天免费试用 | 免费 |
| `free` | 免费版（功能受限） | — |
| `monthly` | 月付 | 29元/月 |
| `quarterly` | 季付 | 79元/季 |
| `yearly` | 年付 | 199元/年 |
| `expired` | 已过期 | — |

**核心功能：** 用户权限校验、计划升级、过期自动检查、Supabase 用户表管理

---

#### 4.4.4 monitoring.py — 监控埋点

**职责：** 推送日志记录、系统指标采集、健康检查、运营统计报表。

**核心函数：**

| 函数 | 说明 |
|------|------|
| `log_push(mode, oamv_status, signals, ...)` | 推送日志记录（写入push_logs表） |
| 系统指标采集 | 写入metrics表（CPU/内存/扫描耗时等） |
| 健康检查端点 | 提供 `/health` 接口 |
| 运营统计报表 | 汇总订阅用户数、推送成功率等 |

---

### 4.5 根目录脚本

#### 4.5.1 trigger_push.py — GitHub API 触发脚本

**职责：** 通过GitHub API触发 `daily_push.yml` 工作流的 `workflow_dispatch` 事件，供本地手动触发使用。

**用法：**
```bash
python trigger_push.py              # 触发每日推送
python trigger_push.py --check      # 检查最近运行状态
```

**核心函数：**

| 函数 | 说明 |
|------|------|
| `trigger_workflow(job_type)` | 触发GitHub Actions工作流 |
| `check_recent_runs()` | 检查最近运行状态 |

**环境变量：** `GITHUB_TOKEN`、`GITHUB_REPO`

---

#### 4.5.2 回测与分析脚本

| 脚本 | 说明 |
|------|------|
| `backtest_ambush_v6.py` | 策略回测的主入口脚本 |
| `compare_v63_v64.py` | V6.3 vs V6.4 对比回测 |
| `compare_v63_v64_large.py` | 大规模对比回测 |
| `full_backtest_v64.py` | V6.4全量回测，输出完整绩效指标 |
| `v64_live_backtest.py` | V6.4实盘回测 |
| `v64_optimized_backtest.py` | V6.4优化回测 |
| `dynamic_score_backtest.py` | 动态评分回测 |
| `fine_dynamic_backtest.py` | 精细动态评分回测 |
| `oamv_threshold_optimizer.py` | OAMV阈值优化器（寻最优upper/lower组合） |
| `param_sensitivity.py` | 参数敏感性分析 |
| `score_filter_compare.py` | 评分过滤对比 |
| `run_all_models_backtest.py` | 全模型回测运行 |
| `analyze_*.py` | 各类个股/模式分析脚本 |
| `cross_model_compare.py` | 跨模型对比分析 |
| `backtest_current_push.py` | 当前推送信号回测 |
| `backtest_signal_quality.py` | 信号质量回测 |
| `backtest_stoploss_compare.py` | 止损参数对比回测 |

---

## 5. 策略版本演进

```
V6.0 ─── 基础框架
  │        SOS锚定 + 情绪冰点潜伏 + 7级退出
  │
  ▼
V6.1 ─── P0级优化
  │        + Spring Test弹簧试探微确认
  │        + 退出精简7级→4级 + 吊灯止盈
  │
  ▼
V6.2 ─── P2级优化
  │        + 行业热度过滤（绝对阈值）
  │
  ▼
V6.3 ─── 四维度深度优化
  │        + 横截面相对强度（百分位排名Top30%）
  │        + 波动率平价仓位管理
  │        + VWAP/VCP微观止跌确认
  │        + 限价单执行逻辑
  │        + 动态止损（ATR百分位）
  │        + UT/AD退出保护
  │
  ▼
V6.4 ─── 入场质量评分
           + 入场质量评分(0~8分): J值深度+量能枯竭+盘面形态+黄白线
           + 精细动态评分过滤（牛熊不同阈值）
           + 差异化仓位（高分加大仓位）
           + 差异化持仓时间（time_stop_days=7）
           + 保本止损
           + common/子模块重构（push_channels/scanner/message_builder等）
```

---

## 6. 依赖关系图

### 模块间继承/调用关系

```
v60_ambush_model (IndicatorCalcBase, DEFAULT_PARAMS, Detect_AmbushSignal)
    │
    ├── v61_ambush_model (V61_PARAMS, Detect_AmbushSignal_V61, Position, TradeRecord, ExitReason,
    │                     detect_buy_climax_v61, StatefulTradeBacktester_V61, compute_v61_metrics)
    │       │
    │       ├── v62_ambush_model (V62_PARAMS, compute_industry_momentum, build_industry_allow_matrix,
    │       │                     StatefulTradeBacktester_V62)
    │       │       │
    │       │       └── v63_ambush_model (V63_PARAMS, add_micro_confirm_indicators,
    │       │                         Detect_AmbushSignal_V63, StatefulTradeBacktester_V63,
    │       │                         calc_volatility_parity_shares, calc_dynamic_stop_params,
    │       │                         calc_limit_price, compute_industry_rs_matrix,
    │       │                         build_industry_allow_matrix_v63, PositionV63)
    │       │               │
    │       │               └── v64_ambush_model (V64_PARAMS, add_entry_quality_indicators,
    │       │                                 Detect_AmbushSignal_V64, add_inst_support_indicators,
    │       │                                 calc_obv)
    │       │
    │       └── classic_ta/daily_push.py  ← 统一版主入口
    │               │
    │               ├── common/push_channels.py   ← 分组推送+定时投递
    │               ├── common/oamv_status.py     ← OAMV状态
    │               ├── common/industry_analysis.py ← 行业分析
    │               ├── common/stock_pool.py      ← 股票池
    │               ├── common/scanner.py         ← 扫描+动态评分
    │               ├── common/signal_analyzer.py ← 信号详情
    │               ├── common/message_builder.py ← 消息构建
    │               ├── v64_ambush_model: Detect_AmbushSignal_V64, V64_PARAMS
    │               ├── stock_data_duckdb: get_stock_data_cached
    │               ├── ml_strategy.oamv_filter: OAMVHysteresisFilter
    │               └── ml_strategy.market_amv_cache: load_cache
    │
    └── stock_data_duckdb (独立数据层，被推送脚本调用)
            └── 数据源: akshare → tushare (降级)

wechat_push (独立推送模块)
    │
    ├── __init__.py (核心推送逻辑)
    │       ├── 微信API: access_token / 素材上传 / 群发
    │       ├── HTML模板: build_article_html
    │       ├── 事件处理: handle_wechat_event
    │       └── 用户管理: Supabase (upsert_user / unsubscribe_user / get_subscriber_count)
    │
    ├── cloud_function.py (腾讯云函数入口)
    │       ├── wechat_handler: 微信事件回调
    │       └── push_handler: 信号推送接口
    │
    ├── subscription.py (订阅墙中间件)
    │       └── 计划管理 + 权限校验 + Supabase用户表
    │
    └── monitoring.py (监控埋点)
            └── 推送日志 + 系统指标 + 健康检查
```

### 外部依赖

```
tushare ──────── 股票日线数据、行业分类、daily_basic、交易日历
akshare ──────── 股票日线数据（优先数据源，无限流）+ 实时行情
mootdx ───────── 通达信数据源（盘前扫描用）
pandas ───────── 数据处理
numpy ────────── 数值计算
duckdb ───────── 列式缓存存储
requests ─────── HTTP请求（Server酱推送、GitHub API、微信API、Supabase）
python-dotenv ── 环境变量管理
tenacity ─────── 重试机制
```

---

## 7. 数据流与运行方式

### 数据流

```
数据源 (akshare/tushare)
        │
        ▼
stock_data_duckdb.get_stock_data_cached()  ← 增量缓存
        │
        ▼
IndicatorCalcBase()  ← 计算技术指标 (白线/黄线/KDJ/ATR/volume_ma)
        │
        ▼
add_micro_confirm_indicators()  ← V6.3 VWAP/VCP微观确认指标
        │
        ▼
add_entry_quality_indicators()  ← V6.4 入场质量评分(0~8分)
        │
        ▼
Detect_AmbushSignal_V64()  ← 信号检测
        │
        ├── OAMV择时过滤 (ml_strategy.oamv_filter)
        ├── 行业RS过滤 (compute_industry_rs_matrix)
        ├── 入场质量评分过滤 (entry_quality_min_score)
        └── 精细动态评分过滤 (apply_dynamic_score_filter)
        │
        ▼
信号输出
        │
        ├── Server酱推送（管理员组 + 内测组，定时投递 14:15/18:15）
        └── 微信公众号群发（订阅用户，HTML图文格式）[仅盘后]
                └── 腾讯云函数 push_handler → push_signals_to_wechat()
```

### 运行方式

**1. 本地运行每日推送（统一版）：**
```bash
python classic_ta/daily_push.py              # 正式推送
python classic_ta/daily_push.py --dry-run    # 干跑模式
```

**2. 本地运行盘前扫描：**
```bash
python -m classic_ta.v63_mootdx_push
```

**3. 本地触发GitHub Actions：**
```bash
python trigger_push.py              # 触发推送工作流
python trigger_push.py --check      # 检查运行状态
```

**4. 本地回测：**
```bash
python backtest_ambush_v6.py          # 基础回测
python full_backtest_v64.py           # V6.4全量回测
python compare_v63_v64.py             # 版本对比
python dynamic_score_backtest.py      # 动态评分回测
python fine_dynamic_backtest.py       # 精细动态评分回测
```

**5. GitHub Actions自动运行：**
- 盘中推送：UTC 05:30（北京时间13:30）+ 备用兜底 UTC 05:45
- 盘后推送：UTC 09:30（北京时间17:30）+ 备用兜底 UTC 09:45
- 均支持通过 `workflow_dispatch` 手动触发

---

## 8. 配置与环境变量

### .env 文件配置

| 变量 | 必需 | 说明 | 获取方式 |
|------|------|------|----------|
| `TUSHARE_TOKEN` | 是 | Tushare API Token | https://tushare.pro/register |
| `SERVERCHAN_KEY` | 是 | Server酱管理员组Key（逗号分隔多Key） | https://sct.ftqq.com/ |
| `SERVERCHAN_KEY_BETA` | 否 | Server酱内测组Key | https://sct.ftqq.com/ |
| `GITHUB_TOKEN` | 否 | GitHub PAT（trigger_push.py用） | https://github.com/settings/tokens |
| `GITHUB_REPO` | 否 | GitHub仓库（owner/repo格式） | — |
| `WECHAT_APP_ID` | 否 | 微信公众号AppID | 微信公众平台 |
| `WECHAT_APP_SECRET` | 否 | 微信公众号AppSecret | 微信公众平台 |
| `WECHAT_TOKEN` | 否 | 微信服务器配置Token（自定义） | — |
| `WECHAT_ENCODING_AES_KEY` | 否 | 微信消息加解密Key（自定义） | — |
| `PUSH_API_KEY` | 否 | 信号推送接口认证Key（自定义） | — |
| `SUPABASE_URL` | 否 | Supabase项目URL | https://supabase.com/ |
| `SUPABASE_KEY` | 否 | Supabase API Key | https://supabase.com/ |
| `SCAN_WORKERS` | 否 | 全市场扫描并发线程数（默认8） | — |

### GitHub Secrets 配置

| Secret | 说明 |
|--------|------|
| `TUSHARE_TOKEN` | Tushare API Token |
| `SERVERCHAN_KEY` | Server酱管理员组Key |
| `SERVERCHAN_KEY_BETA` | Server酱内测组Key |
| `WECHAT_APP_ID` | 微信公众号AppID |
| `WECHAT_APP_SECRET` | 微信公众号AppSecret |
| `WECHAT_TOKEN` | 微信服务器配置Token |
| `SUPABASE_URL` | Supabase项目URL |
| `SUPABASE_KEY` | Supabase API Key |

---

## 9. GitHub Actions CI/CD

### daily_push.yml — 统一定时调度工作流

**触发配置（4个cron + workflow_dispatch）：**

| cron | UTC时间 | 北京时间 | 说明 |
|------|---------|----------|------|
| `30 5 * * 1-5` | 05:30 | 13:30 | 盘中主触发 |
| `45 5 * * 1-5` | 05:45 | 13:45 | 盘中备用兜底 |
| `30 9 * * 1-5` | 09:30 | 17:30 | 盘后主触发 |
| `45 9 * * 1-5` | 09:45 | 17:45 | 盘后备用兜底 |

**concurrency 防重复：**
```yaml
concurrency:
  group: daily-push-${{ github.event.schedule }}
  cancel-in-progress: false
```
> 注：不同cron的concurrency group不同，备用触发不会取消主触发，两者独立运行，通过 check-duplicate job 判断时段来避免实质性重复推送。

**Job 结构：**

| Job | 超时 | 说明 |
|-----|------|------|
| `check-duplicate` | 5分钟 | 判断当前时段（盘中/盘后），输出给后续job |
| `daily-push` | 30分钟 | 主推送（Python 3.11 + pip缓存 + DuckDB缓存） |
| `retry-push` | 25分钟 | 失败自动重试（needs: daily-push, if: failure()） |
| `alert-on-failure` | 5分钟 | 持续失败自动创建GitHub Issue告警 |
| `close-alerts-on-success` | 5分钟 | 成功后自动关闭告警Issue |

**缓存机制（actions/cache@v4）：**
- **缓存对象：** `results/stock_cache.duckdb` + `results/oamv_cache/`
- **缓存Key：** `stock-cache-v3-${{ runner.os }}-${{ steps.date.outputs.date }}`（按天更新）
- **回退匹配：** `restore-keys: stock-cache-v3-${{ runner.os }}-`

**上传产物：**
- `actions/upload-artifact@v4` 上传 `results/daily/`，保留30天

**支持 dry_run 模式：**
- `workflow_dispatch` 输入参数 `dry_run`，干跑模式只扫描不推送

---

## 10. 微信公众号订阅分销方案

### 架构

```
GitHub Actions (扫描计算层)
  13:30 盘中扫描 → Server酱推送管理员组+内测组（定时14:15到达）
  17:30 盘后扫描 → Server酱推送（定时18:15到达）+ POST信号到腾讯云函数 → 公众号群发
       │
       ▼
腾讯云函数 (推送网关)
  wechat_handler: 微信事件回调(关注/取关/消息)
  push_handler:   接收信号 → 生成图文HTML → 上传素材 → 群发推文
       │
  Supabase: 用户表(open_id, 关注状态, 订阅状态)
       │
  ├── 微信公众号 (图文群发给所有关注者)
  └── Server酱 (管理员组完整技术版 + 内测组精简版)
```

### 推送策略

| 时段 | 管理员组(Server酱) | 内测组(Server酱) | 公众号订阅者 |
|------|-------------------|-----------------|-------------|
| 14:15盘中 | 完整技术版 | 精简版 | 无 |
| 18:15盘后 | 完整技术版 | 精简版 | 精简图文版 |

### 微信API流程

1. 获取access_token (GET https://api.weixin.qq.com/cgi-bin/token)
2. 上传图文素材 (POST https://api.weixin.qq.com/cgi-bin/material/add_news)
3. 群发消息 (POST https://api.weixin.qq.com/cgi-bin/message/mass/sendall)

### 数据模型 (Supabase)

```sql
users (
  open_id TEXT PRIMARY KEY,
  nickname TEXT,
  subscribe_at TIMESTAMPTZ,
  is_subscribed BOOLEAN DEFAULT true,
  plan TEXT DEFAULT 'trial',  -- trial/free/monthly/quarterly/yearly/expired
  plan_expire TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
)

push_logs (
  id SERIAL PRIMARY KEY,
  push_time TIMESTAMPTZ,
  mode TEXT,
  signal_count INT,
  signals_json JSONB,
  wechat_mass_id TEXT,
  wechat_success INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
)

metrics (
  id SERIAL PRIMARY KEY,
  recorded_at TIMESTAMPTZ,
  metric_name TEXT,
  metric_value FLOAT,
  extra_json JSONB
)
```

### 变现路径

- **阶段1：** 群发完整内容，积累用户
- **阶段2：** 群发摘要+详情页订阅墙
- **阶段3：** 升级认证服务号+微信支付

---

## 11. 附录：GitHub Actions 准时性分析

### 当前cron配置与时间流

```
盘中链路：
  UTC 05:30 (北京13:30) → GH Actions调度 → 扫描约20分钟
  → Server酱 scheduled=14:15(北京时间) → 准时14:15到达微信

盘后链路：
  UTC 09:30 (北京17:30) → GH Actions调度 → 扫描约20分钟
  → Server酱 scheduled=18:15(北京时间) → 准时18:15到达微信

备用兜底（15分钟后）：
  UTC 05:45 / 09:45 → 若主触发失败或延迟，提供保底覆盖
```

### 准时性评估

| 场景 | 概率 | 说明 |
|------|------|------|
| 正常准时到达 | ~85% | GitHub Actions延迟<10分钟，Server酱定时准确 |
| 轻微延迟（<30分钟） | ~12% | GH Actions队列拥挤，仍在合理范围 |
| 严重延迟或失败 | ~3% | GH runner故障、网络超时等极端情况 |

**最大风险：** GitHub Actions调度延迟（常见5-30分钟），尤其在GitHub全球负载高峰期。

### 已有容错机制

| 机制 | 说明 |
|------|------|
| 备用触发（+15分钟兜底） | 主触发后15分钟再触发一次，保底覆盖 |
| Server酱定时投递 | 扫描完成即提交，到达时间由Server酱保证，与扫描延迟解耦 |
| retry-push job | 主push失败后自动重试，25分钟超时 |
| alert-on-failure job | 连续失败自动创建GitHub Issue告警，通知人工介入 |
| close-alerts-on-success job | 恢复正常后自动关闭告警Issue，减少噪音 |

### 已知潜在问题

| 问题 | 描述 | 影响 |
|------|------|------|
| UTC时区问题 | GitHub runner默认UTC时间，`datetime.now()` 获取UTC | 已通过Server酱scheduled参数（设定北京时间）规避 |
| 缓存日期key | 缓存key含当日日期，每天首次运行必然MISS | 首次运行需重新下载缓存，多20-30秒 |
| pip无镜像源 | `pip install` 未配置国内镜像 | 在GH runner（海外）无影响，但可能偶发慢 |
| 备用触发重复 | 备用cron与主cron的concurrency group不同，不会互相取消 | 若主触发成功，备用会再次扫描并重复推送；需check-duplicate job判断 |

### 改进建议

1. **增加pip安装超时保护：**
   ```yaml
   run: pip install --default-timeout=30 -r requirements.txt
   ```

2. **增大总Job超时时间：** 从30分钟提升到45分钟，减少大盘信号多时超时风险

3. **优化缓存key策略：** 可考虑按周缓存（降低MISS频率）：
   ```yaml
   key: stock-cache-v3-${{ runner.os }}-${{ steps.week.outputs.week }}
   ```

4. **check-duplicate逻辑加固：** 精确判断当前时段是否已有成功推送记录，防止备用触发重复推送
