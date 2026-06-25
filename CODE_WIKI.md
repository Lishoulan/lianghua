# 量化潜伏系统 Code Wiki

> A股潜伏策略量化交易系统 — 基于威科夫量价理论 + VPA量价分析的全市场信号筛选与推送平台

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [核心模块详解](#4-核心模块详解)
   - 4.1 [classic_ta — 核心策略包](#41-classic_ta--核心策略包)
   - 4.2 [classic_ta/common — 公共模块](#42-classic_tacommon--公共模块)
   - 4.3 [ml_strategy — 机器学习策略包](#43-ml_strategy--机器学习策略包)
   - 4.4 [wechat_push — 微信公众号推送包](#44-wechat_push--微信公众号推送包)
   - 4.5 [根目录脚本](#45-根目录脚本)
5. [策略版本演进](#5-策略版本演进)
6. [依赖关系图](#6-依赖关系图)
7. [数据流与运行方式](#7-数据流与运行方式)
8. [Docker 容器化部署](#8-docker-容器化部署)
9. [配置与环境变量](#9-配置与环境变量)
10. [GitHub Actions CI/CD](#10-github-actions-cicd)
11. [Supabase 数据库 Schema](#11-supabase-数据库-schema)
12. [测试与代码质量](#12-测试与代码质量)
13. [微信公众号订阅服务](#13-微信公众号订阅服务)
14. [附录：GitHub Actions 准时性分析](#14-附录github-actions-准时性分析)

---

## 1. 项目概述

**量化潜伏系统**是一个面向 A 股全市场（约4862只股票）的自动化量化筛选工具，融合**威科夫量价理论**与**VPA量价分析**，通过多维度量化模型识别潜伏买入信号，并通过 Server酱 + 微信公众号双通道推送到用户。

**核心策略流程：**

```
SOS锚定（确认主力入场）→ 等待情绪冰点（J值超卖+缩量+小实体）→ 入场质量评分(0~8分) → T+1限价买入 → 多级退出机制
```

**关键特性：**

- **策略迭代**：V6.0→V6.4，每版叠加新维度，当前为 V6.4 入场质量评分体系
- **OAMV择时**：基于活跃市值滞后阈值系统，日线+周线双重确认，自动切换牛熊模式
- **行业动量轮动**：追踪100+行业相对强度(RS)，只在RS排名前20%的强势行业选股
- **DuckDB列式缓存**：单文件存储4862只股票5年日线数据，增量获取，扫描效率提升10倍
- **双通道推送**：Server酱(管理员+内测组) + 微信公众号群发(订阅用户)
- **Serverless网关**：腾讯云函数处理微信回调+推送API
- **容器化部署**：Docker + docker-compose 一键部署
- **CI/CD全链路**：GitHub Actions 定时触发+单元测试+安全扫描+自动告警

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GitHub Actions / Docker Cron                          │
│     盘中梯度触发(13:15/13:30/13:50) + 盘后梯度触发(17:15/17:30/17:50)    │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ 定时触发 / workflow_dispatch
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              classic_ta/daily_push.py  （统一版每日推送主入口）            │
│  盘中模式 (09:00-15:00) │ 盘后模式                                       │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ├── common/oamv_status.py        ← OAMV择时状态
           ├── common/industry_analysis.py  ← 行业热度分析
           ├── common/stock_pool.py         ← 股票池 + 实时行情
           ├── common/scanner.py            ← 全市场扫描引擎 (ThreadPoolExecutor)
           ├── common/signal_analyzer.py    ← 信号详情分析 (威科夫+VPA)
           ├── common/message_builder.py    ← 推送消息构建 (管理员版+内测版)
           └── common/order_execution.py    ← 限价单成交判定 (滑点+安全垫)
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              V6.4 潜伏信号引擎 (继承链 V60→V61→V62→V63→V64)              │
│  Detect_AmbushSignal_V64 → 入场质量评分(0~8) → 精细动态评分过滤           │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ 行业动量过滤  │ │  OAMV择时     │ │ 精细动态评分过滤  │
│ (V62/V63)   │ │ (ml_strategy)│ │ (scanner.py)     │
└──────────────┘ └──────────────┘ └──────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│  Server酱推送             │  │  微信公众号群发                │
│  管理员组 + 内测组         │  │  HTML图文 → 上传素材 → 群发   │
│  定时投递 (14:15/18:15)  │  │  腾讯云函数 push_handler      │
└──────────────────────────┘  └──────────────────────────────┘
```

### 订阅服务架构

```
扫描计算层 (GitHub Actions / Docker)
    daily_push.py → 信号数据
        │
        ├── HTTP POST → 推送网关层
        │   腾讯云函数 / Docker HTTP
        │   ├── push_handler → 生成图文HTML → 上传素材 → 群发推文
        │   ├── wechat_handler → 微信事件回调(关注/取关/消息)
        │   ├── health_handler → 健康检查端点
        │   └── stats_handler → 运营统计端点
        │
        └── 数据层 (Supabase PostgreSQL)
            ├── users — 用户表
            ├── push_logs — 推送日志
            ├── subscription_events — 订阅事件
            └── metrics — 监控指标
```

---

## 3. 目录结构

```
TradingAgents/
├── .github/workflows/
│   ├── daily_push.yml          # CI/CD: 每日定时推送（梯度触发+备用+重试+告警）
│   ├── tests.yml               # CI: 单元测试 + 冒烟测试（Python 3.10/3.11/3.12）
│   ├── codeql.yml              # CI: CodeQL 安全扫描（每周一）
│   └── keepalive.yml           # 防止60天不活动自动禁用
├── classic_ta/                 # 核心策略包
│   ├── common/                 # 公共模块（从daily_push重构出）
│   │   ├── scanner.py          # 全市场扫描引擎（SyncScanner + AsyncScanner）
│   │   ├── stock_pool.py       # 股票池（全A获取、预筛选、实时行情、K线拼接）
│   │   ├── signal_analyzer.py  # 信号详情分析（威科夫+VPA+蜡烛图+支撑阻力）
│   │   ├── oamv_status.py      # OAMV择时状态获取
│   │   ├── industry_analysis.py# 行业热度分析（动量排名、冷热、轮动）
│   │   ├── push_channels.py    # Server酱推送通道（分组+定时投递+重试）
│   │   ├── message_builder.py  # 推送消息构建（管理员版+内测版）
│   │   └── order_execution.py  # 限价单成交判定（滑点+安全垫+流动性）
│   ├── v60_ambush_model.py     # V6.0 基础框架（SOS锚定+情绪冰点+7级退出）
│   ├── v61_ambush_model.py     # V6.1 Spring Test+吊灯止盈+4级退出
│   ├── v62_ambush_model.py     # V6.2 行业热度过滤（绝对阈值）
│   ├── v63_ambush_model.py     # V6.3 四维度深度优化（RS+波动率平价+VWAP+限价单）
│   ├── v64_ambush_model.py     # V6.4 入场质量评分（J值+量能+形态+均线 = 0~8分）
│   ├── daily_push.py           # 统一版每日推送主入口（V6.4精细动态评分）
│   ├── v63_mootdx_push.py      # 盘前Mootdx扫描推送
│   ├── stock_data_duckdb.py    # DuckDB数据缓存引擎（增量+线程安全+除权校验）
│   └── stock_data_cache.py     # CSV缓存（旧版，已弃用）
├── ml_strategy/                # 机器学习策略包
│   ├── oamv_filter.py          # OAMV滞后阈值择时过滤器
│   └── market_amv_cache.py     # 全市场活跃市值缓存（tushare daily_basic）
├── wechat_push/                # 微信公众号推送 + 订阅服务
│   ├── __init__.py             # 群发核心（access_token+HTML模板+事件处理）
│   ├── cloud_function.py       # 腾讯云函数入口（4个handler）
│   ├── subscription.py         # 订阅墙中间件（计划管理+权限+过期检查）
│   └── monitoring.py           # 监控埋点（推送日志+指标+健康检查+报表）
├── tests/                      # 单元测试
│   ├── test_indicator_calc.py  # 指标计算测试
│   ├── test_scanner.py         # 扫描器测试
│   ├── test_signal_detection.py# 信号检测测试
│   └── test_stock_data_duckdb.py # DuckDB缓存测试
├── docs/
│   ├── deploy.md               # 部署指南
│   ├── schema.sql              # Supabase 数据库 Schema
│   └── specs/                  # 设计文档
├── results/                    # 运行结果输出
│   ├── stock_cache.duckdb      # DuckDB股票数据缓存
│   ├── oamv_cache/             # OAMV缓存CSV
│   ├── daily/                  # 每日推送结果JSON
│   └── stock_cache/            # CSV缓存（回退模式）
├── Dockerfile                  # 容器化构建
├── docker-compose.yml          # 全栈编排（scanner + cloud-func）
├── pyproject.toml              # 项目元数据 + ruff/pytest/mypy配置
├── requirements.txt            # Python依赖
├── trigger_push.py             # GitHub API触发脚本
├── .env.example                # 环境变量模板
└── *.py                        # 回测/分析/对比脚本（20+个）
```

---

## 4. 核心模块详解

### 4.1 classic_ta — 核心策略包

#### 4.1.1 v60_ambush_model.py — V6.0 基础潜伏模型

**职责：** 策略体系基础层，定义指标计算工厂、SOS锚定、情绪冰点检测和7级退出机制。

| 函数 | 签名 | 说明 |
|------|------|------|
| `IndicatorCalcBase` | `(df: DataFrame) -> DataFrame` | 指标计算工厂，输入OHLCV输出全部技术指标 |
| `Detect_AmbushSignal` | `(df: DataFrame, params: Dict) -> DataFrame` | V6.0潜伏信号检测引擎 |

**IndicatorCalcBase 追加列：**
- `white_line` — 双重EMA(10)白线（短期趋势）
- `yellow_line` — MA(14,28,57,114)四均线黄线（中期趋势）
- `atr14` — 14日ATR（真实波幅均值）
- `volume_ma` — 20日成交量均线
- `K, D, J` — KDJ随机指标

**DEFAULT_PARAMS 关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sos_body_ratio` | 0.50 | SOS实体占比阈值 |
| `sos_vol_relative` | 1.5 | SOS量能相对阈值（需>1.5倍均量） |
| `ambush_window` | 5 | SOS后潜伏等待窗口(天) |
| `ambush_j_oversold` | 13 | 潜伏J值超卖阈值 |
| `ambush_vol_shrink` | 0.70 | 潜伏缩量阈值（量<70%均量） |
| `hard_stop_atr` | 2.0 | 硬止损ATR倍数 |
| `time_stop_days` | 8 | 时间止损天数 |

---

#### 4.1.2 v61_ambush_model.py — V6.1 Spring Test + 吊灯止盈

**职责：** 弹簧试探微确认 + 退出机制精简(7级→4级) + 吊灯追踪止盈。

| 名称 | 类型 | 说明 |
|------|------|------|
| `Detect_AmbushSignal_V61` | 函数 | 追加Spring Test微确认的信号检测 |
| `detect_buy_climax_v61` | 函数 | Buy Climax派发信号检测 |
| `StatefulTradeBacktester_V61` | 函数 | V6.1状态机回测器 |
| `compute_v61_metrics` | 函数 | 绩效指标计算 |
| `Position` | dataclass | 持仓数据结构 |
| `TradeRecord` | dataclass | 交易记录数据结构 |
| `ExitReason` | enum | 退出原因枚举（ATR_HARD_STOP/CHANDELIER_EXIT/BUY_CLIMAX/TIME_STOP等） |

**Spring Test 三选一条件（OR逻辑）：**
- a) 下影线 > 实体×N（探底回升）
- b) 收盘价 > 开盘价（收阳线）
- c) J值拐头（当日J > 前日J）

---

#### 4.1.3 v62_ambush_model.py — V6.2 行业热度过滤

**职责：** 引入行业动量过滤，只在上涨趋势行业中选股。

| 函数 | 签名 | 说明 |
|------|------|------|
| `compute_industry_momentum` | `(signals_cache, industry_map, days) -> DataFrame` | 计算行业每天动量值 |
| `build_industry_allow_matrix` | `(mom_df, threshold) -> DataFrame` | 构建行业允许买入矩阵 |

**行业动量逻辑：** 110个行业 → 每行业等权平均涨幅 → 近N日累计涨幅 → 阈值过滤

---

#### 4.1.4 v63_ambush_model.py — V6.3 四维度深度优化

**职责：** 4维度量化升级 — 横截面RS + 波动率平价 + VWAP/VCP微观确认 + 限价单。

| 函数 | 签名 | 说明 |
|------|------|------|
| `add_micro_confirm_indicators` | `(df, params) -> DataFrame` | VWAP/VCP微观确认指标 |
| `Detect_AmbushSignal_V63` | `(df, params) -> DataFrame` | V6.3信号检测 |
| `StatefulTradeBacktester_V63` | 函数 | V6.3状态机回测器 |
| `calc_volatility_parity_shares` | `(equity, price, atr, hard_stop, params) -> int` | 波动率平价仓位 |
| `calc_dynamic_stop_params` | `(df, idx, params) -> tuple` | ATR百分位动态止损 |
| `calc_limit_price` | `(close, yellow, atr, params) -> float` | 限价单价格计算 |
| `compute_industry_rs_matrix` | `(signals_cache, industry_map, ...) -> tuple` | 横截面相对强度矩阵 |
| `PositionV63` | dataclass | 扩展持仓信息（含support_score等） |

**四维度优化：**

| 维度 | 改进 | 说明 |
|------|------|------|
| 横截面RS | 绝对阈值→百分位排名Top 20% | 自适应牛熊的行业过滤 |
| 波动率平价 | 固定仓位→shares=(资金×1%)/(ATR×止损倍数) | 风险标准化 |
| 微观确认 | Spring Test→VWAP/VCP右侧微确认 | 更精准的止跌信号 |
| 限价单 | 市价→限价=Close×(1-折扣) | 控制滑点 |

---

#### 4.1.5 v64_ambush_model.py — V6.4 入场质量评分（当前版本）

**职责：** 在V6.3基础上新增4维度入场质量评分(0~8分)，量化"情绪冰点+主力托底"特征。

| 函数 | 签名 | 说明 |
|------|------|------|
| `calc_obv` | `(df) -> Series` | OBV能量潮指标 |
| `add_entry_quality_indicators` | `(df, params) -> DataFrame` | 入场质量评分(0~8分) |
| `add_inst_support_indicators` | `(df, params) -> DataFrame` | 主力托底评分(旧因子体系, A+B+C+D) |
| `Detect_AmbushSignal_V64` | `(df, params) -> DataFrame` | V6.4信号检测（含评分过滤） |
| `StatefulTradeBacktester_V64` | 函数 | V6.4状态机回测器 |
| `analyze_support_score_impact` | `(trades) -> Dict` | 按评分分组分析交易表现 |
| `run_v64_backtest` | `(df, params, ...) -> List[TradeRecord]` | 便捷回测入口 |

**入场质量评分4维度（0~8分）：**

| 维度 | 评分 | 条件 |
|------|------|------|
| E1: J值深度 | 0/1/2 | J<0=极度超卖(2分), J<5=非常超卖(1分) |
| E2: 量能枯竭度 | 0/1/2 | 量<30%均量=极度枯竭(2分), <50%=非常萎缩(1分) |
| E3: 盘面形态 | 0/1/2 | 下影线>实体×2 + 实体<1.5% + 收阳 |
| E4: 黄白线关系 | 0/1/2 | 金叉/白线接近黄线/白>黄 |

**V64_PARAMS 关键新增参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `entry_quality_enabled` | True | 启用入场质量评分 |
| `entry_quality_min_score` | 3 | 最低入场质量分 |
| `eq_trend_dir_enabled` | True | 趋势方向二级过滤（黄线上升） |
| `eq_sub_filter_enabled` | True | 评分=3子模式过滤（排除弱组合） |
| `score_position_enabled` | True | 差异化仓位（高分加大，0.7x~1.5x） |
| `score_time_stop_enabled` | True | 差异化持仓时间（5~10天） |
| `breakeven_stop_enabled` | True | 保本止损（浮盈3%后回调到成本即走） |

---

#### 4.1.6 daily_push.py — 统一版每日推送主入口

**职责：** V6.4精细动态评分版每日实盘推送统一入口，整合所有子模块。

**主流程：**

```
1. 交易日检查 (tushare/akshare交易日历 → 降级为周一~周五)
2. 数据预热 (DuckDB缓存状态 + 数据源连通性检查)
3. 判断盘中/盘后模式 (09:00-15:00为盘中)
4. OAMV活跃市值择时
5. 获取行业分类 (tushare stock_basic)
6. 盘中: 获取akshare实时行情
7. 全市场扫描 (SyncScanner, 10线程, 15min超时)
8. 行业热度分析 + 行业过滤
8.5 个股动量过滤（甜蜜点: 10日跌幅<3%，评分加分+硬过滤）
9. 精细动态评分过滤 (apply_dynamic_score_filter)
10. 构建消息 → Server酱推送(定时投递) → 盘后: 公众号群发
```

**精细动态评分规则（DYNAMIC_SCORE_PARAMS）：**

```python
DYNAMIC_SCORE_PARAMS = {
    "bull_min_score": 4,              # 牛市允许4分信号
    "bull_score4_j_max": 8,           # 4分信号J值上限
    "bull_score4_vol_ratio_max": 0.70,# 4分信号量比上限
    "bear_min_score": 5,              # 熊市允许5分信号
    "j_hard_cap": 8,                  # J值硬上限（一律过滤）
}
```

**命令行：** `python classic_ta/daily_push.py [--dry-run]`

---

#### 4.1.7 stock_data_duckdb.py — DuckDB数据缓存引擎

**职责：** 将4862个CSV碎文件合并为单个DuckDB文件，列式存储+增量缓存。

| 函数 | 签名 | 说明 |
|------|------|------|
| `load_stock_cache` | `(ts_code) -> DataFrame or None` | 加载单只股票缓存 |
| `save_stock_cache` | `(ts_code, df)` | 保存数据到缓存 |
| `delete_stock_cache` | `(ts_code)` | 删除缓存（除权重建用） |
| `get_stock_data_cached` | `(ts_code, min_rows=130) -> DataFrame or None` | **核心入口**：带增量缓存的数据获取 |
| `get_stock_data_readonly` | `(ts_code, min_rows=130) -> DataFrame or None` | 纯只读获取（多线程扫描用） |
| `batch_update_stocks` | `(ts_codes) -> dict` | 批量补全缓存缺失 |
| `get_cache_stats` | `() -> dict` | 缓存统计信息 |
| `migrate_csv_to_duckdb` | `() -> dict` | 一键CSV→DuckDB迁移 |

**DuckDB表结构：** `daily_data (ts_code, date, open, high, low, close, volume)`

**增量缓存流程：**
1. 加载本地缓存（DuckDB优先，CSV回退）
2. 缓存最后日期 >= 今天且历史完整 → 直接返回（零API调用）
3. 否则获取增量（重叠一天用于除权校验）
4. 除权校验：overlap日Close偏差>1% → 全量重建
5. 正常增量合并并保存

**线程安全：** `_duckdb_write_lock` (threading.Lock) 防并发写冲突；线程本地只读连接池 (`_get_thread_local_read_conn`) 避免5000次连接创建/销毁。

**数据清洗（`_clean_dataframe`）：** 排除Volume=0停牌日、排除OHLC为NaN/0、前复权突变检测(>50%)、fillna+dropna。

---

### 4.2 classic_ta/common — 公共模块

#### 4.2.1 scanner.py — 全市场扫描引擎

| 名称 | 类型 | 说明 |
|------|------|------|
| `SyncScanner` | 类 | ThreadPoolExecutor同步扫描器（10线程, 15min超时） |
| `AsyncScanner` | 类 | asyncio+aiohttp异步扫描器（分批处理, Semaphore控制并发） |
| `apply_dynamic_score_filter` | 函数 | 精细动态评分过滤（牛熊不同阈值） |

**SyncScanner 特性：** 断点续传、盘中实时K线拼接、行业过滤、OAMV日期过滤、全局超时保护、扫描后自动补全缓存缺失。

#### 4.2.2 stock_pool.py — 股票池管理

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_all_a_stocks` | `() -> list[(ts_code, name, industry)]` | 全A股列表（排除ST/北交所/N开头） |
| `batch_prefilter_stocks` | `() -> DataFrame or None` | akshare批量预筛选（仅排除ST/北交所/退市/低价股） |
| `get_realtime_quotes` | `() -> dict{ts_code: quote}` | 全市场实时行情 |
| `append_realtime_bar` | `(df, quote) -> DataFrame` | 盘中拼接实时K线 |

#### 4.2.3 signal_analyzer.py — 信号详情分析

| 函数 | 签名 | 说明 |
|------|------|------|
| `analyze_signal_detail` | `(df, signal_idx, params) -> dict` | 威科夫解读+VPA量价+蜡烛图+支撑阻力 |

返回: `{wyckoff: [...], vpa: [...], candle: [...], support: float, resistance: float}`

#### 4.2.4 oamv_status.py — OAMV择时状态

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_oamv_status` | `() -> dict or None` | OAMV择时状态；优先AMV缓存，降级沪深300成交额代理 |

返回: `{can_open_position: bool, latest_x: float, trend_label: str, recent_states: [...], last_transition: {...}}`

#### 4.2.5 industry_analysis.py — 行业热度分析

| 函数 | 签名 | 说明 |
|------|------|------|
| `compute_industry_analysis` | `(signals_data, industry_map, params) -> list[dict]` | 行业动量排名+冷热分布+轮动信号 |
| `compute_industry_lag_signals` | `(signals, mom_df, params) -> list` | 个股动量过滤（甜蜜点: stock>-5%） |

返回每项: `{name, momentum, momentum_change, rotation, signal_count, hot_cold}`

**个股动量过滤（甜蜜点 stock > -3%）：**
- 个股10日收益 > -3% → 动量达标，评分+1分
- 个股10日收益 <= -3% → 动量不达标，硬过滤移除
- 回测: 年31笔, 胜率57%, 均收益+8.21%, 大亏45笔(比-5%阈值少36%)

#### 4.2.6 push_channels.py — Server酱推送通道

| 函数 | 签名 | 说明 |
|------|------|------|
| `send_serverchan` | `(title, desp, keys, scheduled) -> bool` | 推送（3次指数退避重试+定时投递降级为立即发送） |
| `send_group_push` | `(admin_title, admin_desp, beta_title, beta_desp, scheduled) -> dict` | 分组推送（管理员组+内测组独立重试） |

**保障链路：** 每Key最多3次重试(3s→6s→12s) → 定时投递失败降级立即发送 → 内容超长自动截断(60KB限制) → 连接池复用。

#### 4.2.7 message_builder.py — 推送消息构建器

| 函数 | 签名 | 说明 |
|------|------|------|
| `build_push_message` | `(oamv, signals, industry, params, is_intraday) -> (title, desp)` | 管理员版（完整技术Markdown） |
| `build_beta_push_message` | `(oamv, signals, industry, params, is_intraday) -> (title, desp)` | 内测版（精简卡片式） |

**信号分级：** 优先挡(评分≥8/量比<0.3/价格10~20元) → 仓位15%；普通挡 → 仓位5%

#### 4.2.8 order_execution.py — 限价单成交判定

| 函数 | 签名 | 说明 |
|------|------|------|
| `check_limit_order_fill` | `(limit_price, next_bar_ohlcv, params) -> (filled, fill_price, reason)` | T+1限价单成交判定（含滑点+安全垫+流动性检查） |
| `calc_fill_price_with_slippage` | `(base_price, slippage_pct, direction) -> float` | 含滑点成交价计算 |

**成交判定逻辑：**
- 开盘跳空低开 ≤ 限价 → 以开盘价+滑点成交
- 盘中最低价深入限价以下(含1分钱安全垫) → 以限价成交
- 最低价擦边(未穿透安全垫) → 不成交（避免未来函数）
- 成交量<均量×0.5 → 不成交（流动性不足）

---

### 4.3 ml_strategy — 机器学习策略包

#### 4.3.1 oamv_filter.py — OAMV滞后阈值择时

**核心类：** `OAMVHysteresisFilter`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `upper_threshold` | 2.0 | 牛市阈值（OAMV偏离度>+2%） |
| `lower_threshold` | -1.0 | 熊市阈值（OAMV偏离度<-1%） |
| `cost_ma_period` | 42 | 成本均线周期 |
| `smooth_method` | 'sma' | 平滑方式(sma/ema/hybrid/none) |
| `smooth_period` | 15 | 平滑周期 |

| 方法 | 说明 |
|------|------|
| `fit(amv_series/index_df)` | 拟合OAMV数据，计算状态序列 |
| `compute_oamv_proxy(index_df)` | 用成交额代理计算OAMV |
| `compute_oamv_live_chips(daily_basic_df)` | 用活跃筹码计算OAMV |
| `compute_oamv_from_series(amv_series)` | 从预计算序列直接计算 |
| `apply_hysteresis(x_t)` | 滞后阈值状态机 |
| `compute_weekly_oamv(state_df)` | 周线OAMV过滤 |
| `is_trading_allowed(date, require_weekly)` | 判断是否允许交易 |
| `get_transition_dates()` | 获取牛熊切换日期列表 |

**滞后阈值机制：** 牛市→只有偏离度<lower才切熊；熊市→只有偏离度>upper才切牛。避免阈值附近频繁切换。

#### 4.3.2 market_amv_cache.py — 全市场活跃市值缓存

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_market_amv_series` | `() -> Series` | 获取全市场活跃市值时间序列（带缓存+增量更新） |
| `fetch_market_amv` | `(start, end, cache, max_age)` | tushare daily_basic逐日聚合 |
| `load_cache` / `save_cache` | — | CSV缓存读写 |

**活跃市值公式：** `AMV = Σ(circ_mv × turnover_rate_f / 100)`

缓存文件: `results/oamv_cache/market_amv_cache.csv`，限速0.3秒/请求。

---

### 4.4 wechat_push — 微信公众号推送包

#### 4.4.1 \_\_init\_\_.py — 群发核心模块

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_access_token` | `() -> str` | 微信access_token（自动缓存刷新，提前5分钟） |
| `check_signature` | `(signature, timestamp, nonce) -> bool` | 微信签名验证 |
| `build_article_html` | `(oamv, signals, industry, is_intraday) -> str` | 图文HTML（卡片式排版） |
| `upload_news_article` | `(title, html) -> media_id` | 上传图文素材 |
| `mass_send_news` | `(media_id) -> dict` | 群发图文消息 |
| `push_signals_to_wechat` | `(oamv, signals, industry, is_intraday) -> dict` | **主入口**：信号→图文→上传→群发 |
| `handle_wechat_event` | `(xml_data) -> str` | 事件处理（subscribe/unsubscribe/text） |
| `upsert_user` | `(open_id, nickname)` | 新增/更新用户 |
| `_check_push_idempotency` | `(mode) -> bool` | 幂等锁（防重复群发） |

#### 4.4.2 cloud_function.py — 腾讯云函数入口

| Handler | API路由 | 说明 |
|---------|---------|------|
| `wechat_handler` | GET/POST `/wechat` | 微信事件回调 |
| `push_handler` | POST `/push` | 信号推送（Bearer认证） |
| `health_handler` | GET `/health` | 健康检查 |
| `stats_handler` | GET `/stats` | 运营统计（Bearer认证） |

#### 4.4.3 subscription.py — 订阅墙中间件

| 函数 | 说明 |
|------|------|
| `get_user_subscription(open_id)` | 获取用户订阅状态（plan/is_active/has_signals/days_remaining） |
| `check_push_permission(open_id)` | 检查是否有权接收推送 |
| `start_trial(open_id, nickname)` | 开通7天免费试用 |
| `upgrade_plan(open_id, new_plan, amount)` | 升级订阅计划 |
| `expire_overdue_subscriptions()` | 批量过期已到期订阅 |
| `get_subscription_stats()` | 订阅统计概览 |

**订阅计划：** trial(7天免费) → monthly(29元/月) → quarterly(79元/季) → yearly(199元/年) → expired

#### 4.4.4 monitoring.py — 监控埋点

| 函数 | 说明 |
|------|------|
| `log_push(mode, oamv, signals, ...)` | 推送日志（写入push_logs表） |
| `record_metric(name, value, tags)` | 系统指标（scan.duration/push.success等） |
| `health_check()` | 健康检查（Supabase/微信/数据源） |
| `get_operations_report(days)` | 运营报表（推送次数/成功率/信号统计） |

---

### 4.5 根目录脚本

| 脚本 | 说明 |
|------|------|
| `trigger_push.py` | GitHub API触发daily_push.yml工作流 |
| `backtest_ambush_v6.py` | 策略回测主入口 |
| `full_backtest_v64.py` | V6.4全量回测（完整绩效指标） |
| `compare_v63_v64.py` / `_large.py` | V6.3 vs V6.4 对比回测 |
| `dynamic_score_backtest.py` | 动态评分回测 |
| `fine_dynamic_backtest.py` | 精细动态评分回测 |
| `oamv_threshold_optimizer.py` | OAMV阈值优化器 |
| `param_sensitivity.py` | 参数敏感性分析 |
| `run_all_models_backtest.py` | 全模型回测运行 |
| `analyze_*.py` | 个股/模式分析脚本 |

---

## 5. 策略版本演进

```
V6.0 ─── 基础框架
  │        SOS锚定 + 情绪冰点潜伏 + 7级退出
  ▼
V6.1 ─── 风险控制
  │        + Spring Test弹簧试探 + 吊灯追踪止盈
  │        + 退出精简7级→4级 + Buy Climax
  ▼
V6.2 ─── 行业动量
  │        + 行业热度过滤（绝对阈值）
  ▼
V6.3 ─── 微观确认（四维度）
  │        + 横截面RS百分位排名Top 20%
  │        + 波动率平价仓位管理
  │        + VWAP/VCP微观止跌确认
  │        + 限价单执行 + 动态止损 + UT/AD退出
  ▼
V6.4 ─── 入场质量评分（当前版本）
           + 4维度评分(0~8): J值深度+量能枯竭+盘面形态+黄白线
           + 趋势方向过滤 + 评分子模式过滤
           + 差异化仓位(0.7x~1.5x) + 差异化持仓时间(5~10天)
           + 保本止损(浮盈3%后回调到成本即走)
           + 精细动态评分（牛熊不同门槛）
```

---

## 6. 依赖关系图

### 模块继承链

```
v60_ambush_model (IndicatorCalcBase, DEFAULT_PARAMS, Detect_AmbushSignal)
  └─ v61_ambush_model (Position, TradeRecord, ExitReason, Detect_AmbushSignal_V61,
  │    detect_buy_climax_v61, StatefulTradeBacktester_V61, V61_PARAMS)
  │    └─ v62_ambush_model (compute_industry_momentum, build_industry_allow_matrix, V62_PARAMS)
  │         └─ v63_ambush_model (add_micro_confirm_indicators, Detect_AmbushSignal_V63,
  │              calc_volatility_parity_shares, calc_dynamic_stop_params, calc_limit_price,
  │              compute_industry_rs_matrix, PositionV63, V63_PARAMS)
  │              └─ v64_ambush_model (add_entry_quality_indicators, Detect_AmbushSignal_V64,
  │                   add_inst_support_indicators, StatefulTradeBacktester_V64, V64_PARAMS)
  │
  └─ daily_push.py ← 统一版主入口
       ├── common/scanner.py (SyncScanner, apply_dynamic_score_filter)
       ├── common/stock_pool.py (get_all_a_stocks, batch_prefilter_stocks)
       ├── common/oamv_status.py → ml_strategy/oamv_filter.py + market_amv_cache.py
       ├── common/industry_analysis.py → v62/compute_industry_momentum
       ├── common/signal_analyzer.py (analyze_signal_detail)
       ├── common/message_builder.py (build_push_message, build_beta_push_message)
       ├── common/push_channels.py (send_group_push)
       ├── stock_data_duckdb.py (get_cache_stats)
       └── wechat_push/__init__.py (push_signals_to_wechat) [仅盘后]
```

### 外部依赖

| 库 | 用途 |
|---|------|
| `tushare` | 股票日线数据、行业分类、daily_basic、交易日历 |
| `akshare` | 股票日线数据（优先，无限流）+ 实时行情预筛选 |
| `mootdx` | 通达信数据源（盘前扫描） |
| `pandas` + `numpy` | 数据处理与数值计算 |
| `duckdb` | 列式缓存存储（单文件，~500MB） |
| `requests` | HTTP（Server酱/微信API/Supabase/GitHub） |
| `python-dotenv` | 环境变量管理 |
| `tenacity` | 重试机制（指数退避） |

---

## 7. 数据流与运行方式

### 完整数据流

```
数据源 (akshare优先 → tushare降级 → mootdx盘前)
    │
    ▼
stock_data_duckdb.get_stock_data_cached()  ← 增量缓存 + 除权校验
    │
    ▼
IndicatorCalcBase()  ← 白线/黄线/KDJ/ATR/volume_ma
    │
    ▼
add_micro_confirm_indicators()  ← V6.3 VWAP/VCP微观确认
    │
    ▼
add_entry_quality_indicators()  ← V6.4 入场质量评分(0~8)
    │
    ▼
Detect_AmbushSignal_V64()  ← 信号检测 + 评分过滤 + 趋势过滤 + 子模式过滤
    │
    ├── OAMV择时过滤 (日线+周线双重确认)
    ├── 行业RS过滤 (百分位排名Top 20%)
    └── 精细动态评分过滤 (牛市≥4/熊市≥5, J<8)
    │
    ▼
信号输出
    ├── Server酱推送 (管理员组+内测组, 定时投递14:15/18:15)
    └── 微信公众号群发 (仅盘后, HTML图文, 幂等保护)
```

### 运行方式

**1. 本地运行：**
```bash
python classic_ta/daily_push.py              # 正式推送
python classic_ta/daily_push.py --dry-run    # 仅扫描不推送
```

**2. Docker 部署：**
```bash
cp .env.example .env   # 填入API Keys
docker-compose up -d   # 启动scanner + cloud-func
```

**3. GitHub Actions 自动运行：**
- 盘中梯度：UTC 05:15/05:30/05:50（北京13:15/13:30/13:50）
- 盘后梯度：UTC 09:15/09:30/09:50（北京17:15/17:30/17:50）

**4. 手动触发：**
```bash
python trigger_push.py              # 触发工作流
python trigger_push.py --check      # 检查运行状态
```

**5. 回测：**
```bash
python full_backtest_v64.py         # V6.4全量回测
python compare_v63_v64.py           # 版本对比
python dynamic_score_backtest.py    # 动态评分回测
```

---

## 8. Docker 容器化部署

### docker-compose.yml 服务编排

| 服务 | 容器名 | 模式 | 说明 |
|------|--------|------|------|
| `scanner` | tradingagents-scanner | `RUN_MODE=cron` | 定时扫描推送（crond调度） |
| `cloud-func` | tradingagents-cloud-func | `RUN_MODE=server` | HTTP推送网关（Flask） |

**scanner 服务：**
- 内置crond调度，北京时间执行：
  - `13:30` 盘中扫描 → `intraday.log`
  - `17:30` 盘后扫描 → `after_hours.log`
  - 每周一 `03:00` keepalive
- 数据卷: `stock-cache:/app/results`（DuckDB缓存持久化）

**cloud-func 服务：**
- Flask HTTP 服务，端口8080
- 路由映射：
  - `GET/POST /wechat` → `wechat_handler`（微信回调）
  - `POST /push` → `push_handler`（信号推送）
  - `GET /health` → `health_handler`（健康检查）
  - `GET /stats` → `stats_handler`（运营统计）

**数据卷：**
- `stock-cache` — DuckDB缓存 + 推送结果持久化
- `./logs` — 日志目录

### Dockerfile

基于 `python:3.11-alpine`，安装系统依赖(gcc/musl-dev) + pip依赖，WORKDIR=/app。

---

## 9. 配置与环境变量

### .env 文件

| 变量 | 必需 | 说明 |
|------|------|------|
| `TUSHARE_TOKEN` | 是 | Tushare API Token |
| `SERVERCHAN_KEY` | 是 | Server酱管理员Key（逗号分隔多Key） |
| `SERVERCHAN_KEY_BETA` | 否 | Server酱内测组Key |
| `GITHUB_TOKEN` | 否 | GitHub PAT（trigger_push.py用） |
| `WECHAT_APP_ID` | 否 | 微信公众号AppID |
| `WECHAT_APP_SECRET` | 否 | 微信公众号AppSecret |
| `WECHAT_TOKEN` | 否 | 微信服务器配置Token |
| `WECHAT_ENCODING_AES_KEY` | 否 | 微信消息加解密Key |
| `SUPABASE_URL` | 否 | Supabase项目URL |
| `SUPABASE_KEY` | 否 | Supabase service_role Key |
| `PUSH_API_KEY` | 否 | 推送接口认证Key |
| `RUN_MODE` | 否 | Docker模式：cron=定时/server=HTTP |

---

## 10. GitHub Actions CI/CD

### 10.1 daily_push.yml — 每日定时推送

**梯度触发（6个cron + workflow_dispatch）：**

| cron (UTC) | 北京时间 | 说明 |
|------------|----------|------|
| `15 5 * * 1-5` | 13:15 | 盘中提前触发 |
| `30 5 * * 1-5` | 13:30 | 盘中主触发 |
| `50 5 * * 1-5` | 13:50 | 盘中兜底触发 |
| `15 9 * * 1-5` | 17:15 | 盘后提前触发 |
| `30 9 * * 1-5` | 17:30 | 盘后主触发 |
| `50 9 * * 1-5` | 17:50 | 盘后兜底触发 |

**Job 结构：**

| Job | 超时 | 说明 |
|-----|------|------|
| `check-duplicate` | — | 幂等检查（查询artifact是否已存在） |
| `daily-push` | 25min | 主推送（Python 3.11 + pip缓存 + DuckDB缓存） |
| `retry-push` | 25min | 失败/取消后自动重试 |
| `alert-on-failure` | — | 持续失败自动创建GitHub Issue告警 |
| `close-alerts-on-success` | — | 恢复后自动关闭告警Issue |

**幂等机制：** 通过GitHub API查询artifact名(`push-results-{date}-{slot}`)是否已存在，存在则跳过。

**缓存（actions/cache@v4）：**
- 路径: `results/stock_cache.duckdb` + `results/oamv_cache/`
- Key: `stock-cache-v4-{os}`

### 10.2 tests.yml — 单元测试

- **触发：** push to main/develop + PR to main + 手动
- **矩阵：** Python 3.10 / 3.11 / 3.12
- **步骤：** ruff lint → pytest --cov → 上传coverage.xml
- **冒烟测试：** 验证所有核心模块可正常import

### 10.3 codeql.yml — 安全扫描

- **触发：** push/PR to main + 每周一02:00 UTC
- **查询：** `security-and-quality` 套件
- **超时：** 30分钟

### 10.4 keepalive.yml — 保活

- **触发：** 每周一03:00 UTC
- **功能：** 创建空commit防止GitHub Actions因60天不活动自动禁用

---

## 11. Supabase 数据库 Schema

### 核心表

**users — 用户表**
```sql
create table users (
    open_id          text primary key,     -- 微信openid
    nickname         text,
    is_subscribed    boolean default true, -- 是否关注
    plan             text default 'trial', -- trial/free/monthly/quarterly/yearly/expired
    trial_start      timestamptz,
    trial_end        timestamptz,
    plan_start       timestamptz,
    plan_expire      timestamptz,
    subscribe_at     timestamptz default now(),
    unsubscribe_at   timestamptz,
    last_active_at   timestamptz default now(),
    created_at       timestamptz default now(),
    updated_at       timestamptz default now()
);
```

**push_logs — 推送日志表**
```sql
create table push_logs (
    id               serial primary key,
    push_time        timestamptz default now(),
    mode             text,            -- intraday/after_hours
    oamv_status      jsonb,
    signal_count     integer default 0,
    signals_json     jsonb,
    industry_stats   jsonb,
    wechat_mass_id   text,
    wechat_success   integer default 0,
    wechat_error     text,
    duration_seconds real,
    created_at       timestamptz default now()
);
```

**subscription_events — 订阅事件审计日志**
```sql
create table subscription_events (
    id          serial primary key,
    open_id     text references users(open_id),
    event_type  text,      -- subscribe/unsubscribe/trial_start/plan_upgrade/plan_expire
    from_plan   text,
    to_plan     text,
    amount      numeric(10,2),
    note        text,
    created_at  timestamptz default now()
);
```

**metrics — 系统监控指标**
```sql
create table metrics (
    id            serial primary key,
    metric_name   text,        -- scan.duration_seconds / push.wechat.success 等
    metric_value  numeric,
    metric_tags   jsonb,
    recorded_at   timestamptz default now()
);
```

### RLS 策略

所有表启用 Row Level Security，仅 `service_role` 可读写（云函数使用 service_role key），`anon` 角色无任何权限。

### 内置函数

- `expire_subscriptions()` — 自动将已过期订阅的plan更新为'expired'并记录事件
- `get_active_subscriber_count()` — 获取有效订阅用户数
- `handle_updated_at()` — users表更新时自动刷新updated_at

---

## 12. 测试与代码质量

### 测试套件 (tests/)

| 文件 | 覆盖范围 | 说明 |
|------|----------|------|
| `test_indicator_calc.py` | IndicatorCalcBase | 白线/黄线/KDJ/ATR/volume_ma计算正确性 |
| `test_scanner.py` | SyncScanner | 扫描器初始化、信号提取、动态评分过滤 |
| `test_signal_detection.py` | Detect_AmbushSignal系列 | SOS锚定、情绪冰点、入场质量评分 |
| `test_stock_data_duckdb.py` | DuckDB缓存 | 数据清洗、增量缓存、除权校验 |

**运行测试：**
```bash
python -m pytest tests/ -v --cov=classic_ta --cov=ml_strategy --cov-report=term-missing
```

### 代码质量工具

| 工具 | 配置 | 说明 |
|------|------|------|
| Ruff | line-length=120, target=py310 | 代码风格（E/W/F/I/B/UP规则） |
| pytest | minversion=7.0, strict-markers | 测试框架 |
| mypy | python_version=3.10 | 类型检查 |
| CodeQL | security-and-quality | 安全扫描 |

**Ruff 忽略规则：** E501(行长,中文注释多) / B008(函数默认值调用如load_dotenv)

---

## 13. 微信公众号订阅服务

### 推送策略

| 时段 | 管理员组(Server酱) | 内测组(Server酱) | 公众号(群发) |
|------|-------------------|-----------------|-------------|
| 14:15盘中 | 完整技术版 | 精简卡片版 | — |
| 18:15盘后 | 完整技术版 | 精简卡片版 | HTML图文版 |

### 微信API流程

1. `GET /cgi-bin/token` → 获取access_token（缓存2小时，提前5分钟刷新）
2. `POST /cgi-bin/material/add_news` → 上传图文素材 → media_id
3. `POST /cgi-bin/message/mass/sendall` → 群发 → msg_id

### 幂等保护

通过 Supabase push_logs 表查询今日是否已成功群发（`wechat_success > 0`），已发送则跳过，防止 GitHub Actions 备用触发/重试导致重复群发。配合唯一性约束 `(push_date, mode)` 使用 upsert + `ignore-duplicates`。

### 订阅计划与变现

| 计划 | 价格 | 有效期 | 信号推送 |
|------|------|--------|---------|
| trial | 免费 | 7天 | 有 |
| monthly | 29元 | 30天 | 有 |
| quarterly | 79元 | 90天 | 有 |
| yearly | 199元 | 365天 | 有 |
| expired | — | — | 无 |

**变现路径：** 阶段1(群发积累用户) → 阶段2(群发摘要+订阅墙) → 阶段3(认证服务号+微信支付)

---

## 14. 附录：GitHub Actions 准时性分析

### 定时投递链路

```
盘中：UTC 05:15~05:50 → GH Actions扫描(~20min) → Server酱 scheduled=14:15 → 准时到达微信
盘后：UTC 09:15~09:50 → GH Actions扫描(~20min) → Server酱 scheduled=18:15 → 准时到达微信
```

核心设计：扫描完成时间与到达时间解耦，由Server酱服务端保证准时投递。

### 容错机制

| 机制 | 说明 |
|------|------|
| 梯度触发（3个cron/时段） | 提前+主+兜底，覆盖调度延迟 |
| Server酱定时投递 | 扫描完成即提交，到达时间由Server酱保证 |
| retry-push job | 主push失败后自动重试 |
| alert-on-failure | 持续失败自动创建Issue告警 |
| close-alerts-on-success | 恢复后自动关闭告警 |
| 幂等检查 | check-duplicate + artifact查询防重复推送 |

### 性能指标

| 指标 | 首次运行 | 缓存命中 |
|------|---------|---------|
| 全市场扫描(4862只) | ~15min | ~3min |
| OAMV择时计算 | ~30s | ~1s(缓存) |
| GitHub Actions总耗时 | ~20min | ~5min |
| 准时到达率 | ~85%正常 / ~12%轻微延迟 / ~3%失败 |
