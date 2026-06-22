"""
潜伏模型V6.4 每日实盘推送（统一版）
===========================
基于V6.4优化参数 + 个股动量过滤 + 精细动态评分 + 分组推送 + 公众号群发

核心过滤链路:
  V6.4信号检测 → 行业过滤(动量>0) → 个股动量过滤(10日跌幅<3%) → 评分加分(+1) → 动量硬过滤 → 精细动态评分

个股动量过滤(甜蜜点):
  - 个股10日收益 > -3% → 动量达标，评分+1分
  - 个股10日收益 <= -3% → 动量不达标，直接移除
  - 回测验证: 年31笔, 胜率57%, 均收益+8.21%, 大亏45笔(vs原70笔)

推送内容：
  1. OAMV活跃市值择时状态
  2. 行业热度分析（行业动量排名、冷热分布、轮动信号）
  3. 潜伏买入信号（含行业过滤+精细动态评分）
  4. 持仓监控（4级退出：硬止损→吊灯止盈→Buy Climax→时间止损）

推送渠道：Server酱（管理员组+内测组）+ 微信公众号群发（盘后）
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

RESULT_DIR = Path(__file__).parent.parent / "results" / "daily"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── 公共模块导入 ──
from classic_ta.common.push_channels import send_serverchan, send_group_push
from classic_ta.common.oamv_status import get_oamv_status
from classic_ta.common.industry_analysis import compute_industry_analysis, compute_industry_lag_signals
from classic_ta.common.stock_pool import (
    get_all_a_stocks, batch_prefilter_stocks,
    get_realtime_quotes, append_realtime_bar,
)
from classic_ta.common.scanner import SyncScanner, apply_dynamic_score_filter
from classic_ta.common.message_builder import build_push_message, build_beta_push_message

# ── 模型导入 ──
from classic_ta.v64_ambush_model import V64_PARAMS
from classic_ta.v62_ambush_model import compute_industry_momentum, build_industry_allow_matrix
from classic_ta.stock_data_duckdb import get_cache_stats

# ══════════════════════════════════════════════════════════
#  优化参数（回测验证最优组合）
# ══════════════════════════════════════════════════════════
BEST_PARAMS = V64_PARAMS.copy()
BEST_PARAMS.update({
    # 信号过滤参数
    # Task 2: 评分门槛 5→4（交易+21%，胜率+2.3pp，总收益+37%）
    # Task 3: J值保持<5（放宽J值反而降低性能）
    # Task 4: SOS窗口 8→10（交易+55.8%，胜率+0.3pp，总收益+17.7%）
    "entry_quality_min_score": 4,
    "ambush_j_oversold": 5,
    "ambush_window": 10,  # 8→10（Task 4已验证最优）
    "industry_rs_top_pct": 0.20,
    # 子模式过滤（评分=3排除J0V1，评分=4排除J1V0C1M2）
    # 回测验证: J1V0C1M2胜率21.7%，过滤后score>=4总收益+63.7pp
    "eq_j_extreme": 3,
    "eq_sub_filter_score4_enabled": True,
    # 止损优化参数
    "time_stop_loss": 0.0,          # 0.01→0.0（只有浮亏才时间止损）
    "time_stop_days": 10,           # 7→10（给趋势更多时间）
    "max_hold_days": 15,            # 30→15（退出优化: 加速资金周转，胜率+4.4pp，总收益+188pp）
    "chandelier_atr_mult": 3.0,     # 3.5→3.0（更紧的吊灯止盈）
    "dynamic_chandelier_low": 2.5,  # 3.0→2.5
    "dynamic_chandelier_mid": 3.0,  # 3.5→3.0
    "dynamic_chandelier_high": 3.5, # 4.0→3.5
    "breakeven_trigger_pct": 0.02,  # 0.03→0.02（更早激活保本）
    "breakeven_min_profit_pct": 0.003,  # 0.005→0.003
    # 个股动量过滤（甜蜜点: stock > -3%）
    "lag_filter_enabled": True,                # 启用个股动量过滤
    "lag_industry_strong_threshold": 0.02,     # (保留，仅展示用)
    "lag_relative_threshold": 0.03,            # (保留，未使用)
    "lag_stock_max_return": -0.03,             # 个股N日收益下限: >-3%=最佳阈值(年31笔,胜率57%,大亏45笔)
    "lag_score_boost": 1,                      # 动量达标评分加分: +1分
    # P1-1: 信号复活机制（管线回测验证：复活信号质量差，总收益-44.2%，保持关闭）
    "revival_enabled": False,
    "revival_industry_min_score": 6,           # 行业过滤复活最低评分
    "revival_momentum_min_score": 7,           # 动量过滤复活最低评分
    "revival_momentum_min_eq_j_score": 2,      # 动量复活要求J分≥2
    # P1-2: OAMV自适应过滤阈值（管线回测验证：交易+28%总收益+51.3%，已启用）
    "adaptive_filter_enabled": True,
    "bull_industry_top_pct": 0.40,             # 牛市行业RS阈值（前40%）
    "bear_industry_top_pct": 0.20,             # 熊市行业RS阈值（前20%）
    "bull_stock_max_return": -0.05,            # 牛市个股动量下限（允许跌5%）
    "bear_stock_max_return": -0.03,            # 熊市个股动量下限（允许跌3%）
})

# 精细动态评分参数（配合entry_quality_min_score=4）
DYNAMIC_SCORE_PARAMS = {
    "bull_min_score": 4,              # 5→4（牛市允许4分信号）
    "bull_score4_j_max": 8,           # 5→8（4分信号J值上限放宽）
    "bull_score4_vol_ratio_max": 0.70,  # 0.60→0.70（4分信号量比上限放宽）
    "bear_min_score": 5,              # 6→5（熊市也允许5分信号，配合OAMV过滤）
    "j_hard_cap": 8,                  # 5→8（J值硬上限放宽，配合ambush_j_oversold=5）
}

# 定时投递目标时间（北京时间，消息到达微信的时间）
# 工作流提前30-40分钟触发，扫描完成后通过Server酱scheduled参数定时投递
DELIVERY_SCHEDULE = {
    "intraday": "14:15:00",   # 盘中推送 → 14:15准时到达
    "after_hours": "18:15:00", # 盘后推送 → 18:15准时到达
}


# ══════════════════════════════════════════════════════════
#  定时投递计算
# ══════════════════════════════════════════════════════════

# 北京时区
_BJT = timezone(timedelta(hours=8))

def _calc_scheduled_time(is_intraday):
    """计算Server酱定时投递时间

    根据当前时段（盘中/盘后）确定目标投递时间。
    如果当前时间已超过目标投递时间，则立即发送（不设定scheduled）。
    显式使用北京时区，确保在 UTC runner 上也能正确计算。

    Returns:
        str or None: "YYYY-MM-DD HH:MM:SS" 格式的北京时间，或None（立即发送）
    """
    now = datetime.now(_BJT)  # 明确使用北京时间
    today_str = now.strftime("%Y-%m-%d")

    if is_intraday:
        target = DELIVERY_SCHEDULE["intraday"]
    else:
        target = DELIVERY_SCHEDULE["after_hours"]

    scheduled_str = f"{today_str} {target}"

    # 如果当前北京时间已超过目标投递时间，立即发送
    target_dt = datetime.strptime(scheduled_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_BJT)
    if now >= target_dt:
        print(f"  当前时间已过 {target}，立即发送", flush=True)
        return None

    return scheduled_str


# ══════════════════════════════════════════════════════════
#  数据预热
# ══════════════════════════════════════════════════════════

def _is_trading_day():
    """检查今天是否为A股交易日

    优先使用 tushare 交易日历，降级为 akshare，最终降级为"周一到周五默认交易日"。
    """
    today_str = datetime.now(_BJT).strftime("%Y%m%d")

    # 方案1: tushare交易日历
    try:
        import tushare as ts
        pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
        cal = pro.trade_cal(exchange="SSE", start_date=today_str, end_date=today_str)
        if cal is not None and len(cal) > 0:
            return cal.iloc[0]["is_open"] == 1
    except Exception:
        pass

    # 方案2: akshare交易日历
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        today_fmt = datetime.now(_BJT).strftime("%Y-%m-%d")
        return today_fmt in cal['trade_date'].astype(str).values
    except Exception:
        pass

    # 降级：周一到周五默认为交易日
    return datetime.now(_BJT).weekday() < 5


def prewarm_data():
    """数据预热：健康预检 + 确保缓存就绪"""
    print("\n[预热] 数据源健康预检...", flush=True)

    cache_stats = get_cache_stats()
    print(f"  DuckDB缓存: {cache_stats.get('count', 0)}只股票 | {cache_stats.get('size_mb', 0)}MB", flush=True)

    # 快速连通性检查（每个数据源5秒超时）
    akshare_ok = False
    tushare_ok = False

    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and len(df) > 100:
            akshare_ok = True
            print(f"  ✅ akshare实时行情: 可用 ({len(df)}只)", flush=True)
        else:
            print("  ❌ akshare实时行情: 数据异常", flush=True)
    except Exception as e:
        print(f"  ❌ akshare实时行情: 不可用 ({e})", flush=True)

    try:
        import tushare as ts
        pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
        cal = pro.trade_cal(exchange="SSE", is_open="1", limit=1)
        if cal is not None and len(cal) > 0:
            tushare_ok = True
            print("  ✅ tushare接口: 可用", flush=True)
        else:
            print("  ❌ tushare接口: 返回空", flush=True)
    except Exception as e:
        print(f"  ❌ tushare接口: 不可用 ({e})", flush=True)

    if not akshare_ok and not tushare_ok:
        raise RuntimeError("❌ 所有数据源不可用，终止扫描等待重试触发")

    now = datetime.now(_BJT)
    hour = now.hour
    if 9 <= hour < 15:
        print(f"  当前时段: 盘中({hour}:00) → 使用akshare实时数据+盘中信号", flush=True)
    elif hour >= 15:
        print(f"  当前时段: 盘后({hour}:00) → 使用tushare完整日线数据", flush=True)
    else:
        print(f"  当前时段: 盘前({hour}:00) → 使用缓存数据", flush=True)

    print("[预热] 完成", flush=True)
    return {"akshare": akshare_ok, "tushare": tushare_ok}


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def daily_push():
    import time as _time
    _push_start = _time.time()

    def _step_time(step_name, step_start):
        """打印步骤耗时"""
        elapsed = _time.time() - step_start
        total_elapsed = _time.time() - _push_start
        print(f"  ⏱️ {step_name}耗时: {elapsed:.1f}s (总耗时: {total_elapsed:.0f}s)", flush=True)

    print("=" * 80, flush=True)
    print("潜伏模型V6.4 每日实盘推送（精细动态评分版）", flush=True)
    print(f"启动时间: {datetime.now(_BJT).strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"优化参数: 评分≥4 | J<5 | window=10 | mhd=15 | industry_top=20%", flush=True)
    print(f"动态评分: 牛市≥4或(4+J<8+量比<0.7) | 熊市≥5 | J<8", flush=True)
    print("=" * 80, flush=True)

    # 交易日检查
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

    # 0. 数据预热
    _t = _time.time()
    prewarm_data()
    _step_time("数据预热", _t)

    # 0.5 判断盘中/盘后模式
    now = datetime.now(_BJT)
    is_intraday = 9 <= now.hour < 15
    print(f"\n>>> {'盘中实时模式' if is_intraday else '盘后完整模式'} <<<", flush=True)

    # 1. OAMV活跃市值择时
    _t = _time.time()
    print("\n[1/6] 计算OAMV活跃市值择时...", flush=True)
    oamv_status = get_oamv_status()
    if oamv_status:
        can_open = oamv_status["can_open_position"]
        print(f"  择时状态: {'允许开仓(牛市)' if can_open else '禁止开仓(熊市)'} | "
              f"OAMV={oamv_status.get('latest_x', '?')} | {oamv_status.get('trend_label', '')}", flush=True)
    else:
        print("  OAMV计算失败", flush=True)
    _step_time("OAMV择时", _t)

    # 2. 获取行业分类
    _t = _time.time()
    print("\n[2/6] 获取行业分类...", flush=True)
    try:
        import tushare as ts
        pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
        basic = pro.stock_basic(fields="ts_code,industry", list_status="L")
        industry_map = dict(zip(basic["ts_code"], basic["industry"]))
        print(f"  行业映射: {len(industry_map)}只股票", flush=True)
    except Exception as e:
        print(f"  行业分类获取失败: {e}", flush=True)
        industry_map = {}
    _step_time("行业分类", _t)

    # 3. 盘中模式：获取实时行情
    _t = _time.time()
    realtime_quotes = None
    if is_intraday:
        print("\n[3/6] 获取akshare实时行情（盘中拼接）...", flush=True)
        realtime_quotes = get_realtime_quotes()
        if realtime_quotes:
            print(f"  实时行情: {len(realtime_quotes)}只 → 将拼接为今日实时K线", flush=True)
        else:
            print("  实时行情获取失败，回退到盘后模式", flush=True)
            is_intraday = False
    else:
        print("\n[3/6] 盘后模式，跳过实时行情获取", flush=True)
    _step_time("实时行情", _t)

    # 4. 全市场扫描（核心耗时步骤）
    _t = _time.time()
    print("\n[4/6] 全市场扫描潜伏信号...", flush=True)
    cache_stats = get_cache_stats()
    print(f"  股票缓存: {cache_stats.get('count', 0)}只 | {cache_stats.get('size_mb', 0)}MB", flush=True)
    print("  批量预筛选全市场行情...", flush=True)
    prefilter_df = batch_prefilter_stocks()
    if prefilter_df is not None:
        print(f"  预筛选结果: {len(prefilter_df)}只股票通过初筛", flush=True)
    else:
        print(f"  预筛选失败，将扫描全市场", flush=True)

    scanner = SyncScanner(BEST_PARAMS, result_dir=RESULT_DIR, scan_timeout_sec=900)
    signals, all_signals_data = scanner.scan(
        industry_allow_matrix=None,
        industry_map=industry_map,
        prefilter_df=prefilter_df,
        realtime_quotes=realtime_quotes,
    )
    _step_time("全市场扫描", _t)

    # 5. 行业热度分析 + 行业过滤
    _t = _time.time()
    print("\n[5/6] 行业热度分析...", flush=True)
    industry_stats = []
    industry_allow_matrix = None
    if all_signals_data and industry_map:
        industry_stats = compute_industry_analysis(all_signals_data, industry_map, BEST_PARAMS)
        print(f"  行业数: {len(industry_stats)}", flush=True)

        mom_days = BEST_PARAMS.get("industry_momentum_days", 10)
        mom_threshold = BEST_PARAMS.get("industry_momentum_threshold", 0.0)
        mom_df = compute_industry_momentum(all_signals_data, industry_map, mom_days)
        if not mom_df.empty:
            industry_allow_matrix = build_industry_allow_matrix(mom_df, mom_threshold)

        filtered_signals = []
        industry_killed = []  # P1-1: 信号复活机制 - 记录被行业过滤杀掉的信号
        for s in signals:
            industry = s.get("industry", "")
            if industry_allow_matrix is not None and industry and industry in industry_allow_matrix.columns:
                try:
                    signal_date = pd.Timestamp(s["signal_date"])
                    ind_val = industry_allow_matrix[industry].reindex([signal_date])
                    if not ind_val.empty and not ind_val.iloc[0]:
                        industry_killed.append(s)
                        continue
                except Exception:
                    pass
            filtered_signals.append(s)
        print(f"  行业过滤: {len(signals)}只 → {len(filtered_signals)}只", flush=True)
        signals = filtered_signals

        # 5.5 个股动量过滤（P1-2: OAMV自适应阈值）
        if signals and not mom_df.empty:
            # P1-2: 根据OAMV状态动态调整动量阈值
            is_bull = oamv_status and oamv_status.get("can_open_position", False)
            if BEST_PARAMS.get("adaptive_filter_enabled", False) and is_bull:
                adaptive_params = BEST_PARAMS.copy()
                adaptive_params["lag_stock_max_return"] = BEST_PARAMS.get("bull_stock_max_return", -0.05)
                print(f"  📈 牛市自适应: 动量阈值放宽到 {adaptive_params['lag_stock_max_return']}", flush=True)
            else:
                adaptive_params = BEST_PARAMS

            before_momentum = len(signals)
            signals = compute_industry_lag_signals(signals, mom_df, adaptive_params)
            momentum_ok_count = sum(1 for s in signals if s.get("stock_momentum_ok"))
            momentum_fail_count = before_momentum - momentum_ok_count
            print(f"  📊 个股动量过滤: {before_momentum}只 → 达标{momentum_ok_count}只 "
                  f"(不达标{momentum_fail_count}只将被动态评分过滤)", flush=True)
    _step_time("行业分析", _t)

    # 5.6 个股动量硬过滤（动量不达标的信号直接移除）
    momentum_killed = []  # P1-1: 记录被动量过滤杀掉的信号
    if BEST_PARAMS.get("lag_filter_enabled", False):
        before_hard = len(signals)
        new_signals = []
        for s in signals:
            if s.get("stock_momentum_ok", True):
                new_signals.append(s)
            else:
                momentum_killed.append(s)
        signals = new_signals
        if before_hard != len(signals):
            print(f"  动量硬过滤: {before_hard}只 → {len(signals)}只 "
                  f"(移除{before_hard - len(signals)}只动量不达标)", flush=True)

    # P1-1: 信号复活机制 —— 高质量信号被行业/动量过滤误杀时复活
    if BEST_PARAMS.get("revival_enabled", False):
        revival_count = 0
        revival_industry_min = BEST_PARAMS.get("revival_industry_min_score", 6)
        revival_momentum_min = BEST_PARAMS.get("revival_momentum_min_score", 7)
        revival_j_min = BEST_PARAMS.get("revival_momentum_min_eq_j_score", 2)

        # 复活被行业过滤杀掉的高分信号
        for s in industry_killed:
            if s.get("entry_quality_score", 0) >= revival_industry_min:
                signals.append(s)
                revival_count += 1

        # 复活被动量过滤杀掉的极高质量信号
        for s in momentum_killed:
            if (s.get("entry_quality_score", 0) >= revival_momentum_min
                  and s.get("eq_j_score", 0) >= revival_j_min):
                signals.append(s)
                revival_count += 1

        if revival_count > 0:
            print(f"  🔄 信号复活: {revival_count}只高分信号被复活 "
                  f"(行业复活分≥{revival_industry_min}, 动量复活分≥{revival_momentum_min}+J分≥{revival_j_min})", flush=True)

    # 6. 精细动态评分过滤
    _t = _time.time()
    print("\n[6/6] 精细动态评分过滤...", flush=True)
    before_dynamic = len(signals)
    signals = apply_dynamic_score_filter(signals, oamv_status, DYNAMIC_SCORE_PARAMS)
    print(f"  动态评分过滤: {before_dynamic}只 → {len(signals)}只", flush=True)
    if oamv_status:
        is_bull = oamv_status.get("can_open_position", False)
        print(f"  OAMV状态: {'牛市' if is_bull else '熊市'} | "
              f"规则: {'评分≥5或(4+J<5+量比<0.6)' if is_bull else '评分≥6'} | J<5", flush=True)
    _step_time("动态评分", _t)

    # 构建推送消息（两组格式）
    print("\n构建推送消息...", flush=True)
    admin_title, admin_desp = build_push_message(oamv_status, signals, industry_stats, BEST_PARAMS, is_intraday=is_intraday)
    beta_title, beta_desp = build_beta_push_message(oamv_status, signals, industry_stats, BEST_PARAMS, is_intraday=is_intraday)

    # 保存结果
    result = {
        "scan_time": datetime.now(_BJT).strftime("%Y-%m-%d %H:%M:%S"),
        "version": "V6.4-精细动态评分",
        "mode": "盘中实时" if is_intraday else "盘后完整",
        "params": {
            "entry_quality_min_score": BEST_PARAMS["entry_quality_min_score"],
            "ambush_j_oversold": BEST_PARAMS["ambush_j_oversold"],
            "ambush_window": BEST_PARAMS["ambush_window"],
            "industry_rs_top_pct": BEST_PARAMS["industry_rs_top_pct"],
            "eq_j_extreme": BEST_PARAMS["eq_j_extreme"],
            "time_stop_loss": BEST_PARAMS["time_stop_loss"],
            "time_stop_days": BEST_PARAMS["time_stop_days"],
            "max_hold_days": BEST_PARAMS["max_hold_days"],
            "chandelier_atr_mult": BEST_PARAMS["chandelier_atr_mult"],
            "breakeven_trigger_pct": BEST_PARAMS["breakeven_trigger_pct"],
        },
        "dynamic_score_rules": DYNAMIC_SCORE_PARAMS,
        "oamv_status": oamv_status,
        "signal_count": len(signals),
        "signals": signals,
        "industry_stats": industry_stats[:30],
    }
    result_file = RESULT_DIR / f"daily_push_{datetime.now(_BJT).strftime('%Y%m%d')}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  结果已保存: {result_file}", flush=True)

    # ══════════════════════════════════════════════════════════
    #  Server酱推送（定时投递，确保准时到达微信）
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("📤 Server酱推送", flush=True)
    print("=" * 60, flush=True)

    scheduled_time = _calc_scheduled_time(is_intraday)
    if scheduled_time:
        print(f"  📅 定时投递目标: {scheduled_time}（北京时间）", flush=True)
        print(f"  💡 扫描完成时间: {datetime.now(_BJT).strftime('%H:%M:%S')}，消息将在目标时间准时投递到微信", flush=True)
    else:
        print(f"  ⚡ 立即发送（已过目标投递时间）", flush=True)

    push_results = {"admin": False, "beta": False}
    try:
        push_results = send_group_push(admin_title, admin_desp, beta_title, beta_desp, scheduled=scheduled_time)
    except Exception as e:
        print(f"  ❌ 推送异常（非致命，继续后续流程）: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # 推送结果校验
    admin_ok = push_results.get("admin", False)
    beta_ok = push_results.get("beta", False)
    print(f"\n📊 推送结果: 管理员组={'✅成功' if admin_ok else '❌失败'} | 内测组={'✅成功' if beta_ok else '❌失败'}", flush=True)

    if not admin_ok and not beta_ok:
        print("⚠️ 所有推送通道均失败！请检查 SERVERCHAN_KEY 配置和网络连接", flush=True)
        print("  提示: GitHub Actions 的 retry-push job 将自动重试", flush=True)

    # ══════════════════════════════════════════════════════════
    #  公众号群发（仅盘后模式）
    # ══════════════════════════════════════════════════════════
    if not is_intraday:
        print("\n" + "=" * 60, flush=True)
        print("📢 公众号群发（盘后）", flush=True)
        print("=" * 60, flush=True)
        try:
            from wechat_push import push_signals_to_wechat
            wechat_result = push_signals_to_wechat(oamv_status, signals, industry_stats, is_intraday=False)
            wechat_ok = wechat_result.get("success", False)
            skipped = wechat_result.get("skipped", False)
            if skipped:
                print(f"  ⏭️ 公众号群发: 已跳过（今日已成功群发，幂等保护生效）", flush=True)
            elif wechat_ok:
                print(f"  ✅ 公众号群发: 成功", flush=True)
            else:
                print(f"  ❌ 公众号群发: 失败 - {wechat_result.get('error', '未知')}", flush=True)
        except Exception as e:
            print(f"  ❌ 公众号群发异常（非致命）: {e}", flush=True)
            import traceback
            traceback.print_exc()
    else:
        print("\n📢 盘中模式，跳过公众号群发（仅盘后群发）", flush=True)

    # 摘要
    print(f"\n{'='*80}")
    print(f"OAMV择时: {'允许(牛市)' if oamv_status and oamv_status.get('can_open_position') else '禁止(熊市)'}")
    print(f"潜伏信号: {len(signals)}只")
    for s in signals:
        eq = s.get('entry_quality_score', 0)
        print(f"  - {s['name']}({s['code']}) [{s.get('industry', '')}] "
              f"{s['price']:.2f} J:{s.get('J', 0):.1f} 评分:{eq}")
    if industry_stats:
        hot = [s for s in industry_stats if s.get("momentum", 0) > 0]
        rot_in = [s for s in industry_stats if s.get("rotation") == "轮入"]
        rot_out = [s for s in industry_stats if s.get("rotation") == "轮出"]
        print(f"行业热度: {len(hot)}个偏热 / {len(industry_stats)-len(hot)}个偏冷 | 轮入:{len(rot_in)} 轮出:{len(rot_out)}")
        if rot_in:
            print(f"  轮入: {', '.join(s['name'] for s in rot_in[:5])}")
        if rot_out:
            print(f"  轮出: {', '.join(s['name'] for s in rot_out[:5])}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="潜伏模型V6.4 每日实盘推送")
    parser.add_argument("--dry-run", action="store_true", help="仅扫描不推送")
    args = parser.parse_args()

    if args.dry_run:
        import logging
        logging.basicConfig(level=logging.DEBUG)

        def mock_send(title, desp, keys=None):
            print(f"\n[DRY-RUN] 标题: {title}")
            print(f"[DRY-RUN] 内容长度: {len(desp)}字")
            print("\n" + "=" * 80)
            print(desp[:3000])
            if len(desp) > 3000:
                print(f"\n... (截断，共{len(desp)}字)")
            print("=" * 80)

        import classic_ta.common.push_channels as _pc
        _pc.send_serverchan = mock_send

    daily_push()
