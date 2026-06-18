"""
潜伏模型V6.4 每日实盘推送（统一版）
===========================
基于V6.4优化参数 + 精细动态评分 + 分组推送 + 公众号群发

优化参数:
  - 评分阈值 3→5, J值超卖 25→5, SOS窗口 12→8
  - 行业RS前30%→前20%, J极度超卖 0→3

精细动态评分:
  - OAMV牛市: 评分>=5 或 (评分=4且J<5且量比<0.6)
  - OAMV熊市: 评分>=6
  - 所有信号 J<5

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
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

RESULT_DIR = Path(__file__).parent.parent / "results" / "daily"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── 公共模块导入 ──
from classic_ta.common.push_channels import send_serverchan, send_group_push
from classic_ta.common.oamv_status import get_oamv_status
from classic_ta.common.industry_analysis import compute_industry_analysis
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
    # 信号过滤参数（L方案：放宽评分到4分，信号数量+33%，5日胜率45.3%）
    "entry_quality_min_score": 4,  # 5→4（4分信号5日胜率47.2%，质量优于5分）
    "ambush_j_oversold": 5,
    "ambush_window": 8,
    "industry_rs_top_pct": 0.20,
    "eq_j_extreme": 3,
    # 止损优化参数（J方案：全市场5年回测总收益+778%，盈亏比2.71）
    "time_stop_loss": 0.0,          # 0.01→0.0（只有浮亏才时间止损）
    "time_stop_days": 10,           # 7→10（给趋势更多时间）
    "max_hold_days": 30,            # 20→30（延长最大持仓）
    "chandelier_atr_mult": 3.0,     # 3.5→3.0（更紧的吊灯止盈）
    "dynamic_chandelier_low": 2.5,  # 3.0→2.5
    "dynamic_chandelier_mid": 3.0,  # 3.5→3.0
    "dynamic_chandelier_high": 3.5, # 4.0→3.5
    "breakeven_trigger_pct": 0.02,  # 0.03→0.02（更早激活保本）
    "breakeven_min_profit_pct": 0.003,  # 0.005→0.003
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
# 工作流提前20-25分钟触发，扫描完成后通过Server酱scheduled参数定时投递
DELIVERY_SCHEDULE = {
    "intraday": "13:00:00",   # 盘中推送 → 13:00准时到达
    "after_hours": "21:00:00", # 盘后推送 → 21:00准时到达
}


# ══════════════════════════════════════════════════════════
#  定时投递计算
# ══════════════════════════════════════════════════════════

def _calc_scheduled_time(is_intraday):
    """计算Server酱定时投递时间

    根据当前时段（盘中/盘后）确定目标投递时间。
    如果当前时间已超过目标投递时间，则立即发送（不设定scheduled）。

    Returns:
        str or None: "YYYY-MM-DD HH:MM:SS" 格式的北京时间，或None（立即发送）
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    if is_intraday:
        target = DELIVERY_SCHEDULE["intraday"]  # 13:00:00
    else:
        target = DELIVERY_SCHEDULE["after_hours"]  # 21:00:00

    scheduled_str = f"{today_str} {target}"

    # 如果当前时间已超过目标投递时间，立即发送
    target_dt = datetime.strptime(scheduled_str, "%Y-%m-%d %H:%M:%S")
    if now >= target_dt:
        print(f"  当前时间已过 {target}，立即发送", flush=True)
        return None

    return scheduled_str


# ══════════════════════════════════════════════════════════
#  数据预热
# ══════════════════════════════════════════════════════════

def prewarm_data():
    """数据预热：确保缓存就绪，提前获取关键数据"""
    print("\n[预热] 检查数据缓存...", flush=True)

    cache_stats = get_cache_stats()
    print(f"  DuckDB缓存: {cache_stats.get('count', 0)}只股票 | {cache_stats.get('size_mb', 0)}MB", flush=True)

    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and len(df) > 100:
            print(f"  akshare实时行情: 可用 ({len(df)}只)", flush=True)
        else:
            print("  akshare实时行情: 数据异常", flush=True)
    except Exception as e:
        print(f"  akshare实时行情: 不可用 ({e})", flush=True)

    try:
        import tushare as ts
        pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
        pro.trade_cal(exchange="SSE", is_open="1", limit=1)
        print("  tushare接口: 可用", flush=True)
    except Exception as e:
        print(f"  tushare接口: 不可用 ({e})", flush=True)

    now = datetime.now()
    hour = now.hour
    if 9 <= hour < 15:
        print(f"  当前时段: 盘中({hour}:00) → 使用akshare实时数据+盘中信号", flush=True)
    elif hour >= 15:
        print(f"  当前时段: 盘后({hour}:00) → 使用tushare完整日线数据", flush=True)
    else:
        print(f"  当前时段: 盘前({hour}:00) → 使用缓存数据", flush=True)

    print("[预热] 完成", flush=True)


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def daily_push():
    print("=" * 80, flush=True)
    print("潜伏模型V6.4 每日实盘推送（精细动态评分版）", flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"优化参数: 评分≥5 | J<5 | window=8 | industry_top=20% | eq_j_extreme=3", flush=True)
    print(f"动态评分: 牛市≥5或(4+J<5+量比<0.6) | 熊市≥6 | J<5", flush=True)
    print("=" * 80, flush=True)

    # 0. 数据预热
    prewarm_data()

    # 0.5 判断盘中/盘后模式
    now = datetime.now()
    is_intraday = 9 <= now.hour < 15
    print(f"\n>>> {'盘中实时模式' if is_intraday else '盘后完整模式'} <<<", flush=True)

    # 1. OAMV活跃市值择时
    print("\n[1/6] 计算OAMV活跃市值择时...", flush=True)
    oamv_status = get_oamv_status()
    if oamv_status:
        can_open = oamv_status["can_open_position"]
        print(f"  择时状态: {'允许开仓(牛市)' if can_open else '禁止开仓(熊市)'} | "
              f"OAMV={oamv_status.get('latest_x', '?')} | {oamv_status.get('trend_label', '')}", flush=True)
    else:
        print("  OAMV计算失败", flush=True)

    # 2. 获取行业分类
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

    # 3. 盘中模式：获取实时行情
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

    # 4. 全市场扫描
    print("\n[4/6] 全市场扫描潜伏信号...", flush=True)
    cache_stats = get_cache_stats()
    print(f"  股票缓存: {cache_stats.get('count', 0)}只 | {cache_stats.get('size_mb', 0)}MB", flush=True)
    print("  批量预筛选全市场行情...", flush=True)
    prefilter_df = batch_prefilter_stocks()

    scanner = SyncScanner(BEST_PARAMS, result_dir=RESULT_DIR)
    signals, all_signals_data = scanner.scan(
        industry_allow_matrix=None,
        industry_map=industry_map,
        prefilter_df=prefilter_df,
        realtime_quotes=realtime_quotes,
    )

    # 5. 行业热度分析 + 行业过滤
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
        for s in signals:
            industry = s.get("industry", "")
            if industry_allow_matrix is not None and industry and industry in industry_allow_matrix.columns:
                try:
                    signal_date = pd.Timestamp(s["signal_date"])
                    ind_val = industry_allow_matrix[industry].reindex([signal_date])
                    if not ind_val.empty and not ind_val.iloc[0]:
                        continue
                except Exception:
                    pass
            filtered_signals.append(s)
        print(f"  行业过滤: {len(signals)}只 → {len(filtered_signals)}只", flush=True)
        signals = filtered_signals

    # 6. 精细动态评分过滤
    print("\n[6/6] 精细动态评分过滤...", flush=True)
    before_dynamic = len(signals)
    signals = apply_dynamic_score_filter(signals, oamv_status, DYNAMIC_SCORE_PARAMS)
    print(f"  动态评分过滤: {before_dynamic}只 → {len(signals)}只", flush=True)
    if oamv_status:
        is_bull = oamv_status.get("can_open_position", False)
        print(f"  OAMV状态: {'牛市' if is_bull else '熊市'} | "
              f"规则: {'评分≥5或(4+J<5+量比<0.6)' if is_bull else '评分≥6'} | J<5", flush=True)

    # 构建推送消息（两组格式）
    print("\n构建推送消息...", flush=True)
    admin_title, admin_desp = build_push_message(oamv_status, signals, industry_stats, BEST_PARAMS, is_intraday=is_intraday)
    beta_title, beta_desp = build_beta_push_message(oamv_status, signals, industry_stats, BEST_PARAMS, is_intraday=is_intraday)

    # 保存结果
    result = {
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    result_file = RESULT_DIR / f"daily_push_{datetime.now().strftime('%Y%m%d')}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  结果已保存: {result_file}", flush=True)

    # 分组推送（定时投递，确保准时到达微信）
    print("\n分组推送微信...", flush=True)
    scheduled_time = _calc_scheduled_time(is_intraday)
    if scheduled_time:
        print(f"  定时投递: {scheduled_time}（北京时间）", flush=True)
    push_results = send_group_push(admin_title, admin_desp, beta_title, beta_desp, scheduled=scheduled_time)
    print(f"推送完成: 管理员组={'成功' if push_results.get('admin') else '跳过/失败'} | "
          f"内测组={'成功' if push_results.get('beta') else '跳过/失败'}", flush=True)

    # 公众号群发（仅盘后模式）
    if not is_intraday:
        print("\n公众号群发...", flush=True)
        try:
            from wechat_push import push_signals_to_wechat
            wechat_result = push_signals_to_wechat(oamv_status, signals, industry_stats, is_intraday=False)
            print(f"公众号群发: {'成功' if wechat_result.get('success') else '失败'}", flush=True)
        except Exception as e:
            print(f"公众号群发异常: {e}", flush=True)
    else:
        print("\n盘中模式，跳过公众号群发（仅盘后群发）", flush=True)

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
