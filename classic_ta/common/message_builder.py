"""
推送消息构建模块

从 v63_daily_push.py 提取的公共消息构建逻辑。
支持管理员组（完整技术版）和内测组（精简卡片版）两种格式。
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# 季节性风控规则
SEASONAL_RULES = {
    "danger_months": [3, 5],
    "golden_months": [8, 10],
    "optimal_hold_days": 10,
}

# 优先挡规则
PRIORITY_TIER_RULES = {
    "score_8_golden": True,
    "vol_extreme_shrink": 0.3,
    "price_sweet_spot": (10, 20),
    "score_7_downgrade": True,
}


def build_push_message(oamv_status, signals, industry_stats, best_params, is_intraday=False):
    """构建管理员组推送消息（完整技术版，Markdown格式）

    Args:
        oamv_status: OAMV择时状态
        signals: 信号列表
        industry_stats: 行业统计列表
        best_params: 策略参数
        is_intraday: 是否盘中模式

    Returns:
        tuple: (title, desp)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][datetime.now().weekday()]

    if oamv_status:
        can_open = oamv_status["can_open_position"]
        sentiment = "偏多" if can_open else "偏空"
        sentiment_icon = "🟢" if can_open else "🔴"
    else:
        can_open = True
        sentiment = "中性"
        sentiment_icon = "⚪"

    mode_tag = "盘中实时" if is_intraday else "盘后完整"
    title = f"{sentiment_icon} 量化潜伏 {today} | {len(signals)}信号 | {sentiment} | {mode_tag}"

    lines = []

    # 一、今日概览
    lines.append(f"### 📋 今日概览")
    lines.append(f"- 📅 {today} {weekday_cn} | {mode_tag}")
    if is_intraday:
        lines.append(f"- ⏰ 数据截至 {datetime.now().strftime('%H:%M')}（盘中实时）")
    lines.append(f"- 📊 潜伏信号: **{len(signals)}只** | 市场情绪: {sentiment_icon}**{sentiment}**")

    # 二、市场环境
    lines.append("")
    lines.append(f"### 📈 市场环境")
    if oamv_status:
        oamv_label = "🟢 **牛市(允许开仓)**" if can_open else "🔴 **熊市(控制仓位)**"
        lines.append(f"- OAMV活筹: {oamv_label}")
        lines.append(f"  - 趋势: {oamv_status['trend_label']} | 强度: {oamv_status['latest_x']}")
        if oamv_status.get("last_transition"):
            lt = oamv_status["last_transition"]
            lines.append(f"  - 最近切换: {lt['date']} → {lt['to_state']}")
    else:
        lines.append(f"- OAMV活筹: 环境评估中")

    # 季节性风控提示
    current_month = datetime.now().month
    sr = SEASONAL_RULES
    if current_month in sr["danger_months"]:
        month_name = f"{current_month}月"
        if current_month == 3:
            lines.append(f"- ⛔ 季节性警告: {month_name}连续3年负收益(均-8%)，建议空仓或极轻仓!")
        elif current_month == 5:
            lines.append(f"- ⚠️ 季节性警告: {month_name}近3年2年大亏，建议严格控制仓位!")
    elif current_month in sr["golden_months"]:
        month_name = f"{current_month}月"
        lines.append(f"- ✅ 季节性利好: {month_name}连续3年正收益，信号可信度较高")

    # 三、行业风向
    if industry_stats:
        lines.append("")
        lines.append(f"### 🔥 行业风向")
        hot = [s for s in industry_stats if s["momentum"] > 0]
        cold = [s for s in industry_stats if s["momentum"] <= 0]
        rotation_in = [s for s in industry_stats if s["rotation"] == "轮入"]
        rotation_out = [s for s in industry_stats if s["rotation"] == "轮出"]
        accelerating = [s for s in industry_stats if s["rotation"] == "加速"]

        lines.append(f"- 🔥 偏热 **{len(hot)}**个 | ❄️ 偏冷 **{len(cold)}**个")
        if hot:
            hot_detail = "、".join(f"{s['name']}({s['momentum']:+.1f})" for s in hot[:5])
            lines.append(f"  - 领涨: {hot_detail}")
        if cold:
            cold_detail = "、".join(f"{s['name']}({s['momentum']:+.1f})" for s in cold[:3])
            lines.append(f"  - 领跌: {cold_detail}")

        rotation_parts = []
        if rotation_in:
            rotation_parts.append(f"🔄轮入:{'、'.join(s['name'] for s in rotation_in[:3])}")
        if rotation_out:
            rotation_parts.append(f"⏏️轮出:{'、'.join(s['name'] for s in rotation_out[:3])}")
        if accelerating:
            rotation_parts.append(f"🚀加速:{'、'.join(s['name'] for s in accelerating[:3])}")
        if rotation_parts:
            lines.append(f"- {' | '.join(rotation_parts)}")

    # 四、潜伏信号
    lines.append("")
    lines.append(f"### 🎯 潜伏信号")

    if not can_open and signals:
        lines.append(f"> ⚠️ 当前环境偏弱，以下标的仅供跟踪观察")
        lines.append("")

    if not signals:
        if can_open:
            lines.append(f"> 📭 今日无符合条件的潜伏标的")
            lines.append(f"> 市场未出现缩量超卖+需求保护的信号")
            lines.append(f"> 耐心等待，不追涨是纪律")
        else:
            lines.append(f"> 🛡️ 环境偏弱，暂无值得关注的标的")
            lines.append(f"> OAMV显示资金萎缩，建议空仓观望")
        lines.append("")
    else:
        # 信号分级
        priority_signals, normal_signals = _classify_signals(signals)

        if priority_signals:
            lines.append(f"#### 🔴 优先考虑挡 ({len(priority_signals)}只)")
            lines.append(f"> 条件: 8分黄金信号 | 量比<0.3极度缩量 | 10-20元最佳区间")
            lines.append("")

        for i, s in enumerate(priority_signals, 1):
            _append_signal_detail(lines, s, i, "🔴", is_priority=True)

        if normal_signals:
            lines.append(f"#### ⚪ 普通挡 ({len(normal_signals)}只)")
            lines.append(f"> 条件: 评分≥5但未达优先挡标准")
            lines.append("")

        for i, s in enumerate(normal_signals, 1):
            _append_signal_detail(lines, s, i, "⚪", is_priority=False)

    # 五、策略说明
    lines.append(f"### 📖 策略说明")
    lines.append(f"- 核心逻辑: SOS需求大阳线 → 情绪冰点(J超卖+缩量+小实体) → 潜伏买入")
    lines.append(f"- 评分体系: E1情绪冰点(0-2) + E2量能枯竭(0-2) + E3盘面形态(0-2) + E4均线结构(0-2)")
    lines.append(f"- 风控机制: OAMV择时(牛/熊) → 行业动量过滤 → 动态评分门槛 → 4级退出")
    lines.append(f"- 持仓周期: 最优10天 | 仓位管理: 单只≤15%(优先)/≤10%(普通) | 总敞口≤50%")

    # 六、免责声明
    lines.append("")
    lines.append(f"> ⚠️ 以上内容为量化系统自动输出，不构成任何投资建议")
    lines.append(f"> ⚠️ 股市有风险，投资需谨慎，据此操作风险自担")
    lines.append(f"> ⚡ 量化潜伏系统 V6.4")

    desp = "\n".join(lines)
    return title, desp


def build_beta_push_message(oamv_status, signals, industry_stats, best_params, is_intraday=False):
    """构建内测组推送消息（精简卡片式）

    Args:
        同build_push_message

    Returns:
        tuple: (title, desp)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][datetime.now().weekday()]

    if oamv_status:
        can_open = oamv_status["can_open_position"]
        sentiment = "偏多" if can_open else "偏空"
        sentiment_icon = "🟢" if can_open else "🔴"
    else:
        can_open = True
        sentiment = "中性"
        sentiment_icon = "⚪"

    mode_tag = "盘中" if is_intraday else "盘后"
    title = f"{sentiment_icon} 量化潜伏 {today} | {len(signals)}只 | {sentiment} | {mode_tag}"

    lines = []

    lines.append(f"📊 {today} {weekday_cn} | {mode_tag}版")
    if is_intraday:
        lines.append(f"⏰ 数据截至 {datetime.now().strftime('%H:%M')}")
    lines.append(f"📈 市场情绪: {sentiment_icon}{sentiment}")

    # 季节性提示
    current_month = datetime.now().month
    sr = SEASONAL_RULES
    if current_month in sr["danger_months"]:
        lines.append(f"⚠️ {current_month}月历史表现偏弱，建议控制仓位")
    elif current_month in sr["golden_months"]:
        lines.append(f"✅ {current_month}月历史表现较好，信号可信度较高")

    # 行业风向（精简）
    if industry_stats:
        hot = [s for s in industry_stats if s["momentum"] > 0]
        rotation_in = [s for s in industry_stats if s["rotation"] == "轮入"]
        if hot:
            lines.append(f"🔥 热门行业: {'、'.join(s['name'] for s in hot[:5])}")
        if rotation_in:
            lines.append(f"🔄 轮入: {'、'.join(s['name'] for s in rotation_in[:3])}")

    # 信号分级
    if not signals:
        lines.append("")
        if can_open:
            lines.append("📭 今日暂无符合条件的标的，耐心等待")
        else:
            lines.append("🛡️ 市场偏弱，建议观望")
    else:
        priority_signals, normal_signals = _classify_signals(signals)

        if priority_signals:
            lines.append("")
            lines.append(f"━━━ 重点推荐 ({len(priority_signals)}只) ━━━")
            for i, s in enumerate(priority_signals, 1):
                eq = s.get('entry_quality_score', 0)
                vr = s.get('vol_ratio', 1.0)
                j = s.get('J', 99)
                tags = []
                if eq >= 8:
                    tags.append("⭐8分")
                if vr < 0.3:
                    tags.append("💧极度缩量")
                if 10 <= s['price'] <= 20:
                    tags.append("💰最佳区间")
                pos = "仓位15%" if (vr < 0.3 or eq >= 8) else "仓位10%"
                change_sign = "+" if s['change_pct'] >= 0 else ""
                lines.append(f"{i}. {s['name']}({s['code']}) {s['price']:.2f} {change_sign}{s['change_pct']:.2f}%")
                if tags:
                    lines.append(f"   {' '.join(tags)}")
                lines.append(f"   评分:{eq}/8 | J:{j:.0f} | 量比:{vr:.2f} | {pos}")
                lines.append(f"   止损:{s['hard_stop']:.2f} | 持仓10天")
                lines.append("")

        if normal_signals:
            lines.append(f"━━━ 关注标的 ({len(normal_signals)}只) ━━━")
            for i, s in enumerate(normal_signals, 1):
                eq = s.get('entry_quality_score', 0)
                j = s.get('J', 99)
                vr = s.get('vol_ratio', 1.0)
                change_sign = "+" if s['change_pct'] >= 0 else ""
                lines.append(f"{i}. {s['name']}({s['code']}) {s['price']:.2f} {change_sign}{s['change_pct']:.2f}%")
                lines.append(f"   评分:{eq}/8 | J:{j:.0f} | 量比:{vr:.2f} | 仓位5%")
                lines.append(f"   止损:{s['hard_stop']:.2f} | 持仓10天")
                lines.append("")

    # 仓位管理
    lines.append(f"━━━ 仓位管理 ━━━")
    lines.append(f"单只≤15%(推荐)/≤10%(关注) | 总敞口≤50%")
    lines.append(f"持仓周期: 10个交易日")

    lines.append("")
    lines.append(f"⚠️ 量化系统自动输出，不构成投资建议")
    lines.append(f"⚡ 量化潜伏 Pro | V6.4")

    desp = "\n".join(lines)
    return title, desp


def _classify_signals(signals):
    """信号分级：优先考虑挡 vs 普通挡

    Returns:
        tuple: (priority_signals, normal_signals)
    """
    ptr = PRIORITY_TIER_RULES
    priority_signals = []
    normal_signals = []

    for s in signals:
        eq = s.get('entry_quality_score', 0)
        vr = s.get('vol_ratio', 1.0)
        price = s.get('price', 0)

        is_priority = False
        if ptr["score_8_golden"] and eq >= 8:
            is_priority = True
        if vr < ptr["vol_extreme_shrink"]:
            is_priority = True
        p_lo, p_hi = ptr["price_sweet_spot"]
        if p_lo <= price <= p_hi:
            is_priority = True
        if ptr["score_7_downgrade"] and eq == 7:
            is_priority = False

        if is_priority:
            priority_signals.append(s)
        else:
            normal_signals.append(s)

    return priority_signals, normal_signals


def _append_signal_detail(lines, s, index, icon, is_priority=True):
    """追加单个信号的详细信息到lines列表"""
    eq = s.get('entry_quality_score', 0)
    vr = s.get('vol_ratio', 1.0)
    j = s.get('J', 99)

    # 优先挡标签
    if is_priority:
        priority_tags = []
        if eq >= 8:
            priority_tags.append("⭐8分黄金信号(胜率50%)")
        if vr < 0.3:
            priority_tags.append("💧极度缩量(10日均收+3.8%)")
        if 10 <= s['price'] <= 20:
            priority_tags.append("💰10-20元最佳区间(20日均收+4.1%)")

    # 评分星级
    if eq >= 8:
        score_star = "★★★"
    elif eq >= 7:
        score_star = "★★★"
    elif eq >= 5:
        score_star = "★★☆"
    else:
        score_star = "★☆☆"

    lines.append(f"```")
    lines.append(f"{icon} {index}. {s['name']}({s['code']}) | {s['industry']}")
    if is_priority and priority_tags:
        lines.append(f"   {' | '.join(priority_tags)}")
    lines.append(f"")

    # 评分
    lines.append(f"📐 入场评分: {eq}/8 {score_star}")
    score_parts = []
    ej_score = s.get('eq_j_score', None)
    ev_score = s.get('eq_vol_score', None)
    ec_score = s.get('eq_candle_score', None)
    em_score = s.get('eq_ma_score', None)
    if all(v is not None for v in [ej_score, ev_score, ec_score, em_score]):
        score_parts.append(f"E1情绪{ej_score}")
        score_parts.append(f"E2量能{ev_score}")
        score_parts.append(f"E3形态{ec_score}")
        score_parts.append(f"E4均线{em_score}")
    else:
        e1 = 2 if j < 3 else 1 if j < 10 else 0
        e2 = 2 if vr < 0.3 else 1 if vr < 0.5 else 0
        score_parts.append(f"E1情绪≈{e1}")
        score_parts.append(f"E2量能≈{e2}")
        score_parts.append(f"E3形态≈?")
        score_parts.append(f"E4均线≈?")
    lines.append(f"   拆解: {' | '.join(score_parts)}")

    # 价格+涨跌
    change_sign = "+" if s['change_pct'] >= 0 else ""
    change_icon = "🔺" if s['change_pct'] >= 0 else "🔻"
    lines.append(f"")
    lines.append(f"{change_icon} 价格:{s['price']:.2f} | 涨跌:{change_sign}{s['change_pct']:.2f}%")

    # 均线结构
    ma_rel = "白>黄(多头)" if s['white_line'] > s['yellow_line'] else "白<黄(空头)" if s['white_line'] < s['yellow_line'] else "白=黄(收敛)"
    ma_dist = abs(s['white_line'] - s['yellow_line']) / s['atr14'] if s['atr14'] > 0 else 0
    lines.append(f"📏 白线:{s['white_line']:.2f} | 黄线:{s['yellow_line']:.2f} | {ma_rel}")
    lines.append(f"   线距:{ma_dist:.1f}ATR | ATR:{s['atr14']:.2f}")

    # 核心指标
    j_label = "极度超卖" if j < 3 else "深度超卖" if j < 5 else "超卖" if j < 10 else "偏低" if j < 20 else "中性"
    vol_label = "极度缩量" if vr < 0.3 else "明显缩量" if vr < 0.5 else "缩量" if vr < 0.8 else "放量"
    lines.append(f"📉 J值:{j:.1f}({j_label}) | 量比:{vr:.2f}({vol_label})")

    # SOS锚定
    if s.get('sos_dates'):
        lines.append(f"📍 SOS锚定日: {', '.join(s['sos_dates'])}")

    # 威科夫解读
    analysis = s.get('analysis', {})
    wyckoff = analysis.get('wyckoff', [])
    if wyckoff:
        lines.append(f"🔍 威科夫: {'; '.join(wyckoff)}")

    # VPA量价
    vpa = analysis.get('vpa', [])
    if vpa:
        lines.append(f"📊 VPA量价: {'; '.join(vpa)}")

    # 蜡烛图
    candle = analysis.get('candle', [])
    if candle:
        lines.append(f"🕯️ 蜡烛图: {'; '.join(candle)}")

    # 支撑阻力
    support = analysis.get('support', s['yellow_line'])
    resistance = analysis.get('resistance', s['yellow_line'])
    lines.append(f"🎯 支撑:{support} | 阻力:{resistance}")

    # 交易参考
    lines.append(f"")
    lines.append(f"💰 交易参考")
    lines.append(f"   买入: {s['price']:.2f}(T+1开盘价)")
    lines.append(f"   硬止损: {s['hard_stop']:.2f}(-5%) | 吊灯线: {s['chandelier_init']:.2f}")
    risk = s['price'] - s['hard_stop']
    reward = resistance - s['price']
    rr_ratio = reward / risk if risk > 0 else 0
    rr_label = "优" if rr_ratio >= 3 else "良" if rr_ratio >= 2 else "一般" if rr_ratio >= 1 else "差"
    lines.append(f"   风险收益比: 1:{rr_ratio:.1f}({rr_label}) | 亏损空间:{risk / s['price'] * 100:.1f}%")

    if is_priority:
        pos_advice = "极度缩量可加仓至15%" if vr < 0.3 else "8分黄金可加仓至15%" if eq >= 8 else "标准仓10%"
    else:
        pos_advice = "减仓5%"
    lines.append(f"   ⏱️ 持仓: 10个交易日 | 📦 仓位: {pos_advice} | 总敞口≤50%")
    lines.append(f"```")
    lines.append("")
