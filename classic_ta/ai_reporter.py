import os

import pandas as pd
from litellm import completion


def build_feature_summary(df, stock_name, stock_code):
    row = df.iloc[-1]
    prev = df.iloc[-2]

    candlestick_names = {
        "is_hammer": "锤子线",
        "is_bullish_engulfing": "看涨吞没",
        "is_bearish_engulfing": "看跌吞没",
        "is_morning_star": "晨星",
        "is_shooting_star": "射击之星",
    }
    candlestick_signals = [
        label for col, label in candlestick_names.items() if row.get(col, False)
    ]
    candlestick_text = "、".join(candlestick_signals) if candlestick_signals else "无特殊形态"

    vp_names = {
        "vol_surge_stagnant": "放量滞涨",
        "vol_surge_bottom_rejection": "放量底部拒绝",
        "shrink_volume_pullback": "缩量回踩",
    }
    vp_signals = [label for col, label in vp_names.items() if row.get(col, False)]
    vp_text = "、".join(vp_signals) if vp_signals else "无量价异常"

    golden_cross = (row["white_line"] > row["yellow_line"]) and (prev["white_line"] <= prev["yellow_line"])
    cross_text = "是（金叉）" if golden_cross else "否"
    line_position = "白线在黄线上方" if row["white_line"] > row["yellow_line"] else "白线在黄线下方"

    lines = [
        f"股票：{stock_name}（{stock_code}）",
        f"当前价格：{row['Close']:.2f}",
        f"当日涨跌幅：{row['daily_return']:.2f}%",
        f"威科夫阶段：{row['wyckoff_phase']}",
        f"支撑位：{row['support_level']:.2f}",
        f"阻力位：{row['resistance_level']:.2f}",
        f"蜡烛图形态：{candlestick_text}",
        f"量价信号：{vp_text}",
        f"白线：{row['white_line']:.2f}，黄线：{row['yellow_line']:.2f}，{line_position}",
        f"今日是否金叉：{cross_text}",
        f"KDJ-J：{row['J']:.1f}",
    ]
    return "\n".join(lines)


def generate_analysis_report(feature_text, system_prompt=None):
    if system_prompt is None:
        system_prompt = (
            "你现在是一位顶级的A股技术分析大师，你精通《威科夫操盘法》、《量价分析VPA》和《日本蜡烛图》。\n"
            "我每天会给你发送某只股票当天的量化数据和形态特征。请你像一个经验丰富的操盘手一样，进行综合解盘。\n\n"
            "分析框架必须包含：\n"
            "1. 大局观（威科夫视角）：判断目前处于四大阶段的哪一段，有没有主力建仓或出货的迹象。\n"
            "2. 主力意图（量价视角）：根据今天的量价配合，分析主力是在洗盘、拉升还是派发。\n"
            "3. 微观买卖点（蜡烛图视角）：今天的K线是否构成了进场或逃顶信号。\n"
            "4. 操作建议：给出明确的胜率预估和防守策略（止损位）。"
        )

    try:
        response = completion(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": feature_text},
            ],
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI分析生成失败: {str(e)}"
