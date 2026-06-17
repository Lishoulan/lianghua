"""测试 scanner 模块的信号处理和动态评分过滤

覆盖场景：
  - _extract_signal_info: 无信号时返回None
  - _extract_signal_info: 有信号时返回包含预期键的字典
  - apply_dynamic_score_filter: 基于OAMV状态正确过滤信号
  - apply_dynamic_score_filter: J硬上限过滤
  - apply_dynamic_score_filter: 牛市规则
  - apply_dynamic_score_filter: 熊市规则
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from classic_ta.common.scanner import _extract_signal_info, apply_dynamic_score_filter
from classic_ta.v60_ambush_model import IndicatorCalcBase, DEFAULT_PARAMS


# ──────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def signal_df_with_indicators():
    """生成一个已计算指标且有ambush_signal=True的DataFrame"""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    # 上升趋势
    returns = np.random.normal(0.002, 0.02, n)
    close = 20.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.012, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.012, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.randint(500_000, 3_000_000, n).astype(float)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

    df = IndicatorCalcBase(df)

    # 手动添加信号相关列（这些列由Detect_AmbushSignal添加，这里手动设置）
    df["ambush_signal"] = False
    df["tag_sos_anchor"] = False
    df["tag_no_supply"] = False
    df["tag_test"] = False

    # 设置最后一行为信号日
    df.iloc[-1, df.columns.get_loc("ambush_signal")] = True
    # 确保white_line和yellow_line有效
    assert not pd.isna(df.iloc[-1]["white_line"])
    assert not pd.isna(df.iloc[-1]["yellow_line"])

    return df


@pytest.fixture
def no_signal_df_with_indicators():
    """生成一个已计算指标但无信号的DataFrame"""
    np.random.seed(100)
    n = 200
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    returns = np.random.normal(-0.002, 0.015, n)
    close = 30.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.randint(500_000, 3_000_000, n).astype(float)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

    df = IndicatorCalcBase(df)
    # 手动添加信号相关列，确保无信号
    df["ambush_signal"] = False
    return df


@pytest.fixture
def sample_signals():
    """生成合成信号列表用于动态评分过滤测试"""
    return [
        {"code": "600001.SH", "name": "测试1", "J": 3, "entry_quality_score": 6, "vol_ratio": 0.4},
        {"code": "600002.SH", "name": "测试2", "J": 4, "entry_quality_score": 5, "vol_ratio": 0.5},
        {"code": "600003.SH", "name": "测试3", "J": 7, "entry_quality_score": 7, "vol_ratio": 0.3},  # J超限
        {"code": "600004.SH", "name": "测试4", "J": 2, "entry_quality_score": 3, "vol_ratio": 0.55},
        {"code": "600005.SH", "name": "测试5", "J": 1, "entry_quality_score": 2, "vol_ratio": 0.2},  # 低分
        {"code": "600006.SH", "name": "测试6", "J": 4, "entry_quality_score": 4, "vol_ratio": 0.55},
    ]


# ──────────────────────────────────────────────────────────
#  _extract_signal_info 测试
# ──────────────────────────────────────────────────────────

class TestExtractSignalInfo:
    """_extract_signal_info 信号提取测试"""

    def test_returns_none_when_no_signal(self, no_signal_df_with_indicators):
        """当ambush_signal为False时，应返回None"""
        result = _extract_signal_info(
            "600000.SH", "测试", "银行",
            no_signal_df_with_indicators, DEFAULT_PARAMS
        )
        assert result is None

    def test_returns_dict_when_signal(self, signal_df_with_indicators):
        """当ambush_signal为True时，应返回包含预期键的字典"""
        with patch("classic_ta.common.signal_analyzer.analyze_signal_detail", return_value={
            "wyckoff": ["测试"], "vpa": ["测试"], "candle": ["测试"],
            "support": 10.0, "resistance": 12.0,
        }):
            result = _extract_signal_info(
                "600000.SH", "测试", "银行",
                signal_df_with_indicators, DEFAULT_PARAMS
            )
        assert result is not None
        assert isinstance(result, dict)

    def test_signal_info_has_expected_keys(self, signal_df_with_indicators):
        """返回的信号信息应包含所有预期键"""
        with patch("classic_ta.common.signal_analyzer.analyze_signal_detail", return_value={
            "wyckoff": ["测试"], "vpa": ["测试"], "candle": ["测试"],
            "support": 10.0, "resistance": 12.0,
        }):
            result = _extract_signal_info(
                "600000.SH", "测试", "银行",
                signal_df_with_indicators, DEFAULT_PARAMS
            )
        expected_keys = [
            "code", "name", "industry", "price", "change_pct",
            "white_line", "yellow_line", "J", "atr14", "vol_ratio",
            "sos_dates", "analysis", "signal_date", "entry_quality_score",
            "hard_stop", "chandelier_init",
        ]
        for key in expected_keys:
            assert key in result, f"信号信息缺少键: {key}"

    def test_signal_info_code_matches(self, signal_df_with_indicators):
        """返回的信号code应与输入一致"""
        with patch("classic_ta.common.signal_analyzer.analyze_signal_detail", return_value={
            "wyckoff": ["测试"], "vpa": ["测试"], "candle": ["测试"],
            "support": 10.0, "resistance": 12.0,
        }):
            result = _extract_signal_info(
                "600519.SH", "贵州茅台", "白酒",
                signal_df_with_indicators, DEFAULT_PARAMS
            )
        assert result["code"] == "600519.SH"
        assert result["name"] == "贵州茅台"
        assert result["industry"] == "白酒"

    def test_signal_info_price_is_positive(self, signal_df_with_indicators):
        """返回的信号price应为正数"""
        with patch("classic_ta.common.signal_analyzer.analyze_signal_detail", return_value={
            "wyckoff": ["测试"], "vpa": ["测试"], "candle": ["测试"],
            "support": 10.0, "resistance": 12.0,
        }):
            result = _extract_signal_info(
                "600000.SH", "测试", "银行",
                signal_df_with_indicators, DEFAULT_PARAMS
            )
        assert result["price"] > 0

    def test_returns_none_when_nan_lines(self):
        """当white_line或yellow_line为NaN时，应返回None"""
        n = 10
        dates = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame({
            "Open": [10.0] * n, "High": [10.5] * n, "Low": [9.5] * n,
            "Close": [10.2] * n, "Volume": [500000] * n,
            "white_line": [np.nan] * n, "yellow_line": [np.nan] * n,
            "ambush_signal": [True] * n,
        }, index=dates)
        result = _extract_signal_info("600000.SH", "测试", "银行", df, DEFAULT_PARAMS)
        assert result is None


# ──────────────────────────────────────────────────────────
#  apply_dynamic_score_filter 测试
# ──────────────────────────────────────────────────────────

class TestApplyDynamicScoreFilter:
    """动态评分过滤测试"""

    def test_empty_signals_returns_empty(self):
        """空信号列表应返回空列表"""
        result = apply_dynamic_score_filter([], None, {})
        assert result == []

    def test_j_hard_cap_filters_high_j(self, sample_signals):
        """J值硬上限应过滤掉J值过高的信号"""
        dsp = {"j_hard_cap": 5, "bull_min_score": 5, "bear_min_score": 6}
        # 牛市模式
        oamv_status = {"can_open_position": True}
        result = apply_dynamic_score_filter(sample_signals, oamv_status, dsp)
        # J=7的信号应被过滤
        for s in result:
            assert s["J"] < 5, f"J值{ s['J'] }超过硬上限5"

    def test_bull_market_rules(self, sample_signals):
        """牛市规则：评分>=bull_min_score的信号通过"""
        dsp = {"j_hard_cap": 5, "bull_min_score": 5, "bear_min_score": 6}
        oamv_status = {"can_open_position": True}
        result = apply_dynamic_score_filter(sample_signals, oamv_status, dsp)
        # 通过的信号：score=6(J=3), score=5(J=4)
        codes = [s["code"] for s in result]
        assert "600001.SH" in codes  # score=6, J=3
        assert "600002.SH" in codes  # score=5, J=4

    def test_bull_market_score4_special_rule(self):
        """牛市规则：评分=4时，J和vol_ratio需满足额外条件"""
        signals = [
            {"code": "A", "J": 3, "entry_quality_score": 4, "vol_ratio": 0.50},  # 通过
            {"code": "B", "J": 6, "entry_quality_score": 4, "vol_ratio": 0.50},  # J超限
            {"code": "C", "J": 3, "entry_quality_score": 4, "vol_ratio": 0.70},  # vol_ratio超限
        ]
        dsp = {
            "j_hard_cap": 5, "bull_min_score": 5,
            "bull_score4_j_max": 5, "bull_score4_vol_ratio_max": 0.60,
            "bear_min_score": 6,
        }
        oamv_status = {"can_open_position": True}
        result = apply_dynamic_score_filter(signals, oamv_status, dsp)
        codes = [s["code"] for s in result]
        assert "A" in codes
        assert "B" not in codes  # J=6 > j_hard_cap=5
        assert "C" not in codes  # vol_ratio=0.70 > 0.60

    def test_bear_market_rules(self, sample_signals):
        """熊市规则：只有评分>=bear_min_score的信号通过"""
        dsp = {"j_hard_cap": 5, "bull_min_score": 5, "bear_min_score": 6}
        oamv_status = {"can_open_position": False}
        result = apply_dynamic_score_filter(sample_signals, oamv_status, dsp)
        # 熊市只有score>=6的信号通过（且J<5）
        for s in result:
            assert s["entry_quality_score"] >= 6, f"熊市中评分{s['entry_quality_score']}<6"
            assert s["J"] < 5, f"J值{s['J']}超过硬上限"

    def test_bear_market_stricter_than_bull(self, sample_signals):
        """熊市规则应比牛市更严格"""
        dsp = {"j_hard_cap": 5, "bull_min_score": 5, "bear_min_score": 6}
        bull_result = apply_dynamic_score_filter(
            sample_signals, {"can_open_position": True}, dsp
        )
        bear_result = apply_dynamic_score_filter(
            sample_signals, {"can_open_position": False}, dsp
        )
        assert len(bear_result) <= len(bull_result), "熊市过滤应比牛市更严格"

    def test_j_hard_cap_default_value(self):
        """默认J硬上限为5"""
        signals = [
            {"code": "A", "J": 4, "entry_quality_score": 6, "vol_ratio": 0.4},
            {"code": "B", "J": 5, "entry_quality_score": 6, "vol_ratio": 0.4},  # J=5 不<5
        ]
        dsp = {"j_hard_cap": 5, "bull_min_score": 5, "bear_min_score": 6}
        oamv_status = {"can_open_position": True}
        result = apply_dynamic_score_filter(signals, oamv_status, dsp)
        codes = [s["code"] for s in result]
        assert "A" in codes
        assert "B" not in codes  # J=5 >= j_hard_cap=5

    def test_none_oamv_treated_as_bear(self, sample_signals):
        """OAMV状态为None时应按熊市处理"""
        dsp = {"j_hard_cap": 5, "bull_min_score": 5, "bear_min_score": 6}
        result_none = apply_dynamic_score_filter(sample_signals, None, dsp)
        result_bear = apply_dynamic_score_filter(
            sample_signals, {"can_open_position": False}, dsp
        )
        codes_none = [s["code"] for s in result_none]
        codes_bear = [s["code"] for s in result_bear]
        assert codes_none == codes_bear, "None OAMV应与熊市行为一致"
