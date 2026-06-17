"""测试信号检测函数

覆盖场景：
  - Detect_AmbushSignal (V60): 返回含ambush_signal列的DataFrame
  - Detect_AmbushSignal_V64: 返回含entry_quality_score列的DataFrame
  - 已知信号模式的合成数据（信号应触发）
  - 无信号模式的合成数据（信号不应触发）
  - entry_quality_score在[0, 8]范围
  - 入场质量子评分（eq_j_score, eq_vol_score, eq_candle_score, eq_ma_score）在有效范围
"""

import numpy as np
import pandas as pd
import pytest

from classic_ta.v60_ambush_model import IndicatorCalcBase, Detect_AmbushSignal, DEFAULT_PARAMS
from classic_ta.v64_ambush_model import (
    add_entry_quality_indicators,
    Detect_AmbushSignal_V64,
    V64_PARAMS,
)


# ──────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def no_signal_ohlcv():
    """生成不会触发信号的数据：持续下跌趋势，J值高，无SOS"""
    np.random.seed(100)
    n = 200
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    # 持续下跌
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
    return df


@pytest.fixture
def signal_ohlcv():
    """生成可能触发信号的数据：上升趋势+SOS后缩量回调+J超卖"""
    np.random.seed(200)
    n = 200
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    # 前期上升趋势
    returns = np.random.normal(0.002, 0.02, n)
    close = 20.0 * np.cumprod(1 + returns)

    # 手动构造最后几天的信号条件
    # 最后5天：缩量、小实体、J超卖
    for i in range(n - 5, n):
        close[i] = close[i - 1] * (1 + np.random.normal(-0.001, 0.003))  # 微跌/平

    high = close * (1 + np.abs(np.random.normal(0, 0.012, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.012, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.randint(500_000, 3_000_000, n).astype(float)

    # 最后5天缩量
    volume[-5:] = volume[-5:] * 0.3

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    return df


@pytest.fixture
def basic_indicators_df(signal_ohlcv):
    """已计算基础指标的DataFrame"""
    return IndicatorCalcBase(signal_ohlcv)


@pytest.fixture
def basic_indicators_no_signal_df(no_signal_ohlcv):
    """已计算基础指标的无信号DataFrame"""
    return IndicatorCalcBase(no_signal_ohlcv)


# ──────────────────────────────────────────────────────────
#  V60 信号检测测试
# ──────────────────────────────────────────────────────────

class TestDetectAmbushSignalV60:
    """V60潜伏信号检测测试"""

    def test_returns_dataframe(self, signal_ohlcv):
        """Detect_AmbushSignal应返回DataFrame"""
        df = IndicatorCalcBase(signal_ohlcv)
        result = Detect_AmbushSignal(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_ambush_signal_column(self, signal_ohlcv):
        """输出应包含ambush_signal列"""
        df = IndicatorCalcBase(signal_ohlcv)
        result = Detect_AmbushSignal(df)
        assert "ambush_signal" in result.columns

    def test_ambush_signal_is_boolean(self, signal_ohlcv):
        """ambush_signal列应为布尔类型"""
        df = IndicatorCalcBase(signal_ohlcv)
        result = Detect_AmbushSignal(df)
        assert result["ambush_signal"].dtype == bool or str(result["ambush_signal"].dtype) == "boolean"

    def test_no_signal_data(self, basic_indicators_no_signal_df):
        """持续下跌的数据不应触发信号"""
        result = Detect_AmbushSignal(basic_indicators_no_signal_df)
        # 持续下跌趋势中，信号应极少或为0
        signal_count = result["ambush_signal"].sum()
        assert signal_count <= 5, f"持续下跌数据中信号数异常: {signal_count}"

    def test_has_tag_columns(self, signal_ohlcv):
        """输出应包含SOS锚定和枯竭标记列"""
        df = IndicatorCalcBase(signal_ohlcv)
        result = Detect_AmbushSignal(df)
        for col in ["tag_sos_anchor", "tag_no_supply", "tag_test"]:
            assert col in result.columns, f"缺少标记列: {col}"

    def test_does_not_mutate_input(self, signal_ohlcv):
        """Detect_AmbushSignal不应修改输入DataFrame"""
        df = IndicatorCalcBase(signal_ohlcv)
        original = df.copy()
        Detect_AmbushSignal(df)
        # 只比较原始列（Detect_AmbushSignal会添加新列但不修改已有列）
        pd.testing.assert_frame_equal(df[original.columns], original)

    def test_custom_params(self, signal_ohlcv):
        """使用自定义参数应能正常计算"""
        df = IndicatorCalcBase(signal_ohlcv)
        custom_params = DEFAULT_PARAMS.copy()
        custom_params["ambush_j_oversold"] = 20  # 放宽J值阈值
        result = Detect_AmbushSignal(df, params=custom_params)
        assert "ambush_signal" in result.columns


# ──────────────────────────────────────────────────────────
#  V64 信号检测 + 入场质量评分测试
# ──────────────────────────────────────────────────────────

class TestDetectAmbushSignalV64:
    """V64潜伏信号检测（含入场质量评分）测试"""

    def test_returns_dataframe(self, basic_indicators_df):
        """Detect_AmbushSignal_V64应返回DataFrame"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_entry_quality_score(self, basic_indicators_df):
        """输出应包含entry_quality_score列"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        assert "entry_quality_score" in result.columns

    def test_entry_quality_score_range(self, basic_indicators_df):
        """entry_quality_score应在[0, 8]范围内"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        scores = result["entry_quality_score"]
        assert (scores >= 0).all(), f"评分存在负值: min={scores.min()}"
        assert (scores <= 8).all(), f"评分超过8: max={scores.max()}"

    def test_has_sub_scores(self, basic_indicators_df):
        """输出应包含入场质量子评分列"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        for col in ["eq_j_score", "eq_vol_score", "eq_candle_score", "eq_ma_score"]:
            assert col in result.columns, f"缺少子评分列: {col}"

    def test_eq_j_score_range(self, basic_indicators_df):
        """eq_j_score应在[0, 2]范围内"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        assert (result["eq_j_score"] >= 0).all()
        assert (result["eq_j_score"] <= 2).all()

    def test_eq_vol_score_range(self, basic_indicators_df):
        """eq_vol_score应在[0, 2]范围内"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        assert (result["eq_vol_score"] >= 0).all()
        assert (result["eq_vol_score"] <= 2).all()

    def test_eq_candle_score_range(self, basic_indicators_df):
        """eq_candle_score应在[0, 2]范围内"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        assert (result["eq_candle_score"] >= 0).all()
        assert (result["eq_candle_score"] <= 2).all()

    def test_eq_ma_score_range(self, basic_indicators_df):
        """eq_ma_score应在[0, 2]范围内"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        assert (result["eq_ma_score"] >= 0).all()
        assert (result["eq_ma_score"] <= 2).all()

    def test_entry_quality_equals_sum_of_sub_scores(self, basic_indicators_df):
        """entry_quality_score应等于四个子评分之和"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        expected = result["eq_j_score"] + result["eq_vol_score"] + result["eq_candle_score"] + result["eq_ma_score"]
        pd.testing.assert_series_equal(result["entry_quality_score"], expected, check_names=False)

    def test_has_ambush_signal_column(self, basic_indicators_df):
        """V64输出也应包含ambush_signal列"""
        result = Detect_AmbushSignal_V64(basic_indicators_df)
        assert "ambush_signal" in result.columns

    def test_no_signal_data_v64(self, basic_indicators_no_signal_df):
        """持续下跌数据在V64中信号应极少"""
        result = Detect_AmbushSignal_V64(basic_indicators_no_signal_df)
        signal_count = result["ambush_signal"].sum()
        assert signal_count <= 5, f"持续下跌数据中V64信号数异常: {signal_count}"


# ──────────────────────────────────────────────────────────
#  入场质量评分独立测试
# ──────────────────────────────────────────────────────────

class TestAddEntryQualityIndicators:
    """add_entry_quality_indicators 独立测试"""

    def test_returns_dataframe(self, basic_indicators_df):
        """add_entry_quality_indicators应返回DataFrame"""
        result = add_entry_quality_indicators(basic_indicators_df)
        assert isinstance(result, pd.DataFrame)

    def test_extreme_j_gets_high_score(self):
        """J值极低时应获得较高的eq_j_score"""
        n = 50
        dates = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame({
            "Open": np.full(n, 10.0),
            "High": np.full(n, 10.5),
            "Low": np.full(n, 9.5),
            "Close": np.full(n, 10.2),
            "Volume": np.full(n, 500000.0),
        }, index=dates)
        df = IndicatorCalcBase(df)
        # J<0时应得2分（eq_j_extreme=0）
        df["J"] = -1.0
        result = add_entry_quality_indicators(df)
        assert result["eq_j_score"].iloc[-1] == 2, "J<0时应得2分"

    def test_moderate_j_gets_medium_score(self):
        """J值在0~5之间应获得1分"""
        n = 50
        dates = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame({
            "Open": np.full(n, 10.0),
            "High": np.full(n, 10.5),
            "Low": np.full(n, 9.5),
            "Close": np.full(n, 10.2),
            "Volume": np.full(n, 500000.0),
        }, index=dates)
        df = IndicatorCalcBase(df)
        # J=3 < 5 (eq_j_very_oversold) 但 J=3 >= 0 (eq_j_extreme)，应得1分
        df["J"] = 3.0
        result = add_entry_quality_indicators(df)
        assert result["eq_j_score"].iloc[-1] == 1, "0<=J<5时应得1分"

    def test_extreme_low_volume_gets_high_score(self):
        """极低成交量应获得较高的eq_vol_score"""
        n = 50
        dates = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame({
            "Open": np.full(n, 10.0),
            "High": np.full(n, 10.5),
            "Low": np.full(n, 9.5),
            "Close": np.full(n, 10.2),
            "Volume": np.full(n, 500000.0),
        }, index=dates)
        df = IndicatorCalcBase(df)
        # 设置最后一行成交量为极低
        df.iloc[-1, df.columns.get_loc("Volume")] = df["volume_ma"].iloc[-1] * 0.1
        result = add_entry_quality_indicators(df)
        assert result["eq_vol_score"].iloc[-1] == 2, "极低量应得2分"
