"""测试 IndicatorCalcBase 指标计算工厂函数

覆盖场景：
  - 基本指标计算：输入合法OHLCV数据，输出全部技术指标列
  - 输出列完整性：white_line, yellow_line, atr14, volume_ma, K, D, J
  - 指标值合理性：近期行无NaN、ATR非负、J值在[0,100]范围
  - 边界情况：极短DataFrame、所有价格相同的DataFrame
"""

import numpy as np
import pandas as pd
import pytest

from classic_ta.v60_ambush_model import IndicatorCalcBase


# ──────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """生成200天的合成OHLCV数据，模拟一只正常波动的股票"""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    base_price = 20.0
    # 随机游走 + 微弱上升趋势
    returns = np.random.normal(0.001, 0.025, n)
    close = base_price * np.cumprod(1 + returns)
    # 构造 OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.015, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.015, n)))
    open_ = close * (1 + np.random.normal(0, 0.008, n))
    volume = np.random.randint(100_000, 5_000_000, n).astype(float)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    # 确保 High >= max(Open,Close) 且 Low <= min(Open,Close)
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    return df


@pytest.fixture
def short_ohlcv():
    """生成5天的极短OHLCV数据"""
    dates = pd.bdate_range(start="2024-01-01", periods=5)
    return pd.DataFrame({
        "Open": [10.0, 10.5, 11.0, 10.8, 10.6],
        "High": [10.5, 11.0, 11.5, 11.2, 11.0],
        "Low": [9.8, 10.2, 10.5, 10.3, 10.2],
        "Close": [10.2, 10.8, 11.2, 10.5, 10.4],
        "Volume": [100000, 200000, 150000, 180000, 120000],
    }, index=dates)


@pytest.fixture
def flat_ohlcv():
    """生成所有价格完全相同的DataFrame（零波动）"""
    n = 50
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    return pd.DataFrame({
        "Open": [15.0] * n,
        "High": [15.0] * n,
        "Low": [15.0] * n,
        "Close": [15.0] * n,
        "Volume": [500000] * n,
    }, index=dates)


# ──────────────────────────────────────────────────────────
#  测试用例
# ──────────────────────────────────────────────────────────

class TestIndicatorCalcBaseBasic:
    """基本指标计算测试"""

    def test_returns_dataframe(self, sample_ohlcv):
        """IndicatorCalcBase应返回一个DataFrame"""
        result = IndicatorCalcBase(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, sample_ohlcv):
        """输出应包含所有必需的技术指标列"""
        result = IndicatorCalcBase(sample_ohlcv)
        required_cols = ["white_line", "yellow_line", "atr14", "volume_ma", "K", "D", "J"]
        for col in required_cols:
            assert col in result.columns, f"缺少必需列: {col}"

    def test_output_preserves_original_columns(self, sample_ohlcv):
        """输出应保留原始OHLCV列"""
        result = IndicatorCalcBase(sample_ohlcv)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_output_length_matches_input(self, sample_ohlcv):
        """输出行数应与输入行数一致"""
        result = IndicatorCalcBase(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_does_not_mutate_input(self, sample_ohlcv):
        """IndicatorCalcBase不应修改原始DataFrame"""
        original = sample_ohlcv.copy()
        IndicatorCalcBase(sample_ohlcv)
        pd.testing.assert_frame_equal(sample_ohlcv, original)


class TestIndicatorCalcBaseValues:
    """指标值合理性测试"""

    def test_no_nan_in_recent_rows(self, sample_ohlcv):
        """近期行（最后50行）的关键指标不应有NaN"""
        result = IndicatorCalcBase(sample_ohlcv)
        key_cols = ["white_line", "yellow_line", "atr14", "volume_ma", "K", "D", "J"]
        recent = result[key_cols].tail(50)
        assert not recent.isna().any().any(), "近期行存在NaN值"

    def test_atr_non_negative(self, sample_ohlcv):
        """ATR14不应为负值"""
        result = IndicatorCalcBase(sample_ohlcv)
        assert (result["atr14"] >= 0).all(), "ATR14存在负值"

    def test_atr_positive_for_volatile_data(self, sample_ohlcv):
        """有波动的数据ATR应大于0"""
        result = IndicatorCalcBase(sample_ohlcv)
        assert (result["atr14"].tail(50) > 0).all(), "波动数据的ATR14不应为0"

    def test_j_value_in_range(self, sample_ohlcv):
        """J值应在[0, 100]范围内（代码中做了clip）"""
        result = IndicatorCalcBase(sample_ohlcv)
        assert (result["J"] >= 0).all() and (result["J"] <= 100).all(), \
            f"J值超出范围: min={result['J'].min()}, max={result['J'].max()}"

    def test_kd_values_reasonable(self, sample_ohlcv):
        """K、D值应在合理范围内（0~100附近）"""
        result = IndicatorCalcBase(sample_ohlcv)
        # K/D 使用 ewm，理论上可以略超出0-100，但不应极端
        assert (result["K"].tail(50) >= -10).all() and (result["K"].tail(50) <= 110).all()
        assert (result["D"].tail(50) >= -10).all() and (result["D"].tail(50) <= 110).all()

    def test_volume_ma_positive(self, sample_ohlcv):
        """成交量均线应为正值"""
        result = IndicatorCalcBase(sample_ohlcv)
        assert (result["volume_ma"].tail(50) > 0).all(), "volume_ma存在非正值"

    def test_white_line_close_to_price(self, sample_ohlcv):
        """白线（双EWM平滑）应与收盘价接近"""
        result = IndicatorCalcBase(sample_ohlcv)
        recent = result.tail(50)
        # 白线应与收盘价在同一个数量级
        ratio = recent["white_line"] / recent["Close"]
        assert (ratio > 0.8).all() and (ratio < 1.2).all(), \
            "白线与收盘价偏离过大"

    def test_yellow_line_close_to_price(self, sample_ohlcv):
        """黄线（多均线均值）应与收盘价接近"""
        result = IndicatorCalcBase(sample_ohlcv)
        recent = result.tail(50)
        ratio = recent["yellow_line"] / recent["Close"]
        assert (ratio > 0.7).all() and (ratio < 1.3).all(), \
            "黄线与收盘价偏离过大"


class TestIndicatorCalcBaseEdgeCases:
    """边界情况测试"""

    def test_short_dataframe(self, short_ohlcv):
        """极短DataFrame（5行）应能正常计算，不抛异常"""
        result = IndicatorCalcBase(short_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        # 关键列应存在
        for col in ["white_line", "yellow_line", "atr14", "volume_ma", "K", "D", "J"]:
            assert col in result.columns

    def test_short_dataframe_no_crash(self, short_ohlcv):
        """极短DataFrame计算后，关键指标列应存在值（可能前几行有NaN）"""
        result = IndicatorCalcBase(short_ohlcv)
        # 至少最后一行应有有效值
        for col in ["white_line", "yellow_line", "atr14", "K", "D", "J"]:
            assert not pd.isna(result[col].iloc[-1]), f"短数据最后一行的{col}为NaN"

    def test_flat_prices(self, flat_ohlcv):
        """所有价格相同的DataFrame应能正常计算"""
        result = IndicatorCalcBase(flat_ohlcv)
        assert isinstance(result, pd.DataFrame)
        # ATR应为0（无波动）
        assert (result["atr14"].tail(10) == 0).all(), "零波动数据ATR应为0"
        # J值：当high==low时，RSV=(Close-low)/(high-low+1e-8)*100
        # 由于high==low，分母≈1e-8，分子≈0，RSV≈0，因此J被clip到0
        # 这是代码中1e-8防零除的设计结果
        assert result["J"].iloc[-1] == 0, \
            f"零波动数据J值应为0（RSV→0），实际为{result['J'].iloc[-1]}"

    def test_flat_prices_no_nan(self, flat_ohlcv):
        """零波动数据不应产生NaN"""
        result = IndicatorCalcBase(flat_ohlcv)
        key_cols = ["white_line", "yellow_line", "atr14", "volume_ma", "K", "D", "J"]
        assert not result[key_cols].isna().any().any(), "零波动数据存在NaN"

    def test_single_row_dataframe(self):
        """单行DataFrame应不抛异常"""
        df = pd.DataFrame({
            "Open": [10.0],
            "High": [10.5],
            "Low": [9.5],
            "Close": [10.2],
            "Volume": [100000],
        }, index=pd.bdate_range("2024-01-01", periods=1))
        result = IndicatorCalcBase(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
