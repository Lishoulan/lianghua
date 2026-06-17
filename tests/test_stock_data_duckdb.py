"""测试 stock_data_duckdb 缓存模块

覆盖场景：
  - _clean_dataframe: 移除Volume=0的停牌日
  - _clean_dataframe: 移除OHLC为NaN或0的行
  - _clean_dataframe: 前复权突变检测（单日涨跌幅>50%）
  - DuckDB缓存保存/加载往返测试
  - get_cache_stats返回预期键
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from classic_ta.stock_data_duckdb import (
    _clean_dataframe,
    save_stock_cache,
    load_stock_cache,
    get_cache_stats,
    _is_duckdb_available,
)


# ──────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def clean_ohlcv():
    """生成干净的OHLCV数据"""
    n = 50
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    np.random.seed(42)
    close = 20.0 + np.random.normal(0, 0.5, n).cumsum()
    return pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.3,
        "Low": close - 0.3,
        "Close": close,
        "Volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=dates)


@pytest.fixture
def dirty_ohlcv():
    """生成包含脏数据的OHLCV：Volume=0、NaN OHLC、前复权突变"""
    n = 50
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    np.random.seed(42)
    close = 20.0 + np.random.normal(0, 0.5, n).cumsum()
    df = pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.3,
        "Low": close - 0.3,
        "Close": close,
        "Volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=dates)

    # 注入脏数据
    # Volume=0（停牌日）
    df.iloc[5, df.columns.get_loc("Volume")] = 0
    df.iloc[15, df.columns.get_loc("Volume")] = 0

    # Close为NaN
    df.iloc[10, df.columns.get_loc("Close")] = np.nan

    # Open为0
    df.iloc[20, df.columns.get_loc("Open")] = 0

    # 前复权突变（单日涨跌幅>50%）
    df.iloc[30, df.columns.get_loc("Close")] = df.iloc[29, df.columns.get_loc("Close")] * 2.0

    return df


# ──────────────────────────────────────────────────────────
#  _clean_dataframe 测试
# ──────────────────────────────────────────────────────────

class TestCleanDataframe:
    """_clean_dataframe 数据清洗测试"""

    def test_clean_data_removes_volume_zero(self, dirty_ohlcv):
        """清洗应移除Volume=0的行"""
        result = _clean_dataframe(dirty_ohlcv)
        assert (result["Volume"] > 0).all(), "清洗后仍存在Volume=0的行"

    def test_clean_data_removes_nan_ohlc(self, dirty_ohlcv):
        """清洗应移除OHLC为NaN的行"""
        result = _clean_dataframe(dirty_ohlcv)
        for col in ["Open", "High", "Low", "Close"]:
            assert not result[col].isna().any(), f"清洗后{col}列仍存在NaN"

    def test_clean_data_removes_zero_ohlc(self, dirty_ohlcv):
        """清洗应移除OHLC为0的行"""
        result = _clean_dataframe(dirty_ohlcv)
        for col in ["Open", "High", "Low", "Close"]:
            assert (result[col] > 0).all(), f"清洗后{col}列仍存在0值"

    def test_clean_data_removes_forward_adjustment_anomaly(self, dirty_ohlcv):
        """清洗应移除前复权突变行（单日涨跌幅>50%）"""
        result = _clean_dataframe(dirty_ohlcv)
        if len(result) > 1:
            pct_change = result["Close"].pct_change()
            # 清洗后不应有>50%的单日涨跌幅（少量异常被移除）
            extreme_changes = (pct_change.abs() > 0.5).sum()
            assert extreme_changes == 0, f"清洗后仍存在{extreme_changes}行涨跌幅>50%"

    def test_clean_data_preserves_good_rows(self, clean_ohlcv):
        """干净数据清洗后行数不应减少太多"""
        result = _clean_dataframe(clean_ohlcv)
        assert len(result) == len(clean_ohlcv), "干净数据清洗后行数不应减少"

    def test_clean_data_empty_input(self):
        """空DataFrame输入应返回空DataFrame"""
        df = pd.DataFrame()
        result = _clean_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_clean_data_none_input(self):
        """None输入应返回None"""
        result = _clean_dataframe(None)
        assert result is None

    def test_clean_data_missing_column(self):
        """缺少必要列应返回空DataFrame"""
        df = pd.DataFrame({"Open": [1], "High": [2], "Low": [0.5]})
        result = _clean_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_clean_data_column_case_insensitive(self):
        """列名大小写应被标准化"""
        n = 10
        dates = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame({
            "open": [10.0] * n,
            "high": [10.5] * n,
            "low": [9.5] * n,
            "close": [10.2] * n,
            "volume": [500000] * n,
        }, index=dates)
        result = _clean_dataframe(df)
        assert len(result) == n
        # 列名应被标准化为首字母大写
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_clean_data_many_anomalies_warning(self):
        """大量前复权异常（>3行）应保留数据（仅记录警告）"""
        n = 50
        dates = pd.bdate_range("2024-01-01", periods=n)
        np.random.seed(42)
        close = 20.0 + np.random.normal(0, 0.5, n).cumsum()
        df = pd.DataFrame({
            "Open": close - 0.1,
            "High": close + 0.3,
            "Low": close - 0.3,
            "Close": close,
            "Volume": np.random.randint(100_000, 1_000_000, n).astype(float),
        }, index=dates)
        # 注入5个突变行（>3个，应保留数据仅警告）
        for i in [10, 15, 20, 25, 30]:
            df.iloc[i, df.columns.get_loc("Close")] = df.iloc[i - 1, df.columns.get_loc("Close")] * 2.0
        result = _clean_dataframe(df)
        # 大量异常时不删除，数据应保留
        assert len(result) > 0


# ──────────────────────────────────────────────────────────
#  DuckDB 缓存往返测试
# ──────────────────────────────────────────────────────────

class TestDuckDBCacheRoundtrip:
    """DuckDB缓存保存/加载往返测试"""

    @pytest.fixture
    def mock_duckdb_env(self, tmp_path):
        """Mock DuckDB环境，使用临时目录"""
        duckdb_path = tmp_path / "test_cache.duckdb"
        with patch("classic_ta.stock_data_duckdb._is_duckdb_available", return_value=True), \
             patch("classic_ta.stock_data_duckdb.DUCKDB_PATH", duckdb_path), \
             patch("classic_ta.stock_data_duckdb._get_duckdb_conn") as mock_conn_factory:
            # 创建真实的DuckDB连接
            import duckdb
            conn = duckdb.connect(str(duckdb_path), read_only=False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_data (
                    ts_code VARCHAR,
                    date DATE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE
                )
            """)
            mock_conn_factory.return_value = conn
            yield conn, duckdb_path
            conn.close()

    def test_save_and_load_roundtrip(self, mock_duckdb_env, clean_ohlcv):
        """保存后加载应返回相同数据（OHLCV列）"""
        conn, duckdb_path = mock_duckdb_env
        ts_code = "600519.SH"

        with patch("classic_ta.stock_data_duckdb._is_duckdb_available", return_value=True), \
             patch("classic_ta.stock_data_duckdb.DUCKDB_PATH", duckdb_path), \
             patch("classic_ta.stock_data_duckdb._get_duckdb_conn", return_value=conn):
            save_stock_cache(ts_code, clean_ohlcv)
            loaded = load_stock_cache(ts_code)

        assert loaded is not None, "加载返回None"
        assert len(loaded) > 0, "加载数据为空"
        # 验证列名
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in loaded.columns, f"加载后缺少列: {col}"

    def test_save_cleans_data_before_storing(self, mock_duckdb_env, dirty_ohlcv):
        """保存时应对数据进行清洗"""
        conn, duckdb_path = mock_duckdb_env
        ts_code = "600519.SH"

        with patch("classic_ta.stock_data_duckdb._is_duckdb_available", return_value=True), \
             patch("classic_ta.stock_data_duckdb.DUCKDB_PATH", duckdb_path), \
             patch("classic_ta.stock_data_duckdb._get_duckdb_conn", return_value=conn):
            save_stock_cache(ts_code, dirty_ohlcv)
            loaded = load_stock_cache(ts_code)

        if loaded is not None:
            assert (loaded["Volume"] > 0).all(), "保存后仍存在Volume=0"
            for col in ["Open", "High", "Low", "Close"]:
                assert not loaded[col].isna().any(), f"保存后{col}仍存在NaN"
                assert (loaded[col] > 0).all(), f"保存后{col}仍存在0值"

    def test_load_nonexistent_returns_none(self, mock_duckdb_env):
        """加载不存在的股票代码应返回None"""
        conn, duckdb_path = mock_duckdb_env

        with patch("classic_ta.stock_data_duckdb._is_duckdb_available", return_value=True), \
             patch("classic_ta.stock_data_duckdb.DUCKDB_PATH", duckdb_path), \
             patch("classic_ta.stock_data_duckdb._get_duckdb_conn", return_value=conn):
            result = load_stock_cache("999999.SZ")

        assert result is None


# ──────────────────────────────────────────────────────────
#  get_cache_stats 测试
# ──────────────────────────────────────────────────────────

class TestGetCacheStats:
    """缓存统计信息测试"""

    def test_returns_dict(self):
        """get_cache_stats应返回字典"""
        with patch("classic_ta.stock_data_duckdb._is_duckdb_available", return_value=False):
            stats = get_cache_stats()
        assert isinstance(stats, dict)

    def test_has_expected_keys(self):
        """返回的字典应包含mode、count键"""
        with patch("classic_ta.stock_data_duckdb._is_duckdb_available", return_value=False):
            stats = get_cache_stats()
        assert "mode" in stats, "缺少mode键"
        assert "count" in stats, "缺少count键"

    def test_duckdb_mode_keys(self, tmp_path):
        """DuckDB模式应返回mode、count、dir、size_mb键"""
        duckdb_path = tmp_path / "test_stats.duckdb"
        # 创建空DuckDB文件
        import duckdb
        conn = duckdb.connect(str(duckdb_path), read_only=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_data (
                ts_code VARCHAR, date DATE,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE
            )
        """)
        conn.close()

        with patch("classic_ta.stock_data_duckdb._is_duckdb_available", return_value=True), \
             patch("classic_ta.stock_data_duckdb.DUCKDB_PATH", duckdb_path):
            stats = get_cache_stats()

        assert stats["mode"] == "duckdb"
        assert "count" in stats
        assert "dir" in stats
        assert "size_mb" in stats
