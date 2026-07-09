import pandas as pd
import pytest

from classic_ta.common.stock_pool import append_realtime_bar


def test_append_realtime_bar_scales_new_intraday_row_to_qfq_space():
    df = pd.DataFrame(
        {
            "Open": [9.8, 10.4],
            "High": [10.2, 10.8],
            "Low": [9.7, 10.2],
            "Close": [10.0, 10.5],
            "Volume": [1000.0, 1200.0],
            "Amount": [10000.0, 12600.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    realtime_quote = {
        "Open": 9.7,
        "High": 10.3,
        "Low": 9.6,
        "Close": 10.1,
        "PrevClose": 10.0,
        "Volume": 1500.0,
        "Amount": 15150.0,
    }

    result = append_realtime_bar(df.copy(), realtime_quote, today_str="2024-01-04")

    ratio = 10.5 / 10.0
    latest = result.iloc[-1]
    assert latest["Open"] == pytest.approx(9.7 * ratio)
    assert latest["High"] == pytest.approx(10.3 * ratio)
    assert latest["Low"] == pytest.approx(9.6 * ratio)
    assert latest["Close"] == pytest.approx(10.1 * ratio)
    assert latest["Volume"] == pytest.approx(1500.0)


def test_append_realtime_bar_uses_previous_trading_day_when_today_row_exists():
    df = pd.DataFrame(
        {
            "Open": [9.8, 10.4, 10.6],
            "High": [10.2, 10.8, 10.9],
            "Low": [9.7, 10.2, 10.5],
            "Close": [10.0, 10.5, 10.7],
            "Volume": [1000.0, 1200.0, 1300.0],
            "Amount": [10000.0, 12600.0, 13910.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    realtime_quote = {
        "Open": 9.7,
        "High": 10.3,
        "Low": 9.6,
        "Close": 10.1,
        "PrevClose": 10.0,
        "Volume": 1500.0,
        "Amount": 15150.0,
    }

    result = append_realtime_bar(df.copy(), realtime_quote, today_str="2024-01-04")

    ratio = 10.5 / 10.0
    latest = result.iloc[-1]
    assert latest["Open"] == pytest.approx(9.7 * ratio)
    assert latest["High"] == pytest.approx(10.3 * ratio)
    assert latest["Low"] == pytest.approx(9.6 * ratio)
    assert latest["Close"] == pytest.approx(10.1 * ratio)
