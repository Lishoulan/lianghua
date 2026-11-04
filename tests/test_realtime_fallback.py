"""测试实时行情的多源降级与降级标注

背景（真实故障）：
    GitHub Actions runner（境外 IP）上 akshare 的东财接口被限流，
    实时行情返回空/非 JSON，导致盘中推送静默退化成"盘后"推送，
    用户在 13:15 收到的却是盘后数据。

覆盖场景：
    - 东财源可用时直接使用
    - 东财源失败时自动降级到新浪源（这是修复的核心）
    - 新浪源的代码前缀（sh600000）被正确剥掉
    - 新浪源的成交量单位（股）被换算成"手"（与缓存日线一致）
    - 全部源失败时返回空字典（不抛异常）
    - 降级时推送标题/正文显式标注，不装作盘中推送
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classic_ta.common import stock_pool
from classic_ta.common.message_builder import build_beta_push_message, build_push_message
from classic_ta.common.stock_pool import (
    _get_spot_df,
    _get_spot_df_em,
    _get_spot_df_sina,
    _normalize_spot_code,
    get_realtime_quotes,
)

# ──────────────────────────────────────────────────────────
#  样本数据
# ──────────────────────────────────────────────────────────

EM_DF = pd.DataFrame(
    {
        "代码": ["600000", "000001"],
        "名称": ["浦发银行", "平安银行"],
        "最新价": [9.63, 11.31],
        "今开": [9.66, 11.41],
        "最高": [9.70, 11.42],
        "最低": [9.55, 11.26],
        "昨收": [9.66, 11.41],
        "成交量": [410000.43, 1053674.72],      # 东财单位：手
        "成交额": [394714699.0, 1192231536.0],
        "涨跌幅": [-0.31, -0.88],
        "换手率": [0.14, 0.55],
        "量比": [1.02, 0.98],
    }
)

SINA_DF = pd.DataFrame(
    {
        "代码": ["sh600000", "sz000001"],      # 新浪带市场前缀
        "名称": ["浦发银行", "平安银行"],
        "最新价": [9.63, 11.31],
        "今开": [9.66, 11.41],
        "最高": [9.70, 11.42],
        "最低": [9.55, 11.26],
        "昨收": [9.66, 11.41],
        "成交量": [41000043.0, 105367472.0],    # 新浪单位：股
        "成交额": [394714699.0, 1192231536.0],
        "涨跌幅": [-0.31, -0.88],
        "换手率": [0.14, 0.55],
        "量比": [1.02, 0.98],
    }
)


def _pad(df, code_col, prefix, total=120):
    """补齐到生产量级（>100 行），真实接口只会返回 5000+ 行。

    prefix 决定代码形态：东财是 6 位裸代码，新浪是 8 位带前缀。
    """
    extra_rows = []
    for i in range(total - len(df)):
        extra_rows.append({
            code_col: f"{prefix}{i:04d}",
            "名称": f"测试股{i}",
            "最新价": 10.0, "今开": 10.0, "最高": 10.5, "最低": 9.5, "昨收": 10.0,
            "成交量": 1000.0, "成交额": 10000.0, "涨跌幅": 0.0,
            "换手率": 0.1, "量比": 1.0,
        })
    return pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)


@pytest.fixture(autouse=True)
def _clear_spot_cache():
    """现货缓存是模块级的，测试之间必须清干净，否则互相串数据。"""
    stock_pool._SPOT_CACHE["df"] = None
    stock_pool._SPOT_CACHE["ts"] = 0.0
    yield
    stock_pool._SPOT_CACHE["df"] = None
    stock_pool._SPOT_CACHE["ts"] = 0.0


@pytest.fixture
def fake_akshare(monkeypatch):
    """伪造 akshare 模块：可控制东财/新浪两个源各自的成败。"""
    import types

    state = {"em_ok": True, "sina_ok": True, "em_calls": 0, "sina_calls": 0}

    def spot_em():
        state["em_calls"] += 1
        if not state["em_ok"]:
            raise RuntimeError("Expecting value: line 1 column 1 (char 0)")
        return _pad(EM_DF.copy(), "代码", "61")

    def spot_sina():
        state["sina_calls"] += 1
        if not state["sina_ok"]:
            raise RuntimeError("新浪源也不可用")
        return _pad(SINA_DF.copy(), "代码", "sh61")

    module = types.ModuleType("akshare")
    module.stock_zh_a_spot_em = spot_em
    module.stock_zh_a_spot = spot_sina
    monkeypatch.setitem(sys.modules, "akshare", module)
    return state


# ──────────────────────────────────────────────────────────
#  代码归一化
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("sh600000", "600000"),
    ("sz000001", "000001"),
    ("bj920000", "920000"),
    ("600000", "600000"),
])
def test_normalize_spot_code(raw, expected):
    assert _normalize_spot_code(raw) == expected


# ──────────────────────────────────────────────────────────
#  数据源降级链
# ──────────────────────────────────────────────────────────

def test_em_source_used_when_available(fake_akshare):
    df = _get_spot_df()
    assert df is not None
    assert fake_akshare["em_calls"] == 1
    assert fake_akshare["sina_calls"] == 0


# ──────────────────────────────────────────────────────────
#  腾讯源（HTTPS 直连，第二顺位）
# ──────────────────────────────────────────────────────────

def _tencent_payload(code="600000", name="浦发银行", close=9.68, prev=9.66, open_=9.66,
                     high=9.72, low=9.55, volume=599733, amount=578212864):
    """按 qt.gtimg.cn 的标准字段顺序拼一条行情。"""
    f = [""] * 100
    f[0] = "1"
    f[1] = name
    f[2] = code
    f[3] = str(close)
    f[4] = str(prev)
    f[5] = str(open_)
    f[6] = str(volume)
    f[30] = "20261022145604"
    f[31] = f"{close - prev:.2f}"
    f[32] = f"{(close - prev) / prev * 100:.2f}"
    f[33] = str(high)
    f[34] = str(low)
    f[35] = f"{close}/{volume}/{amount}"
    f[36] = str(volume)
    f[37] = str(int(amount / 10000))
    f[38] = "0.18"
    return f'v_sh{code}="' + "~".join(f) + '";'


TENCENT_PAYLOAD = _tencent_payload()


def _fake_tencent(monkeypatch, payload=TENCENT_PAYLOAD):
    import types

    calls = {"n": 0}

    class _Resp:
        status_code = 200
        encoding = None
        text = payload

    def get(url, timeout=None, **kwargs):
        calls["n"] += 1
        return _Resp()

    requests_mod = types.ModuleType("requests")
    requests_mod.Session = lambda: types.SimpleNamespace(
        headers={},
        get=get,
    )
    monkeypatch.setitem(sys.modules, "requests", requests_mod)
    return calls


def test_tencent_source_parses_fields(monkeypatch):
    """腾讯字段解析：成交量单位是「手」，与东财/缓存日线一致。"""
    _fake_tencent(monkeypatch)
    monkeypatch.setattr(stock_pool, "_a_share_symbols", lambda: ["sh600000"])

    df = stock_pool._get_spot_df_tencent()
    assert df is not None
    assert len(df) == 1

    row = df.iloc[0]
    assert row["代码"] == "600000"
    assert row["名称"] == "浦发银行"
    assert row["最新价"] == pytest.approx(9.68)
    assert row["昨收"] == pytest.approx(9.66)
    assert row["今开"] == pytest.approx(9.66)
    assert row["最高"] == pytest.approx(9.72)
    assert row["最低"] == pytest.approx(9.55)
    assert row["成交量"] == pytest.approx(599733.0)     # 手，无需换算
    assert row["成交额"] == pytest.approx(578212864.0)  # 元
    assert row["涨跌幅"] == pytest.approx((9.68 - 9.66) / 9.66 * 100)


def test_tencent_used_when_em_fails(monkeypatch, fake_akshare):
    """东财失败 → 腾讯 → 新浪 的降级顺序。"""
    fake_akshare["em_ok"] = False
    _fake_tencent(monkeypatch)
    monkeypatch.setattr(stock_pool, "_a_share_symbols", lambda: ["sh600000"])

    df = _get_spot_df()
    assert df is not None
    assert len(df) == 1
    assert fake_akshare["em_calls"] == 1
    assert fake_akshare["sina_calls"] == 0, "腾讯可用时不该再打新浪"


def test_tencent_failure_falls_back_to_sina(monkeypatch, fake_akshare):
    """腾讯也挂 → 继续降级到新浪。"""
    fake_akshare["em_ok"] = False
    _fake_tencent(monkeypatch, payload="")   # 解析不出任何行
    monkeypatch.setattr(stock_pool, "_a_share_symbols", lambda: ["sh600000"])

    df = _get_spot_df()
    assert df is not None
    assert fake_akshare["sina_calls"] == 1


def test_spot_cache_avoids_refetch(monkeypatch, fake_akshare):
    """TTL 缓存：一次流程里不该反复拉现货。"""
    fake_akshare["em_ok"] = False
    _fake_tencent(monkeypatch)
    monkeypatch.setattr(stock_pool, "_a_share_symbols", lambda: ["sh600000"])
    monkeypatch.setitem(stock_pool._SPOT_CACHE, "df", None)
    monkeypatch.setitem(stock_pool._SPOT_CACHE, "ts", 0.0)

    first = stock_pool._get_spot_df_cached()
    calls_after_first = fake_akshare["em_calls"]
    second = stock_pool._get_spot_df_cached()

    assert first is not None
    assert second is first
    assert fake_akshare["em_calls"] == calls_after_first, "缓存命中不应再打数据源"


def test_falls_back_to_sina_when_em_fails(fake_akshare):
    """核心场景：东财被限流时必须自动切到新浪，而不是返回空。"""
    fake_akshare["em_ok"] = False

    df = _get_spot_df()
    assert df is not None, "东财失败时应降级到新浪源"
    assert fake_akshare["em_calls"] == 1
    assert fake_akshare["sina_calls"] == 1
    assert len(df) == 120


def test_returns_none_when_all_sources_fail(fake_akshare):
    fake_akshare["em_ok"] = False
    fake_akshare["sina_ok"] = False

    assert _get_spot_df() is None
    assert _get_spot_df_em() is None
    assert _get_spot_df_sina() is None


def test_sina_source_strips_market_prefix(fake_akshare):
    fake_akshare["em_ok"] = False

    df = _get_spot_df_sina()
    assert list(df["代码"][:2]) == ["600000", "000001"]


def test_sina_source_converts_volume_shares_to_lots(fake_akshare):
    """新浪成交量单位是「股」，缓存日线用「手」，必须换算（否则量能差 100 倍）。"""
    fake_akshare["em_ok"] = False

    df = _get_spot_df_sina()
    assert df["成交量"].iloc[0] == pytest.approx(410000.43)
    assert df["成交量"].iloc[1] == pytest.approx(1053674.72)


def test_volume_units_match_between_sources(fake_akshare):
    """两个源的成交量换算后必须一致，否则盘中 K 线量能会错。"""
    em = _get_spot_df_em()
    fake_akshare["em_ok"] = False
    sina = _get_spot_df_sina()

    assert em["成交量"].iloc[0] == pytest.approx(sina["成交量"].iloc[0])
    assert em["成交额"].iloc[0] == pytest.approx(sina["成交额"].iloc[0])


# ──────────────────────────────────────────────────────────
#  实时行情
# ──────────────────────────────────────────────────────────

def test_get_realtime_quotes_uses_sina_fallback(fake_akshare):
    fake_akshare["em_ok"] = False

    quotes = get_realtime_quotes()
    assert len(quotes) == 120
    assert "600000.SH" in quotes
    assert "000001.SZ" in quotes

    q = quotes["600000.SH"]
    assert q["Close"] == pytest.approx(9.63)
    assert q["Open"] == pytest.approx(9.66)
    assert q["High"] == pytest.approx(9.70)
    assert q["Low"] == pytest.approx(9.55)
    assert q["PrevClose"] == pytest.approx(9.66)
    # 成交量已从「股」换算成「手」
    assert q["Volume"] == pytest.approx(410000.43)


def test_get_realtime_quotes_empty_when_all_fail(fake_akshare):
    fake_akshare["em_ok"] = False
    fake_akshare["sina_ok"] = False

    assert get_realtime_quotes() == {}


def test_prefilter_uses_sina_fallback(fake_akshare):
    fake_akshare["em_ok"] = False

    df = stock_pool.batch_prefilter_stocks()
    assert df is not None, "预筛选也必须能降级，否则会退化成全市场扫描"
    assert len(df) == 120
    assert list(df["ts_code"][:2]) == ["600000.SH", "000001.SZ"]


# ──────────────────────────────────────────────────────────
#  降级标注
# ──────────────────────────────────────────────────────────

def test_degraded_push_message_is_marked():
    title, desp = build_push_message(None, [], [], {}, is_intraday=False, degraded=True)

    assert "降级" in title
    assert "实时行情" in desp
    assert "勿当作盘中实时信号" in desp


def test_normal_push_message_not_marked():
    title, desp = build_push_message(None, [], [], {}, is_intraday=True, degraded=False)

    assert "降级" not in title
    assert "盘中实时" in title


def test_beta_degraded_push_message_is_marked():
    title, _ = build_beta_push_message(None, [], [], {}, is_intraday=False, degraded=True)
    assert "降级" in title


# ──────────────────────────────────────────────────────────
#  时段判定（mode 优先于墙上时钟）
# ──────────────────────────────────────────────────────────

def test_mode_intraday_wins_over_late_wall_clock():
    """13:15 触发但容器 15:05 才开跑 —— 必须仍是盘中，不能变成盘后。"""
    from datetime import datetime

    from classic_ta.daily_push import _resolve_intraday

    late = datetime(2026, 10, 22, 15, 5)
    assert _resolve_intraday("intraday", late) is True
    assert _resolve_intraday("after_hours", late) is False


def test_falls_back_to_wall_clock_without_mode():
    from datetime import datetime

    from classic_ta.daily_push import _resolve_intraday

    assert _resolve_intraday(None, datetime(2026, 10, 22, 13, 30)) is True
    assert _resolve_intraday(None, datetime(2026, 10, 22, 17, 30)) is False
    assert _resolve_intraday(None, datetime(2026, 10, 22, 8, 0)) is False
