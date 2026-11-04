"""测试 dry_run 真的不推送

背景（真实故障）：
    工作流通过 workflow_dispatch 的 dry_run 输入传 DRY_RUN 环境变量，
    但 daily_push() 从来没读过它 —— 干跑照样把消息推给了用户。
    干跑逻辑只写在 `if __name__ == "__main__"` 块里（依赖 --dry-run 命令行参数），
    而 CI 走的是 run_scan_pipeline → run_daily_push() → 直接调用函数。

覆盖：
    - daily_push / run_daily_push 支持 dry_run 参数
    - pipeline 从 DRY_RUN 环境变量解析干跑标记并传下去
    - 干跑时跳过 Server酱推送与公众号群发（真实断言：不调用推送接口）
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_daily_push_accepts_dry_run_parameter():
    import inspect

    from classic_ta.daily_push import daily_push

    sig = inspect.signature(daily_push)
    assert "dry_run" in sig.parameters
    assert sig.parameters["dry_run"].default is False


def test_run_daily_push_accepts_dry_run():
    import inspect

    from scripts.run_bytecode_daily_push import run_daily_push

    sig = inspect.signature(run_daily_push)
    assert "dry_run" in sig.parameters
    assert sig.parameters["dry_run"].default is False


def test_pipeline_reads_dry_run_env(monkeypatch):
    """DRY_RUN 环境变量要被真正识别并传下去。"""
    import inspect

    import scripts.run_scan_pipeline as pipeline

    src = inspect.getsource(pipeline.main)
    assert "DRY_RUN" in src, "pipeline 必须读 DRY_RUN 环境变量"
    assert "dry_run=dry_run" in src, "pipeline 必须把 dry_run 传给 run_daily_push"


@pytest.mark.parametrize("value,expected", [
    ("1", True),
    ("true", True),
    ("TRUE", True),
    ("yes", True),
    ("0", False),
    ("false", False),
    ("", False),
])
def test_dry_run_env_parsing(monkeypatch, value, expected):
    """干跑标记解析：只有 1/true/yes 算干跑。"""
    import scripts.run_scan_pipeline as pipeline

    parsed = pipeline._parse_dry_run(value)
    assert parsed is expected


def test_dry_run_does_not_call_push_channels(monkeypatch):
    """核心断言：干跑时一次推送接口都不能被调用。"""
    from classic_ta import daily_push as dp

    calls = {"serverchan": 0, "wechat": 0}

    def fake_send_group_push(*args, **kwargs):
        calls["serverchan"] += 1
        return {"admin": True, "beta": True}

    def fake_wechat(*args, **kwargs):
        calls["wechat"] += 1
        return {"success": True}

    monkeypatch.setattr(dp, "send_group_push", fake_send_group_push)
    monkeypatch.setitem(sys.modules, "wechat_push", type("m", (), {
        "push_signals_to_wechat": staticmethod(fake_wechat),
    }))

    dp._run_pushes(
        admin_title="标题A", admin_desp="内容A",
        beta_title="标题B", beta_desp="内容B",
        scheduled_time=None,
        dry_run=True, is_intraday=False, degraded=False,
        oamv_status=None, signals=[], industry_stats=[],
    )

    assert calls["serverchan"] == 0, "干跑不能调用 Server酱"
    assert calls["wechat"] == 0, "干跑不能调用公众号群发"


def test_normal_run_calls_push_channels(monkeypatch):
    """反证：非干跑时确实会推送（别把推送改没了）。"""
    from classic_ta import daily_push as dp

    calls = {"serverchan": 0, "wechat": 0}

    def fake_send_group_push(*args, **kwargs):
        calls["serverchan"] += 1
        return {"admin": True, "beta": True}

    def fake_wechat(*args, **kwargs):
        calls["wechat"] += 1
        return {"success": True}

    monkeypatch.setattr(dp, "send_group_push", fake_send_group_push)
    monkeypatch.setitem(sys.modules, "wechat_push", type("m", (), {
        "push_signals_to_wechat": staticmethod(fake_wechat),
    }))

    dp._run_pushes(
        admin_title="标题A", admin_desp="内容A",
        beta_title="标题B", beta_desp="内容B",
        scheduled_time=None,
        dry_run=False, is_intraday=False, degraded=False,
        oamv_status=None, signals=[], industry_stats=[],
    )

    assert calls["serverchan"] == 1
    assert calls["wechat"] == 1


def test_intraday_skips_wechat_group_push(monkeypatch):
    """盘中模式不发公众号群发（只有盘后发）。"""
    from classic_ta import daily_push as dp

    calls = {"wechat": 0}

    monkeypatch.setattr(dp, "send_group_push", lambda *a, **k: {"admin": True, "beta": True})
    monkeypatch.setitem(sys.modules, "wechat_push", type("m", (), {
        "push_signals_to_wechat": staticmethod(lambda *a, **k: calls.__setitem__("wechat", calls["wechat"] + 1)),
    }))

    dp._run_pushes(
        admin_title="t", admin_desp="d", beta_title="t", beta_desp="d",
        scheduled_time=None,
        dry_run=False, is_intraday=True, degraded=False,
        oamv_status=None, signals=[], industry_stats=[],
    )

    assert calls["wechat"] == 0
