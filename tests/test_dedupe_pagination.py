"""测试 artifact 去重查找必须翻页

背景（真实故障）：
    check-duplicate 用 `gh api "repos/.../actions/artifacts"` 查当天是否推过，
    但 gh api 默认只返回第一页（30 条）。仓库里每次 CI 都会产生
    coverage-* / *.dockerbuild 等无关 artifact，一天就能把首页挤满，
    当天的 push-results artifact 被挤出首页 → 查不到 → 去重失效 → 同一天重复推送。

覆盖：
    - 目标 artifact 在第一页 → 能找到
    - 目标 artifact 在第 2 页及以后 → 也能找到（这是修复的核心）
    - 不存在 → 无输出
    - 工作流确实调用了这个脚本
"""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "has_push_artifact.sh"

BASH = None
_candidates = [os.environ.get("BASH"), shutil.which("bash")]

# Git for Windows 的 bash 不在 PATH 里，从 git.exe 位置反推
_git = shutil.which("git")
if _git:
    _git_root = Path(_git).resolve().parent.parent
    _candidates += [
        str(_git_root / "bin" / "bash.exe"),
        str(_git_root / "usr" / "bin" / "bash.exe"),
    ]
_candidates.append(r"C:\Program Files\Git\bin\bash.exe")

for _candidate in _candidates:
    if _candidate and Path(_candidate).exists():
        BASH = _candidate
        break

pytestmark = pytest.mark.skipif(BASH is None, reason="需要 bash")


def _fake_gh(tmp_path, pages):
    """造一个假的 gh：模拟 --paginate（合并所有页）并应用 --jq 过滤。"""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    payload = tmp_path / "pages.json"
    payload.write_text(json.dumps(pages), encoding="utf-8")

    fake = tmp_path / "fake_gh.py"
    fake.write_text(
        "import json, re, sys\n"
        "pages = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "args = sys.argv[2:]\n"
        "filter_text = ''\n"
        "for i, a in enumerate(args):\n"
        "    if a == '--jq' and i + 1 < len(args):\n"
        "        filter_text = args[i + 1]\n"
        "    elif a.startswith('--jq='):\n"
        "        filter_text = a[5:]\n"
        "m = re.search(r'select\\(\\.name == \"([^\"]+)\"\\)', filter_text)\n"
        "target = m.group(1) if m else None\n"
        "out = []\n"
        "for page in pages:\n"
        "    for artifact_name in page:\n"
        "        if target is None or artifact_name == target:\n"
        "            out.append('99999')\n"
        "print('\\n'.join(out))\n",
        encoding="utf-8",
    )

    gh = bindir / "gh"
    gh.write_text(
        "#!/usr/bin/env bash\n"
        f'"{sys.executable}" "{fake}" "{payload}" "$@"\n',
        encoding="utf-8",
    )
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(bindir)


def _run_lookup(tmp_path, pages, artifact_name):
    env = dict(os.environ)
    env["PATH"] = _fake_gh(tmp_path, pages) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [BASH, str(SCRIPT), "xihaoxiao/lianghua", artifact_name],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )
    return proc.stdout.strip()


def test_finds_artifact_on_first_page(tmp_path):
    out = _run_lookup(tmp_path, [["a", "b", "push-results-2026-10-22-intraday"]], "push-results-2026-10-22-intraday")
    assert out, "第一页就能找到"


def test_finds_artifact_beyond_first_page(tmp_path):
    """核心场景：目标 artifact 被无关 artifact 挤到第 2 页及以后。"""
    noise = [f"coverage-py3.{i}" for i in range(30)] + [f"build-{i}.dockerbuild" for i in range(30)]
    pages = [noise[:60], ["push-results-2026-10-22-intraday"] + noise[60:]]

    out = _run_lookup(tmp_path, pages, "push-results-2026-10-22-intraday")
    assert out, "翻页后必须能找到第二页的 artifact"


def test_missing_artifact_returns_empty(tmp_path):
    out = _run_lookup(tmp_path, [["coverage-a", "coverage-b"]], "push-results-2026-10-22-intraday")
    assert out == "", "不存在时应无输出"


def test_script_uses_paginate():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "--paginate" in content, "查找必须翻页"
    assert "per_page=100" in content


def test_workflow_calls_the_script():
    workflow = (REPO_ROOT / ".github" / "workflows" / "daily_push.yml").read_text(encoding="utf-8")
    assert "has_push_artifact.sh" in workflow, "工作流必须调用翻页查找脚本"
    # 不能再有裸的 gh api 查 artifact（会只拿到第一页）
    assert 'gh api "repos/${{ github.repository }}/actions/artifacts"' not in workflow


def test_check_duplicate_job_checks_out_repo():
    """回归防护：check-duplicate 用了仓库里的脚本，必须先 checkout。

    真实故障：没 checkout → bash: scripts/has_push_artifact.sh: No such file or
    directory → 退出码 127 → 整个去重 job 失败 → 定时推送直接挂掉。
    """
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "daily_push.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["check-duplicate"]["steps"]

    uses = [str(step.get("uses", "")) for step in steps]
    assert any(u.startswith("actions/checkout@") for u in uses), (
        "check-duplicate 必须先 checkout 仓库，否则脚本不存在"
    )

    # checkout 必须在调用脚本的那一步之前
    checkout_idx = next(i for i, u in enumerate(uses) if u.startswith("actions/checkout@"))
    script_step_idx = next(
        i for i, step in enumerate(steps) if "has_push_artifact.sh" in str(step.get("run", ""))
    )
    assert checkout_idx < script_step_idx, "checkout 必须在调用脚本之前"
