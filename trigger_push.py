"""
推送触发器
通过 GitHub API 触发 daily_push.yml 工作流，或本地直接运行。

用法:
    python trigger_push.py                    # 触发GitHub Actions
    python trigger_push.py --local            # 本地直接运行
    python trigger_push.py --local --dry-run  # 本地干跑
    python trigger_push.py --check            # 检查最近运行状态

环境变量:
    GITHUB_TOKEN: GitHub Personal Access Token (需要 workflow 权限)
    GITHUB_REPO: 仓库名称 (默认: Lishoulan/lianghua)
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Lishoulan/lianghua")
WORKFLOW_FILE = "daily_push.yml"

API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}/actions"


def trigger_workflow(dry_run=False) -> bool:
    """通过 GitHub API 触发推送工作流"""
    if not GITHUB_TOKEN:
        print("错误: GITHUB_TOKEN 环境变量未设置")
        print("请创建 GitHub Personal Access Token (需要 workflow 权限)")
        return False

    import requests

    url = f"{API_BASE}/workflows/{WORKFLOW_FILE}/dispatches"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "ref": "main",
        "inputs": {"dry_run": str(dry_run).lower()},
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        if resp.status_code == 204:
            print(f"推送工作流已触发 {'(干跑模式)' if dry_run else ''} ({datetime.now().strftime('%H:%M:%S')})")
            return True
        else:
            print(f"触发失败: HTTP {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"触发失败: {e}")
        return False


def check_status():
    """检查最近一次推送工作流的运行状态"""
    if not GITHUB_TOKEN:
        print("错误: GITHUB_TOKEN 环境变量未设置")
        return

    import requests

    url = f"{API_BASE}/workflows/{WORKFLOW_FILE}/runs?per_page=5"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            runs = resp.json().get("workflow_runs", [])
            if not runs:
                print("暂无运行记录")
                return
            for run in runs:
                status = run["status"]
                conclusion = run.get("conclusion", "—")
                created = run["created_at"]
                print(f"  {created} | {status} | {conclusion}")
        else:
            print(f"查询失败: HTTP {resp.status_code}")
    except Exception as e:
        print(f"查询失败: {e}")


def run_local(dry_run=False):
    """本地直接运行推送"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from classic_ta.daily_push import daily_push
    daily_push()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="推送触发器")
    parser.add_argument("--local", action="store_true", help="本地直接运行（不触发GitHub Actions）")
    parser.add_argument("--dry-run", action="store_true", help="干跑模式（不实际推送）")
    parser.add_argument("--check", action="store_true", help="检查最近运行状态")
    args = parser.parse_args()

    if args.check:
        check_status()
    elif args.local:
        run_local(dry_run=args.dry_run)
    else:
        trigger_workflow(dry_run=args.dry_run)
