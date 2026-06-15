"""
GitHub Actions 触发脚本
========================
通过 GitHub API 触发 api_trigger.yml 工作流。
供 cron-job.org 或本地 cron 调用，替代 GitHub Actions 原生 schedule。

用法:
    python trigger_push.py pre-market      # 触发盘前扫描
    python trigger_push.py daily-push      # 触发每日推送
    python trigger_push.py --check         # 检查最近运行状态

环境变量:
    GITHUB_TOKEN: GitHub Personal Access Token (需要 workflow 权限)
    GITHUB_REPO: 仓库名称 (默认: Lishoulan/lianghua)

cron-job.org 配置示例:
    URL:    https://your-server.com/trigger?job=pre-market
    Method: GET
    时间:   每天 06:30 UTC (14:30 北京时间)

    或者直接用本脚本配合系统 cron:
    30 6 * * * cd /path/to/TradingAgents && python trigger_push.py pre-market
    30 10 * * * cd /path/to/TradingAgents && python trigger_push.py daily-push
"""

import sys
import os
import json
import requests
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Lishoulan/lianghua")
WORKFLOW_FILE = "api_trigger.yml"

API_BASE = f"https://api.github.com/repos/{GITHUB_REPO}/actions"


def trigger_workflow(job_type: str) -> bool:
    """触发 GitHub Actions 工作流

    参数:
        job_type: 'pre-market' 或 'daily-push'

    返回:
        是否成功触发
    """
    if not GITHUB_TOKEN:
        print("错误: GITHUB_TOKEN 环境变量未设置")
        print("请创建 GitHub Personal Access Token (需要 workflow 权限)")
        print("  https://github.com/settings/tokens")
        return False

    if job_type not in ("pre-market", "daily-push"):
        print(f"错误: job_type 必须是 'pre-market' 或 'daily-push'，当前: {job_type}")
        return False

    url = f"{API_BASE}/workflows/{WORKFLOW_FILE}/dispatches"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "ref": "main",
        "inputs": {"job_type": job_type},
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        if resp.status_code == 204:
            label = "盘前扫描" if job_type == "pre-market" else "每日推送"
            print(f"✅ {label} 已成功触发 ({datetime.now().strftime('%H:%M:%S')})")
            return True
        else:
            print(f"❌ 触发失败: HTTP {resp.status_code}")
            print(f"   响应: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ 触发异常: {e}")
        return False


def check_recent_runs():
    """检查最近的运行状态"""
    if not GITHUB_TOKEN:
        print("错误: GITHUB_TOKEN 环境变量未设置")
        return

    url = f"{API_BASE}/runs?per_page=5"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"查询失败: HTTP {resp.status_code}")
            return

        runs = resp.json().get("workflow_runs", [])
        if not runs:
            print("最近没有运行记录")
            return

        print(f"最近 {len(runs)} 次运行:")
        print("-" * 80)
        for run in runs:
            status = run["status"]
            conclusion = run.get("conclusion", "运行中")
            created = run["created_at"]
            name = run["name"]
            event = run["event"]
            print(f"  {created}  {name}  [{event}]  状态:{status}  结果:{conclusion}")

    except Exception as e:
        print(f"查询异常: {e}")


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python trigger_push.py pre-market    # 触发盘前扫描")
        print("  python trigger_push.py daily-push    # 触发每日推送")
        print("  python trigger_push.py --check       # 检查运行状态")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "--check":
        check_recent_runs()
    elif cmd in ("pre-market", "daily-push"):
        trigger_workflow(cmd)
    else:
        print(f"未知命令: {cmd}")
        print("可用命令: pre-market, daily-push, --check")
        sys.exit(1)


if __name__ == "__main__":
    main()
