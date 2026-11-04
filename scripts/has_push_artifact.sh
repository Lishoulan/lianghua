#!/usr/bin/env bash
# 检查某个 artifact 名称是否已存在（用于"今天这个时段是否已经推过"的去重判断）。
#
# 为什么要翻页：gh api 默认只返回第一页（30 条），而仓库里每次 CI 都会产生
# coverage-* / *.dockerbuild 等无关 artifact，一天就能把首页挤满，
# 今天的 push-results artifact 被挤出首页 → 查不到 → 去重失效 → 重复推送。
# --paginate 翻所有页，head -1 命中即停。
#
# 用法: has_push_artifact.sh <owner/repo> <artifact-name>
# 输出: 找到则打印 artifact id，未找到则无输出，退出码始终为 0
set -uo pipefail

REPO="${1:-}"
NAME="${2:-}"

if [ -z "$REPO" ] || [ -z "$NAME" ]; then
    echo "用法: $0 <owner/repo> <artifact-name>" >&2
    exit 2
fi

gh api --paginate "repos/${REPO}/actions/artifacts?per_page=100" \
    --jq ".artifacts[] | select(.name == \"${NAME}\") | .id" \
    2>/dev/null | head -1 || true
