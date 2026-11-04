# Cloudflare Worker 定时触发器

GitHub Actions 的 `schedule` 事件高峰期会延迟 5-30 分钟，导致"推送不准时"。
这个 Worker 用 Cloudflare Cron（通常秒级精度）准时敲一下 GitHub 的
`workflow_dispatch` 接口，由 GitHub 立刻开始跑工作流。

```
Cloudflare Cron (UTC 05:15 / 09:15)
        └──> Worker.scheduled()
                 └──> POST api.github.com/repos/{owner}/{repo}/actions/workflows/daily_push.yml/dispatches
                          └──> GitHub Actions: daily_push.yml (event = workflow_dispatch)
                                   └──> 盘中/盘后扫描 → 微信/Server酱推送
```

GitHub 自带的 6 个 `schedule` 事件**保留**作为兜底：万一 Worker 挂了，
GitHub 那边延迟几十分钟也会把当天的推送补上。工作流内的
`concurrency: daily-push` + artifact 去重保证不会重复推送。

---

## 一、需要的配置

### 环境变量（Worker → Settings → Variables and Secrets）

| 名称 | 示例 | 说明 |
|------|------|------|
| `GITHUB_OWNER` | `xihaoxiao` | 仓库 owner |
| `GITHUB_REPO` | `lianghua` | 仓库名 |
| `WORKFLOW_FILE` | `daily_push.yml` | 工作流文件名（不是完整路径） |
| `GITHUB_REF` | `main` | 触发用的分支 |

### 密钥（Worker → Settings → Variables and Secrets → Encrypt）

| 名称 | 说明 |
|------|------|
| `GH_TOKEN` | GitHub PAT。fine-grained 需勾 **Actions: Read and write**；classic 需 **repo** scope |
| `TRIGGER_TOKEN` | 随机串，用来保护 `/trigger` 端点（可选，不配则该端点禁用） |

> 少配 `GITHUB_OWNER` / `GITHUB_REPO` 是最常见的坑：URL 会拼成
> `.../repos/undefined/undefined/...`，GitHub 直接返回 404。
> `WORKFLOW_FILE` / `GITHUB_REF` 没配时会走默认值，一般没问题。

### Cron Trigger（Worker → Settings → Triggers → Cron Triggers）

```
15 5 * * 1-5     # 05:15 UTC = 13:15 北京时间（盘中）
15 9 * * 1-5     # 09:15 UTC = 17:15 北京时间（盘后）
```

Cloudflare cron **一律按 UTC 解释**，不要填北京时间。

---

## 二、部署

### 方式 A：Dashboard（当前用法）

1. Workers & Pages → 选中 Worker `lianhua` → **Edit code**
2. 用 `src/index.js` 的内容整体替换编辑器里的代码
3. 点 **Deploy**（改完必须点 Deploy 才生效）
4. Settings → Variables and Secrets → 按上表补齐变量和密钥
5. Settings → Triggers → 添加上面两条 Cron
6. Settings → Observability → 打开日志（否则失败信息看不到）

### 方式 B：wrangler CLI（可复现，便于排查）

```bash
cd deploy/cloudflare-scheduler
npm install
npx wrangler login                 # 或设置 CLOUDFLARE_API_TOKEN
npx wrangler secret put GH_TOKEN
npx wrangler secret put TRIGGER_TOKEN
npx wrangler deploy
```

`wrangler.toml` 里的 `name` 必须和 Dashboard 里那个 Worker 一致（`lianhua`），
否则会新建一个 Worker，两边各挂一条 cron → 重复触发。

---

## 三、诊断（出问题时按顺序走）

### 1. `/health` —— 先看环境变量和 GitHub 侧是否通

```
https://lianhua.573183877.workers.dev/health
```

只读，不会触发任何工作流。返回里几个关键字段：

| 字段 | 含义 |
|------|------|
| `missing` | 缺少的变量；非空说明配置有问题 |
| `github.token.ok` / `.login` | Token 是否有效；401 = Token 过期或 PAT 填错 |
| `github.workflow.ok` / `.state` | 工作流文件是否在默认分支上；404 = 文件名或仓库不对 |
| `github.recent_runs.items` | 最近 5 次运行的 `event`：`workflow_dispatch` = Cloudflare 触发成功，`schedule` = GitHub 自己的定时（会延迟） |

### 2. `/trigger` —— 立刻验证整条链路

```
curl -X POST "https://lianhua.573183877.workers.dev/trigger?token=<TRIGGER_TOKEN>"
```

返回 `{"ok":true,...}` 就说明 Cloudflare → GitHub 这段是通的；
然后去 GitHub Actions 页面确认出现了 `event = workflow_dispatch` 的运行记录。

### 3. Worker 日志 —— 看定时触发时到底发生了什么

Dashboard → Worker `lianhua` → **Logs**（Live 可实时看），或：

```bash
npx wrangler tail
```

关键日志字段：

```json
{"ts":"...","level":"info","msg":"cron fired","cron":"15 5 * * 1-5","beijing":"..."}
{"ts":"...","level":"info","msg":"workflow dispatched","status":204,"elapsed_ms":412}
{"ts":"...","level":"error","msg":"dispatch failed","status":403,"error":"GitHub dispatch failed: HTTP 403 ..."}
```

### 4. GitHub 侧确认

```bash
curl -H "Authorization: Bearer $GH_TOKEN" \
  "https://api.github.com/repos/xihaoxiao/lianghua/actions/workflows/daily_push.yml/runs?per_page=5"
```

`event` 字段是 `workflow_dispatch` 就说明是 Cloudflare 触发的，
是 `schedule` 就是 GitHub 自己的定时任务。

---

## 四、故障对照表

| 现象 | 原因 | 处理 |
|------|------|------|
| `/` 正常但到点没动静 | `scheduled` 抛错且没看日志 | 看 Logs / `wrangler tail` |
| 日志 `missing env vars` | 变量没配全 | 按上表补 `GITHUB_OWNER` / `GITHUB_REPO` |
| 日志 `HTTP 401` | PAT 无效/过期 | 重新生成 PAT 并 `secret put` |
| 日志 `HTTP 403` | PAT 缺 Actions:write / repo scope | 换 token |
| 日志 `HTTP 404` | 仓库名或工作流文件名错 | 检查 `GITHUB_OWNER`/`GITHUB_REPO`/`WORKFLOW_FILE` |
| 日志 `HTTP 422` | 传了工作流里不存在的 input | 代码里默认不传 inputs |
| 有运行但没推送 | 工作流内 artifact 去重判定当天已推过 | 看 `check-duplicate` job 输出 |
| 推送时间还是不准 | Cron 表达式按 UTC 解释 | 05:15 UTC = 13:15 北京 |
| GitHub 上没看到运行 | `workflow_dispatch` 要求工作流文件在默认分支 | 确认 `daily_push.yml` 已在 main |
