/**
 * Cloudflare Worker —— GitHub Actions 定时触发器
 *
 * 作用：Cloudflare Cron（准时）→ 调用 GitHub workflow_dispatch → daily_push.yml 推送。
 * GitHub 自带的 schedule 事件会延迟 5-30 分钟，这里用外部调度器解决"推送不准时"。
 *
 * 暴露端点：
 *   GET  /         存活探针（纯文本），确认 Worker 在跑
 *   GET  /health   诊断端点：环境变量检查 + 只读校验 GitHub Token / 工作流 / 最近运行记录
 *                  （不会触发工作流，可放心反复访问）
 *   POST /trigger  手动触发一次工作流（需 TRIGGER_TOKEN 保护），用来立刻验证整条链路
 *
 * 触发链路的每一步都会打结构化日志，Dashboard → lianhua → Logs 可见。
 */

const GH_API = "https://api.github.com";
const GH_API_VERSION = "2022-11-28";
const USER_AGENT = "lianghua-cloudflare-scheduler";

// 缺任何一个都会导致 dispatch 失败，/health 会直接点名
const REQUIRED_VARS = ["GH_TOKEN", "GITHUB_OWNER", "GITHUB_REPO"];

// ─────────────────────────────────────────────────────────────
//  基础工具
// ─────────────────────────────────────────────────────────────

function ts() {
  return new Date().toISOString();
}

function beijingNow() {
  return new Date(Date.now() + 8 * 3600 * 1000).toISOString().replace("T", " ").slice(0, 19) + " +08:00";
}

function log(level, msg, extra = {}) {
  console.log(JSON.stringify({ ts: ts(), level, msg, ...extra }));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

// 常量时间比较，避免 /trigger 的 token 被计时攻击
function safeEqual(a, b) {
  const s1 = String(a ?? "");
  const s2 = String(b ?? "");
  if (s1.length !== s2.length) return false;
  let diff = 0;
  for (let i = 0; i < s1.length; i += 1) {
    diff |= s1.charCodeAt(i) ^ s2.charCodeAt(i);
  }
  return diff === 0;
}

// ─────────────────────────────────────────────────────────────
//  配置
// ─────────────────────────────────────────────────────────────

function repoSlug(env) {
  return `${env.GITHUB_OWNER}/${env.GITHUB_REPO}`;
}

function workflowFile(env) {
  return env.WORKFLOW_FILE || "daily_push.yml";
}

function missingVars(env) {
  return REQUIRED_VARS.filter((key) => !env[key] || String(env[key]).trim() === "");
}

function ghHeaders(env, extra = {}) {
  return {
    Authorization: `Bearer ${env.GH_TOKEN}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": GH_API_VERSION,
    "User-Agent": USER_AGENT,
    ...extra,
  };
}

// ─────────────────────────────────────────────────────────────
//  GitHub 调用
// ─────────────────────────────────────────────────────────────

/**
 * 触发 workflow_dispatch。
 * 成功返回 204（部分 API 版本返回 200），失败抛出带 status / body 的错误。
 */
async function dispatchWorkflow(env, inputs) {
  const file = workflowFile(env);
  const ref = env.GITHUB_REF || "main";
  const url = `${GH_API}/repos/${repoSlug(env)}/actions/workflows/${file}/dispatches`;

  const payload = { ref };
  // 只有显式指定时才带 inputs，否则沿用工作流文件里的默认值（避免输入项改名后 422）
  if (inputs && Object.keys(inputs).length > 0) payload.inputs = inputs;

  const started = Date.now();
  const response = await fetch(url, {
    method: "POST",
    headers: ghHeaders(env, { "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  const body = await response.text();
  const elapsed = Date.now() - started;

  if (!response.ok) {
    const error = new Error(`GitHub dispatch failed: HTTP ${response.status} ${body.slice(0, 500)}`);
    error.status = response.status;
    error.body = body;
    throw error;
  }

  return { status: response.status, elapsed_ms: elapsed, repo: repoSlug(env), workflow: file, ref };
}

/**
 * 只读诊断：Token 是否有效、工作流是否存在、最近几次运行是什么事件触发的。
 * 不产生任何副作用。
 */
async function describeGithub(env) {
  const result = {};
  const file = workflowFile(env);

  // 1) Token 有效性
  try {
    const res = await fetch(`${GH_API}/user`, { headers: ghHeaders(env) });
    result.token = { ok: res.ok, status: res.status };
    if (res.ok) {
      const body = await res.json();
      result.token.login = body.login;
    } else {
      result.token.error = (await res.text()).slice(0, 300);
    }
  } catch (err) {
    result.token = { ok: false, error: String(err && err.message ? err.message : err) };
  }

  // 2) 工作流是否存在（文件必须在默认分支上）
  try {
    const res = await fetch(`${GH_API}/repos/${repoSlug(env)}/actions/workflows/${file}`, {
      headers: ghHeaders(env),
    });
    result.workflow = { ok: res.ok, status: res.status, file };
    if (res.ok) {
      const body = await res.json();
      result.workflow.name = body.name;
      result.workflow.state = body.state;
      result.workflow.id = body.id;
    } else {
      result.workflow.error = (await res.text()).slice(0, 300);
    }
  } catch (err) {
    result.workflow = { ok: false, error: String(err && err.message ? err.message : err) };
  }

  // 3) 最近运行：event=workflow_dispatch 说明 Cloudflare 触发成功，
  //    event=schedule 说明是 GitHub 自己的定时任务（可能延迟）
  try {
    const res = await fetch(
      `${GH_API}/repos/${repoSlug(env)}/actions/workflows/${file}/runs?per_page=5`,
      { headers: ghHeaders(env) },
    );
    result.recent_runs = { ok: res.ok, status: res.status };
    if (res.ok) {
      const body = await res.json();
      result.recent_runs.items = (body.workflow_runs || []).map((run) => ({
        id: run.id,
        event: run.event,
        status: run.status,
        conclusion: run.conclusion,
        created_at: run.created_at,
      }));
    } else {
      result.recent_runs.error = (await res.text()).slice(0, 300);
    }
  } catch (err) {
    result.recent_runs = { ok: false, error: String(err && err.message ? err.message : err) };
  }

  return result;
}

// ─────────────────────────────────────────────────────────────
//  事件处理器
// ─────────────────────────────────────────────────────────────

export default {
  async scheduled(controller, env) {
    const cron = (controller && controller.cron) || "unknown";
    log("info", "cron fired", { cron, utc: ts(), beijing: beijingNow() });

    const missing = missingVars(env);
    if (missing.length > 0) {
      log("error", "config error: missing env vars", { cron, missing });
      throw new Error(`missing environment variables: ${missing.join(", ")}`);
    }

    try {
      const dispatched = await dispatchWorkflow(env);
      log("info", "workflow dispatched", { cron, ...dispatched });
    } catch (err) {
      // 网络抖动 / GitHub 5xx：重试一次；4xx 属于配置或权限问题，重试没有意义
      const status = err && err.status;
      if (!status || status >= 500) {
        log("warn", "dispatch failed, retrying in 5s", { cron, status: status ?? null, error: err.message });
        await sleep(5000);
        try {
          const dispatched = await dispatchWorkflow(env);
          log("info", "workflow dispatched on retry", { cron, ...dispatched });
          return;
        } catch (retryErr) {
          log("error", "dispatch failed after retry", {
            cron,
            status: retryErr.status ?? null,
            error: retryErr.message,
          });
          throw retryErr;
        }
      }
      log("error", "dispatch failed", { cron, status: status ?? null, error: err.message });
      throw err;
    }
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";

    // 存活探针
    if (path === "/") {
      return new Response("lianghua scheduler is running\n", {
        status: 200,
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Cache-Control": "no-store",
        },
      });
    }

    // 诊断端点：只读，不触发任何工作流
    if (path === "/health") {
      const missing = missingVars(env);
      const payload = {
        ok: missing.length === 0,
        checked_at: ts(),
        beijing: beijingNow(),
        env: {
          GITHUB_OWNER: env.GITHUB_OWNER || null,
          GITHUB_REPO: env.GITHUB_REPO || null,
          WORKFLOW_FILE: workflowFile(env),
          GITHUB_REF: env.GITHUB_REF || "main",
          GH_TOKEN: env.GH_TOKEN ? `set(len=${env.GH_TOKEN.length})` : null,
          TRIGGER_TOKEN: env.TRIGGER_TOKEN ? "set" : null,
        },
        missing,
        github: missing.length === 0 ? await describeGithub(env) : { skipped: "missing env vars" },
      };
      return json(payload, payload.ok ? 200 : 500);
    }

    // 手动触发：用来立刻验证 Cloudflare → GitHub 的整条链路
    if (path === "/trigger") {
      if (!env.TRIGGER_TOKEN) {
        return json({ ok: false, error: "TRIGGER_TOKEN not configured; /trigger is disabled" }, 403);
      }
      const provided = url.searchParams.get("token") || request.headers.get("x-trigger-token") || "";
      if (!safeEqual(provided, env.TRIGGER_TOKEN)) {
        return json({ ok: false, error: "invalid token" }, 401);
      }
      const missing = missingVars(env);
      if (missing.length > 0) {
        return json({ ok: false, error: `missing env vars: ${missing.join(", ")}` }, 500);
      }
      const dryRun = url.searchParams.get("dry_run") === "true";
      try {
        const dispatched = await dispatchWorkflow(env, dryRun ? { dry_run: "true" } : undefined);
        log("info", "manual trigger", { ...dispatched, dry_run: dryRun });
        return json({ ok: true, dispatched, dry_run: dryRun, checked_at: ts() });
      } catch (err) {
        log("error", "manual trigger failed", { status: err.status ?? null, error: err.message });
        return json({ ok: false, error: err.message, status: err.status ?? null }, 502);
      }
    }

    return json({ ok: false, error: "not found", paths: ["/", "/health", "/trigger"] }, 404);
  },
};
