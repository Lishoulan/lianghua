/**
 * Worker 冒烟测试（零依赖，Node 内置 test runner）
 *
 *   cd deploy/cloudflare-scheduler && npm test
 *
 * 覆盖：存活探针、/health 诊断、/trigger 鉴权与触发、cron 触发、
 * 缺变量时的报错、GitHub 5xx 时的重试。
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import worker from "../src/index.js";

const realFetch = globalThis.fetch;

function mockGithub({ dispatchStatus = 204, dispatchBody = "" } = {}) {
  const calls = [];
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), method: init.method || "GET" });
    const u = String(url);
    if (u.endsWith("/dispatches")) {
      // 204 是 null-body 状态码，不能带 body
      return dispatchStatus === 204
        ? new Response(null, { status: 204 })
        : new Response(dispatchBody, { status: dispatchStatus });
    }
    if (u.endsWith("/user")) {
      return Response.json({ login: "octocat" });
    }
    if (u.includes("/actions/workflows/") && u.includes("/runs")) {
      return Response.json({
        workflow_runs: [
          {
            id: 1,
            event: "workflow_dispatch",
            status: "completed",
            conclusion: "success",
            created_at: "2026-10-21T05:15:00Z",
          },
        ],
      });
    }
    if (u.includes("/actions/workflows/")) {
      return Response.json({ id: 9, name: "Daily Push", state: "active" });
    }
    return new Response("not found", { status: 404 });
  };
  return calls;
}

const ENV = {
  GH_TOKEN: "ghp_test_token",
  GITHUB_OWNER: "xihaoxiao",
  GITHUB_REPO: "lianghua",
  WORKFLOW_FILE: "daily_push.yml",
  GITHUB_REF: "main",
  TRIGGER_TOKEN: "s3cret",
};

test.afterEach(() => {
  globalThis.fetch = realFetch;
});

test("GET / 返回存活文本", async () => {
  globalThis.fetch = async () => {
    throw new Error("不应该发起网络请求");
  };
  const res = await worker.fetch(new Request("https://x.workers.dev/"), ENV);
  assert.equal(res.status, 200);
  assert.match(await res.text(), /lianghua scheduler is running/);
});

test("GET /health 汇报变量与 GitHub 连通性", async () => {
  mockGithub();
  const res = await worker.fetch(new Request("https://x.workers.dev/health"), ENV);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.ok, true);
  assert.deepEqual(body.missing, []);
  assert.equal(body.env.GITHUB_OWNER, "xihaoxiao");
  assert.match(body.env.GH_TOKEN, /^set\(len=\d+\)$/);
  assert.equal(body.github.token.ok, true);
  assert.equal(body.github.token.login, "octocat");
  assert.equal(body.github.workflow.ok, true);
  assert.equal(body.github.recent_runs.items[0].event, "workflow_dispatch");
});

test("GET /health 点名缺失的变量", async () => {
  globalThis.fetch = async () => {
    throw new Error("不应该发起网络请求");
  };
  const res = await worker.fetch(new Request("https://x.workers.dev/health"), { ...ENV, GITHUB_REPO: "" });
  assert.equal(res.status, 500);
  const body = await res.json();
  assert.equal(body.ok, false);
  assert.deepEqual(body.missing, ["GITHUB_REPO"]);
});

test("POST /trigger 拒绝错误 token", async () => {
  globalThis.fetch = async () => {
    throw new Error("不应该发起网络请求");
  };
  const res = await worker.fetch(new Request("https://x.workers.dev/trigger?token=wrong"), ENV);
  assert.equal(res.status, 401);
});

test("POST /trigger 触发工作流", async () => {
  const calls = mockGithub();
  const res = await worker.fetch(new Request("https://x.workers.dev/trigger?token=s3cret", { method: "POST" }), ENV);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.ok, true);
  assert.equal(body.dispatched.status, 204);
  const call = calls.find((c) => c.url.endsWith("/dispatches"));
  assert.equal(call.method, "POST");
  assert.equal(
    call.url,
    "https://api.github.com/repos/xihaoxiao/lianghua/actions/workflows/daily_push.yml/dispatches",
  );
});

test("POST /trigger 把 GitHub 错误透出", async () => {
  mockGithub({ dispatchStatus: 403, dispatchBody: "Resource not accessible by personal access token" });
  const res = await worker.fetch(new Request("https://x.workers.dev/trigger?token=s3cret", { method: "POST" }), ENV);
  assert.equal(res.status, 502);
  const body = await res.json();
  assert.equal(body.ok, false);
  assert.match(body.error, /HTTP 403/);
});

test("scheduled 在 cron 到点时触发工作流", async () => {
  const calls = mockGithub();
  await worker.scheduled({ cron: "15 5 * * 1-5" }, ENV);
  assert.equal(calls.filter((c) => c.url.endsWith("/dispatches")).length, 1);
});

test("scheduled 缺变量时抛出明确错误", async () => {
  globalThis.fetch = async () => {
    throw new Error("不应该发起网络请求");
  };
  await assert.rejects(
    () => worker.scheduled({ cron: "15 5 * * 1-5" }, { ...ENV, GITHUB_OWNER: "" }),
    /missing environment variables: GITHUB_OWNER/,
  );
});

test("scheduled 遇到 5xx 会重试一次再报错", async () => {
  let dispatchCalls = 0;
  globalThis.fetch = async (url) => {
    if (String(url).endsWith("/dispatches")) {
      dispatchCalls += 1;
      return new Response("boom", { status: 502 });
    }
    return new Response("{}", { status: 200 });
  };
  await assert.rejects(() => worker.scheduled({ cron: "15 5 * * 1-5" }, ENV), /HTTP 502/);
  assert.equal(dispatchCalls, 2);
});
