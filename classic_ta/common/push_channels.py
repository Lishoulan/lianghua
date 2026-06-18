"""
推送通道模块 — Server酱微信推送（高可用版）

从 v63_daily_push.py 提取的公共推送逻辑，供各版本推送脚本复用。
增强功能：
  - 3次重试（指数退避）
  - 定时投递降级为立即发送（宁可早到也不丢失）
  - pushid记录 + 推送结果校验
  - 内容长度预检（Server酱限制64KB）
  - 全链路 print(flush=True) 确保 GitHub Actions 日志可见
"""
import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
#  推送配置
# ══════════════════════════════════════════════════════════

# 推送分组配置
SERVERCHAN_KEYS_ADMIN = [k.strip() for k in os.getenv("SERVERCHAN_KEY", "").split(",") if k.strip()]
SERVERCHAN_KEYS_BETA = [k.strip() for k in os.getenv("SERVERCHAN_KEY_BETA", "").split(",") if k.strip()]

# 兼容旧变量名
SERVERCHAN_KEYS = SERVERCHAN_KEYS_ADMIN

# Server酱内容长度限制（保守值，实际限制约64KB）
_MAX_CONTENT_LENGTH = 60000

# 重试配置
_MAX_RETRIES = 3                  # 最大重试次数
_RETRY_BACKOFF_BASE = 3           # 退避基数（秒），实际等待 = base * 2^attempt
_REQUEST_TIMEOUT = 30             # 单次请求超时（秒）
_DEGRADATION_RETRIES = 2          # 降级（立即发送）重试次数

# 最近一次推送的 pushid 记录（用于状态查询）
_last_push_ids = []

# 复用连接池（减少TCP握手开销）
_session = None


def _get_session():
    """获取/创建 requests Session（连接池复用）"""
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
        # 配置连接池：最大重试由上层控制，这里仅处理传输层重试
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(
            pool_connections=2,
            pool_maxsize=4,
            max_retries=0,  # 传输层重试由上层逻辑控制
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


# ══════════════════════════════════════════════════════════
#  核心发送
# ══════════════════════════════════════════════════════════

def _truncate_content(desp):
    """内容长度预检，超长则截断（保留头部核心信息）"""
    if len(desp) <= _MAX_CONTENT_LENGTH:
        return desp, False
    truncated = desp[:_MAX_CONTENT_LENGTH - 200]
    truncated += "\n\n---\n⚠️ 内容过长已截断，完整信息请查看管理员推送"
    print(f"  ⚠️ 推送内容超长({len(desp)}字)，已截断至{len(truncated)}字", flush=True)
    return truncated, True


def _do_send(key, title, desp, scheduled=None):
    """执行单次Server酱推送请求

    错误分类：
      - timeout: 请求超时（网络问题，值得重试）
      - network: 连接错误（网络问题，值得重试）
      - api_error: API返回非0 code（参数/配额问题，重试意义不大）
      - http_error: 非200状态码（服务端问题，值得重试）

    Returns:
        tuple: (success: bool, pushid: str or None, error: str or None, error_type: str)
    """
    import requests
    session = _get_session()
    try:
        url = f"https://sctapi.ftqq.com/{key}.send"
        data = {"title": title, "desp": desp}
        if scheduled:
            data["scheduled"] = scheduled
        resp = session.post(url, data=data, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("code") == 0:
                pushid = result.get("data", {}).get("pushid", "")
                return True, pushid, None, None
            else:
                return False, None, f"API错误: {result}", "api_error"
        else:
            return False, None, f"HTTP {resp.status_code}: {resp.text[:200]}", "http_error"
    except requests.exceptions.Timeout:
        return False, None, f"请求超时({_REQUEST_TIMEOUT}s)", "timeout"
    except requests.exceptions.ConnectionError as e:
        return False, None, f"连接错误: {e}", "network"
    except Exception as e:
        return False, None, f"未知异常: {e}", "unknown"


def _send_with_retry(key, title, desp, scheduled=None, max_retries=_MAX_RETRIES):
    """带指数退避的重试发送

    Args:
        key: Server酱Key
        title: 推送标题
        desp: 推送内容
        scheduled: 定时发送时间（可选）
        max_retries: 最大重试次数

    Returns:
        tuple: (success: bool, pushid: str or None, attempts: int)
    """
    for attempt in range(max_retries):
        ok, pushid, err, err_type = _do_send(key, title, desp, scheduled=scheduled)
        if ok:
            return True, pushid, attempt + 1

        # api_error 类型的错误重试意义不大（参数/配额问题），但仍尝试一次
        if err_type == "api_error" and attempt > 0:
            print(f"    ❌ API错误，停止重试: {err}", flush=True)
            return False, None, attempt + 1

        if attempt < max_retries - 1:
            wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
            print(f"    ⏳ 第{attempt+1}次失败({err_type})，{wait}秒后重试...", flush=True)
            time.sleep(wait)
        else:
            print(f"    ❌ 第{attempt+1}次失败({err_type}): {err}", flush=True)

    return False, None, max_retries


# ══════════════════════════════════════════════════════════
#  对外接口
# ══════════════════════════════════════════════════════════

def send_serverchan(title, desp, keys=None, scheduled=None):
    """Server酱推送，支持指定key列表（分组推送）和定时发送

    完整保障链路：
      1. 每个Key最多重试3次（指数退避 3s → 6s → 12s）
      2. 定时投递全部失败 → 降级为立即发送（再重试2次）
      3. 内容超长自动截断
      4. 全链路 print(flush=True) 确保 GitHub Actions 日志可见

    Args:
        title: 推送标题
        desp: 推送内容（Markdown格式）
        keys: Server酱Key列表，默认使用管理员组
        scheduled: 定时发送时间，格式 "YYYY-MM-DD HH:MM:SS"（北京时间）
                   消息会缓存在Server酱服务器，到指定时间才投递到微信

    Returns:
        bool: 是否至少有一个Key推送成功
    """
    global _last_push_ids

    if keys is None:
        keys = SERVERCHAN_KEYS_ADMIN
    if not keys:
        print("  ⚠️ Server酱推送: 无可用Key，跳过", flush=True)
        return False

    # 内容长度预检
    desp, was_truncated = _truncate_content(desp)

    success_count = 0
    push_ids = []

    for i, key in enumerate(keys):
        key_label = f"Key[{i+1}/{len(keys)}]"
        print(f"  📤 {key_label} 发送中...", flush=True)

        # ── 阶段1：定时投递（或立即发送）──
        ok, pushid, attempts = _send_with_retry(key, title, desp, scheduled=scheduled)
        if ok:
            success_count += 1
            if pushid:
                push_ids.append(pushid)
            if scheduled:
                print(f"  ✅ {key_label} 定时投递成功: {scheduled} (pushid={pushid}, 尝试{attempts}次)", flush=True)
            else:
                print(f"  ✅ {key_label} 立即发送成功 (pushid={pushid}, 尝试{attempts}次)", flush=True)
            continue

        print(f"  ❌ {key_label} 主投递失败（已尝试{attempts}次）", flush=True)

        # ── 阶段2：降级策略（定时投递失败 → 立即发送）──
        if scheduled:
            print(f"  🔄 {key_label} 降级为立即发送（宁可早到也不丢失）", flush=True)
            ok, pushid, attempts = _send_with_retry(
                key, title, desp, scheduled=None, max_retries=_DEGRADATION_RETRIES
            )
            if ok:
                success_count += 1
                if pushid:
                    push_ids.append(pushid)
                print(f"  ✅ {key_label} 降级立即发送成功 (pushid={pushid}, 尝试{attempts}次)", flush=True)
            else:
                print(f"  ❌ {key_label} 彻底失败（降级也失败，共{attempts + _MAX_RETRIES}次尝试）", flush=True)

    _last_push_ids = push_ids
    result_emoji = "✅" if success_count > 0 else "❌"
    print(f"  {result_emoji} Server酱推送结果: {success_count}/{len(keys)} 个Key成功", flush=True)
    return success_count > 0


def get_last_push_ids():
    """获取最近一次推送的 pushid 列表"""
    return _last_push_ids.copy()


def send_group_push(admin_title, admin_desp, beta_title, beta_desp, scheduled=None):
    """分组推送：管理员组（完整技术版）+ 内测组（精简卡片版）

    保障策略：
      - 管理员组优先推送，确保核心通道畅通
      - 内测组失败不影响管理员组结果
      - 两组推送独立重试，互不干扰

    Args:
        admin_title: 管理员组标题
        admin_desp: 管理员组内容
        beta_title: 内测组标题
        beta_desp: 内测组内容
        scheduled: 定时发送时间，格式 "YYYY-MM-DD HH:MM:SS"（北京时间）

    Returns:
        dict: {"admin": bool, "beta": bool}
    """
    results = {}

    # 管理员组推送（优先）
    print(f"\n{'─'*40}", flush=True)
    if SERVERCHAN_KEYS_ADMIN:
        print(f"📢 推送管理员组 ({len(SERVERCHAN_KEYS_ADMIN)}个Key)...", flush=True)
        results["admin"] = send_serverchan(admin_title, admin_desp, keys=SERVERCHAN_KEYS_ADMIN, scheduled=scheduled)
    else:
        print("⚠️ 管理员组: 无Key配置，跳过", flush=True)
        results["admin"] = False

    # 内测组推送
    print(f"{'─'*40}", flush=True)
    if SERVERCHAN_KEYS_BETA:
        print(f"📢 推送内测组 ({len(SERVERCHAN_KEYS_BETA)}个Key)...", flush=True)
        results["beta"] = send_serverchan(beta_title, beta_desp, keys=SERVERCHAN_KEYS_BETA, scheduled=scheduled)
    else:
        print("⚠️ 内测组: 无Key配置，跳过", flush=True)
        results["beta"] = False

    # 汇总
    print(f"{'─'*40}", flush=True)
    admin_status = "✅成功" if results.get("admin") else "❌失败"
    beta_status = "✅成功" if results.get("beta") else "❌失败"
    print(f"📊 推送汇总: 管理员组={admin_status} | 内测组={beta_status}", flush=True)

    return results
