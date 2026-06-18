"""
订阅墙中间件
===========================
管理用户订阅状态、计划升级、过期检查、权限校验

订阅计划：
  - trial:     7天免费试用（关注后自动开通）
  - free:      免费版（仅市场报告，无信号）
  - monthly:   月度订阅
  - quarterly: 季度订阅
  - yearly:    年度订阅
  - expired:   已过期

价格体系：
  - monthly:   29元/月
  - quarterly: 79元/季（省8元）
  - yearly:    199元/年（省149元）
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)

# ── 订阅计划配置 ──
PLAN_CONFIG: Dict[str, Dict[str, Any]] = {
    "trial":     {"name": "免费试用", "duration_days": 7,  "price": 0,   "has_signals": True},
    "free":      {"name": "免费版",   "duration_days": None, "price": 0,   "has_signals": False},
    "monthly":   {"name": "月度订阅", "duration_days": 30, "price": 29,  "has_signals": True},
    "quarterly": {"name": "季度订阅", "duration_days": 90, "price": 79,  "has_signals": True},
    "yearly":    {"name": "年度订阅", "duration_days": 365,"price": 199, "has_signals": True},
    "expired":   {"name": "已过期",   "duration_days": 0,  "price": 0,   "has_signals": False},
}

# ── Supabase 配置 ──
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def _supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _check_supabase() -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.debug("Supabase 未配置，订阅功能降级")
        return False
    return True


# ══════════════════════════════════════════════════════════
#  订阅状态查询
# ══════════════════════════════════════════════════════════

def get_user_subscription(open_id: str) -> Dict[str, Any]:
    """
    获取用户订阅状态

    返回:
      {
        "open_id": "...",
        "plan": "trial",
        "plan_name": "免费试用",
        "is_active": True,
        "has_signals": True,
        "plan_expire": "2026-06-25T...",
        "days_remaining": 5,
        "is_subscribed": True
      }
    """
    if not _check_supabase():
        # 降级模式：Supabase 未配置时，所有用户视为试用
        return {
            "open_id": open_id,
            "plan": "trial",
            "plan_name": "免费试用（降级）",
            "is_active": True,
            "has_signals": True,
            "plan_expire": None,
            "days_remaining": None,
            "is_subscribed": True,
        }

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/users?open_id=eq.{open_id}&select=*",
            headers=_supabase_headers(),
            timeout=10,
        )
        users = resp.json()
        if not users:
            return {
                "open_id": open_id,
                "plan": "unknown",
                "plan_name": "未知",
                "is_active": False,
                "has_signals": False,
                "plan_expire": None,
                "days_remaining": 0,
                "is_subscribed": False,
            }

        user = users[0]
        plan = user.get("plan", "trial")
        plan_expire_str = user.get("plan_expire")
        plan_expire = None
        days_remaining = None
        is_active = False

        if plan_expire_str:
            try:
                plan_expire = datetime.fromisoformat(plan_expire_str.replace("Z", "+00:00"))
                now = datetime.now(plan_expire.tzinfo)
                days_remaining = max(0, (plan_expire - now).days)
                is_active = plan_expire > now and user.get("is_subscribed", True)
            except (ValueError, TypeError):
                is_active = user.get("is_subscribed", False)
        else:
            is_active = user.get("is_subscribed", False)

        config = PLAN_CONFIG.get(plan, PLAN_CONFIG["expired"])

        return {
            "open_id": open_id,
            "plan": plan,
            "plan_name": config["name"],
            "is_active": is_active,
            "has_signals": config["has_signals"] and is_active,
            "plan_expire": plan_expire_str,
            "days_remaining": days_remaining,
            "is_subscribed": user.get("is_subscribed", False),
        }

    except Exception as e:
        logger.error(f"获取用户订阅状态失败: {e}")
        return {
            "open_id": open_id,
            "plan": "error",
            "plan_name": "查询失败",
            "is_active": False,
            "has_signals": False,
            "plan_expire": None,
            "days_remaining": 0,
            "is_subscribed": False,
        }


def check_push_permission(open_id: str) -> bool:
    """检查用户是否有权限接收信号推送"""
    sub = get_user_subscription(open_id)
    return sub["has_signals"]


# ══════════════════════════════════════════════════════════
#  订阅计划管理
# ══════════════════════════════════════════════════════════

def start_trial(open_id: str, nickname: str = "") -> bool:
    """为新用户开通7天免费试用"""
    if not _check_supabase():
        return False

    now = datetime.utcnow()
    trial_end = now + timedelta(days=PLAN_CONFIG["trial"]["duration_days"])

    try:
        # 先查询是否已存在
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/users?open_id=eq.{open_id}&select=*",
            headers=_supabase_headers(),
            timeout=10,
        )
        existing = resp.json()

        if existing and len(existing) > 0:
            # 已存在用户，如果是 expired 或 free 则重新开通试用
            user = existing[0]
            if user.get("plan") in ("expired", "free", "trial"):
                requests.patch(
                    f"{SUPABASE_URL}/rest/v1/users?open_id=eq.{open_id}",
                    headers=_supabase_headers(),
                    json={
                        "plan": "trial",
                        "trial_start": now.isoformat(),
                        "trial_end": trial_end.isoformat(),
                        "plan_start": now.isoformat(),
                        "plan_expire": trial_end.isoformat(),
                        "is_subscribed": True,
                    },
                    timeout=10,
                )
                _log_event(open_id, "trial_start", None, "trial", note="重新开通试用")
            return True
        else:
            # 新用户
            data = {
                "open_id": open_id,
                "nickname": nickname,
                "is_subscribed": True,
                "plan": "trial",
                "trial_start": now.isoformat(),
                "trial_end": trial_end.isoformat(),
                "plan_start": now.isoformat(),
                "plan_expire": trial_end.isoformat(),
            }
            requests.post(
                f"{SUPABASE_URL}/rest/v1/users",
                headers=_supabase_headers(),
                json=data,
                timeout=10,
            )
            _log_event(open_id, "trial_start", None, "trial", note="新用户试用")
            return True

    except Exception as e:
        logger.error(f"开通试用失败: {e}")
        return False


def upgrade_plan(open_id: str, new_plan: str, amount: float = 0) -> bool:
    """
    升级用户订阅计划

    参数:
      open_id: 用户openid
      new_plan: monthly / quarterly / yearly
      amount: 实际支付金额（元）
    """
    if not _check_supabase():
        return False

    if new_plan not in ("monthly", "quarterly", "yearly"):
        logger.error(f"无效的订阅计划: {new_plan}")
        return False

    config = PLAN_CONFIG[new_plan]
    now = datetime.utcnow()
    plan_expire = now + timedelta(days=config["duration_days"])

    try:
        # 获取当前计划
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/users?open_id=eq.{open_id}&select=plan",
            headers=_supabase_headers(),
            timeout=10,
        )
        existing = resp.json()
        from_plan = existing[0]["plan"] if existing else "unknown"

        requests.patch(
            f"{SUPABASE_URL}/rest/v1/users?open_id=eq.{open_id}",
            headers=_supabase_headers(),
            json={
                "plan": new_plan,
                "plan_start": now.isoformat(),
                "plan_expire": plan_expire.isoformat(),
                "is_subscribed": True,
            },
            timeout=10,
        )

        _log_event(open_id, "plan_upgrade", from_plan, new_plan, amount=amount)
        logger.info(f"用户 {open_id} 升级到 {new_plan}，到期 {plan_expire}")
        return True

    except Exception as e:
        logger.error(f"升级订阅失败: {e}")
        return False


def expire_overdue_subscriptions() -> int:
    """
    批量过期已到期的订阅

    返回: 过期的用户数
    """
    if not _check_supabase():
        return 0

    try:
        now = datetime.utcnow().isoformat()
        # 查询需要过期的用户
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/users"
            f"?plan=in.(trial,monthly,quarterly,yearly)"
            f"&plan_expire=lt.{now}"
            f"&select=open_id,plan",
            headers=_supabase_headers(),
            timeout=10,
        )
        to_expire = resp.json()
        if not to_expire:
            return 0

        count = 0
        for user in to_expire:
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/users?open_id=eq.{user['open_id']}",
                headers=_supabase_headers(),
                json={"plan": "expired"},
                timeout=10,
            )
            _log_event(user["open_id"], "plan_expire", user["plan"], "expired", note="自动过期")
            count += 1

        logger.info(f"已过期 {count} 个订阅")
        return count

    except Exception as e:
        logger.error(f"批量过期失败: {e}")
        return 0


# ══════════════════════════════════════════════════════════
#  统计与报表
# ══════════════════════════════════════════════════════════

def get_subscription_stats() -> Dict[str, Any]:
    """获取订阅统计概览"""
    if not _check_supabase():
        return {"total": 0, "active": 0, "by_plan": {}}

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/users?select=open_id,plan,is_subscribed,plan_expire",
            headers=_supabase_headers(),
            timeout=15,
        )
        users = resp.json()

        now = datetime.utcnow()
        stats = {
            "total": len(users),
            "active": 0,
            "by_plan": {plan: 0 for plan in PLAN_CONFIG},
        }

        for user in users:
            plan = user.get("plan", "expired")
            is_subscribed = user.get("is_subscribed", False)
            plan_expire_str = user.get("plan_expire")

            is_active = is_subscribed
            if plan_expire_str and plan in ("trial", "monthly", "quarterly", "yearly"):
                try:
                    expire = datetime.fromisoformat(plan_expire_str.replace("Z", "+00:00"))
                    if expire.replace(tzinfo=None) < now:
                        plan = "expired"
                        is_active = False
                except (ValueError, TypeError):
                    pass

            stats["by_plan"][plan] = stats["by_plan"].get(plan, 0) + 1
            if is_active:
                stats["active"] += 1

        return stats

    except Exception as e:
        logger.error(f"获取统计失败: {e}")
        return {"total": 0, "active": 0, "by_plan": {}, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  内部工具
# ══════════════════════════════════════════════════════════

def _log_event(
    open_id: str,
    event_type: str,
    from_plan: Optional[str] = None,
    to_plan: Optional[str] = None,
    amount: float = 0,
    note: str = "",
) -> None:
    """记录订阅事件到审计日志"""
    if not _check_supabase():
        return

    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/subscription_events",
            headers=_supabase_headers(),
            json={
                "open_id": open_id,
                "event_type": event_type,
                "from_plan": from_plan,
                "to_plan": to_plan,
                "amount": amount,
                "note": note,
            },
            timeout=10,
        )
    except Exception as e:
        logger.error(f"记录订阅事件失败: {e}")
