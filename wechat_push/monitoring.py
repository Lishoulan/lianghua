"""
监控埋点模块
===========================
记录推送日志、系统指标、运行状态，用于长期运营监控

功能：
  1. 推送日志记录（push_logs 表）
  2. 系统指标采集（metrics 表）
  3. 健康检查端点
  4. 运营统计报表
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import requests

logger = logging.getLogger(__name__)

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
    return bool(SUPABASE_URL and SUPABASE_KEY)


# ══════════════════════════════════════════════════════════
#  推送日志
# ══════════════════════════════════════════════════════════

def log_push(
    mode: str,
    oamv_status: Optional[Dict] = None,
    signals: Optional[List[Dict]] = None,
    industry_stats: Optional[List[Dict]] = None,
    wechat_result: Optional[Dict] = None,
    duration_seconds: float = 0,
) -> Optional[int]:
    """
    记录一次推送日志

    参数:
      mode: intraday / after_hours
      oamv_status: OAMV 择时状态
      signals: 信号列表
      industry_stats: 行业统计
      wechat_result: 微信群发结果
      duration_seconds: 总耗时

    返回: 日志 ID（失败返回 None）
    """
    if not _check_supabase():
        logger.debug("Supabase 未配置，跳过推送日志")
        return None

    try:
        import json as _json
        data = {
            "push_time": datetime.utcnow().isoformat(),
            "mode": mode,
            "oamv_status": _json.dumps(oamv_status, ensure_ascii=False) if oamv_status else None,
            "signal_count": len(signals) if signals else 0,
            "signals_json": _json.dumps(signals, ensure_ascii=False) if signals else None,
            "industry_stats": _json.dumps(industry_stats, ensure_ascii=False) if industry_stats else None,
            "wechat_mass_id": wechat_result.get("msg_id") if wechat_result else None,
            "wechat_success": 1 if (wechat_result and wechat_result.get("success")) else 0,
            "wechat_error": wechat_result.get("error") if wechat_result else None,
            "duration_seconds": round(duration_seconds, 2),
        }

        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/push_logs",
            headers=_supabase_headers(),
            json=data,
            timeout=10,
        )
        result = resp.json()
        if result and isinstance(result, list):
            return result[0].get("id")
        return None

    except Exception as e:
        logger.error(f"记录推送日志失败: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  系统指标
# ══════════════════════════════════════════════════════════

def record_metric(name: str, value: float, tags: Optional[Dict] = None) -> bool:
    """
    记录系统指标

    常用指标名:
      - scan.duration_seconds    扫描耗时
      - scan.signal_count        信号数量
      - scan.stock_count         扫描股票数
      - push.wechat.success      微信推送成功(0/1)
      - push.serverchan.success  Server酱推送成功(0/1)
      - cache.hit_rate           缓存命中率
      - subscribers.active       活跃订阅数
    """
    if not _check_supabase():
        return False

    try:
        import json as _json
        requests.post(
            f"{SUPABASE_URL}/rest/v1/metrics",
            headers=_supabase_headers(),
            json={
                "metric_name": name,
                "metric_value": value,
                "metric_tags": _json.dumps(tags or {}, ensure_ascii=False),
            },
            timeout=5,
        )
        return True
    except Exception as e:
        logger.error(f"记录指标失败: {e}")
        return False


def get_recent_metrics(name: str, hours: int = 24) -> List[Dict]:
    """获取最近 N 小时的指标数据"""
    if not _check_supabase():
        return []

    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/metrics"
            f"?metric_name=eq.{name}"
            f"&recorded_at=gte.{since.isoformat()}"
            f"&order=recorded_at.desc&limit=500",
            headers=_supabase_headers(),
            timeout=10,
        )
        return resp.json() or []
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        return []


# ══════════════════════════════════════════════════════════
#  健康检查
# ══════════════════════════════════════════════════════════

def health_check() -> Dict[str, Any]:
    """
    系统健康检查

    返回各组件状态：
      - supabase: 数据库连接
      - wechat: 微信 access_token 获取
      - data_source: 数据源可用性
    """
    from datetime import timedelta

    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
    }

    # 1. Supabase
    if _check_supabase():
        try:
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/push_logs?select=id&limit=1",
                headers=_supabase_headers(),
                timeout=5,
            )
            status["components"]["supabase"] = "ok" if resp.status_code == 200 else f"error:{resp.status_code}"
        except Exception as e:
            status["components"]["supabase"] = f"error:{e}"
            status["status"] = "degraded"
    else:
        status["components"]["supabase"] = "not_configured"
        status["status"] = "degraded"

    # 2. 微信公众号
    wechat_app_id = os.getenv("WECHAT_APP_ID", "")
    wechat_app_secret = os.getenv("WECHAT_APP_SECRET", "")
    if wechat_app_id and wechat_app_secret:
        try:
            from wechat_push import get_access_token
            token = get_access_token()
            status["components"]["wechat"] = "ok" if token else "error:no_token"
        except Exception as e:
            status["components"]["wechat"] = f"error:{e}"
            status["status"] = "degraded"
    else:
        status["components"]["wechat"] = "not_configured"

    # 3. 数据源（轻量检查，不实际拉取数据）
    tushare_token = os.getenv("TUSHARE_TOKEN", "")
    status["components"]["tushare"] = "configured" if tushare_token else "not_configured"

    # 4. 最近推送状态
    if _check_supabase():
        try:
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/push_logs?order=push_time.desc&limit=1&select=push_time,mode,wechat_success",
                headers=_supabase_headers(),
                timeout=5,
            )
            recent = resp.json()
            if recent:
                status["last_push"] = recent[0]
            else:
                status["last_push"] = None
        except Exception:
            status["last_push"] = None

    if any(v.startswith("error") for v in status["components"].values()):
        status["status"] = "unhealthy"

    return status


# ══════════════════════════════════════════════════════════
#  运营统计报表
# ══════════════════════════════════════════════════════════

def get_operations_report(days: int = 7) -> Dict[str, Any]:
    """
    获取运营统计报表

    返回:
      - 推送次数、成功率
      - 信号数量统计
      - 订阅用户变化
      - 最近 N 天每日推送详情
    """
    if not _check_supabase():
        return {"error": "supabase not configured"}

    from datetime import timedelta
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    try:
        # 推送日志
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/push_logs"
            f"?push_time=gte.{since}"
            f"&order=push_time.desc&limit=500",
            headers=_supabase_headers(),
            timeout=15,
        )
        logs = resp.json() or []

        total = len(logs)
        success = sum(1 for l in logs if l.get("wechat_success", 0) > 0)
        total_signals = sum(l.get("signal_count", 0) for l in logs)
        total_duration = sum(l.get("duration_seconds", 0) or 0 for l in logs)

        # 按日聚合
        daily: Dict[str, Dict] = {}
        for log in logs:
            day = (log.get("push_time") or "")[:10]
            if day not in daily:
                daily[day] = {"pushes": 0, "signals": 0, "success": 0}
            daily[day]["pushes"] += 1
            daily[day]["signals"] += log.get("signal_count", 0)
            daily[day]["success"] += 1 if log.get("wechat_success") else 0

        return {
            "period_days": days,
            "total_pushes": total,
            "success_rate": round(success / total, 4) if total else 0,
            "total_signals": total_signals,
            "avg_signals_per_push": round(total_signals / total, 2) if total else 0,
            "avg_duration_seconds": round(total_duration / total, 2) if total else 0,
            "daily": daily,
        }

    except Exception as e:
        logger.error(f"获取运营报表失败: {e}")
        return {"error": str(e)}
