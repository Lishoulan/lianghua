"""
腾讯云函数入口
===========================
函数1: wechat_handler  - 微信事件回调（关注/取关/消息）
函数2: push_handler    - 接收 GitHub Actions 信号推送
函数3: health_handler  - 健康检查端点（监控用）
函数4: stats_handler   - 订阅统计端点（运营用）

部署方式：
  腾讯云函数 → API网关 → 绑定自定义域名（可选）

API 路由：
  GET  /wechat   → 微信验证签名
  POST /wechat   → 微信事件推送
  POST /push     → GitHub Actions 信号推送
  GET  /health   → 健康检查
  GET  /stats    → 运营统计
"""

import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wechat_push import (
    check_signature,
    handle_wechat_event,
    push_signals_to_wechat,
    PUSH_API_KEY,
)
from wechat_push.subscription import (
    start_trial,
    get_user_subscription,
    get_subscription_stats,
    expire_overdue_subscriptions,
)
from wechat_push.monitoring import (
    log_push,
    health_check,
    get_operations_report,
    record_metric,
)


def wechat_handler(event, context):
    """
    微信事件回调处理函数

    API网关配置:
      GET  /wechat → 微信验证签名
      POST /wechat → 接收微信事件推送
    """
    try:
        # API网关事件格式
        if "queryString" in event:
            # GET请求 - 微信验证签名
            query = event.get("queryString", {})
            signature = query.get("signature", "")
            timestamp = query.get("timestamp", "")
            nonce = query.get("nonce", "")
            echostr = query.get("echostr", "")

            if check_signature(signature, timestamp, nonce):
                return {
                    "isBase64Encoded": False,
                    "statusCode": 200,
                    "headers": {"Content-Type": "text/plain"},
                    "body": echostr,
                }
            else:
                return {
                    "isBase64Encoded": False,
                    "statusCode": 403,
                    "body": "Invalid signature",
                }

        elif "body" in event:
            # POST请求 - 微信事件推送
            body = event.get("body", "")
            if event.get("isBase64Encoded"):
                import base64
                body = base64.b64decode(body).decode("utf-8")

            reply_xml = handle_wechat_event(body)

            return {
                "isBase64Encoded": False,
                "statusCode": 200,
                "headers": {"Content-Type": "application/xml"},
                "body": reply_xml,
            }

        return {
            "isBase64Encoded": False,
            "statusCode": 400,
            "body": "Bad request",
        }

    except Exception as e:
        print(f"wechat_handler error: {e}")
        return {
            "isBase64Encoded": False,
            "statusCode": 500,
            "body": str(e),
        }


def push_handler(event, context):
    """
    信号推送处理函数

    GitHub Actions扫描完成后调用此函数：
      POST /push
      Headers: Authorization: Bearer <PUSH_API_KEY>
      Body: {
        "oamv_status": {...},
        "signals": [...],
        "industry_stats": [...],
        "is_intraday": false
      }
    """
    try:
        # 验证API Key
        headers = event.get("headers", {})
        auth = headers.get("authorization", "") or headers.get("Authorization", "")
        if auth.replace("Bearer ", "") != PUSH_API_KEY:
            return {
                "isBase64Encoded": False,
                "statusCode": 401,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Unauthorized"}),
            }

        # 解析请求体
        body = event.get("body", "{}")
        if event.get("isBase64Encoded"):
            import base64
            body = base64.b64decode(body).decode("utf-8")

        data = json.loads(body)
        oamv_status = data.get("oamv_status")
        signals = data.get("signals", [])
        industry_stats = data.get("industry_stats", [])
        is_intraday = data.get("is_intraday", False)

        print(f"收到推送请求: {len(signals)}只信号, 盘中={is_intraday}")

        # 先过期到期的订阅
        try:
            expired = expire_overdue_subscriptions()
            if expired:
                print(f"  自动过期 {expired} 个订阅")
        except Exception as e:
            print(f"  过期检查失败（非致命）: {e}")

        # 调用公众号群发
        result = push_signals_to_wechat(oamv_status, signals, industry_stats, is_intraday)

        # 记录推送日志和指标
        mode = "intraday" if is_intraday else "after_hours"
        try:
            log_push(
                mode=mode,
                oamv_status=oamv_status,
                signals=signals,
                industry_stats=industry_stats,
                wechat_result=result,
            )
            record_metric("push.wechat.success", 1 if result.get("success") else 0, {"mode": mode})
            record_metric("scan.signal_count", len(signals), {"mode": mode})
        except Exception as e:
            print(f"  记录日志失败（非致命）: {e}")

        return {
            "isBase64Encoded": False,
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result, ensure_ascii=False),
        }

    except Exception as e:
        print(f"push_handler error: {e}")
        return {
            "isBase64Encoded": False,
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }


# ══════════════════════════════════════════════════════════
#  健康检查端点
# ══════════════════════════════════════════════════════════

def health_handler(event, context):
    """
    健康检查端点

    GET /health
    用于外部监控（如 UptimeRobot、腾讯云监控）探活
    """
    try:
        status = health_check()
        http_code = 200 if status["status"] == "healthy" else 503
        return {
            "isBase64Encoded": False,
            "statusCode": http_code,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(status, ensure_ascii=False),
        }
    except Exception as e:
        return {
            "isBase64Encoded": False,
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "unhealthy", "error": str(e)}),
        }


# ══════════════════════════════════════════════════════════
#  运营统计端点
# ══════════════════════════════════════════════════════════

def stats_handler(event, context):
    """
    运营统计端点

    GET /stats?days=7
    返回推送统计和订阅概览
    """
    try:
        # 验证 API Key（运营数据需认证）
        headers = event.get("headers", {})
        auth = headers.get("authorization", "") or headers.get("Authorization", "")
        if auth.replace("Bearer ", "") != PUSH_API_KEY:
            return {
                "isBase64Encoded": False,
                "statusCode": 401,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Unauthorized"}),
            }

        query = event.get("queryString", {}) or {}
        days = int(query.get("days", "7"))

        report = get_operations_report(days=days)
        sub_stats = get_subscription_stats()

        result = {
            "operations": report,
            "subscriptions": sub_stats,
            "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
        }

        return {
            "isBase64Encoded": False,
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result, ensure_ascii=False, default=str),
        }
    except Exception as e:
        return {
            "isBase64Encoded": False,
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
