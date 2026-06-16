"""
腾讯云函数入口
===========================
函数1: wechat_handler - 微信事件回调
函数2: push_handler - 接收GitHub Actions信号推送

部署方式：
  腾讯云函数 → API网关 → 绑定自定义域名（可选）
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

        # 调用公众号群发
        result = push_signals_to_wechat(oamv_status, signals, industry_stats, is_intraday)

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
