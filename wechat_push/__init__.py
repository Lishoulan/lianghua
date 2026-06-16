"""
微信公众号推送模块
===========================
通过订阅号群发图文推文，将量化潜伏信号推送给所有关注者

功能：
  1. 微信事件回调处理（关注/取关/消息）
  2. 信号接收 → 生成图文HTML → 上传素材 → 群发推文
  3. access_token自动缓存刷新
"""

import os
import time
import json
import hashlib
import requests
from datetime import datetime, timedelta
from functools import lru_cache

# 环境变量
WECHAT_APP_ID = os.getenv("WECHAT_APP_ID", "")
WECHAT_APP_SECRET = os.getenv("WECHAT_APP_SECRET", "")
WECHAT_TOKEN = os.getenv("WECHAT_TOKEN", "lianghua2026")
WECHAT_ENCODING_AES_KEY = os.getenv("WECHAT_ENCODING_AES_KEY", "")
PUSH_API_KEY = os.getenv("PUSH_API_KEY", "lianghua_push_2026")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


# ══════════════════════════════════════════════════════════
#  access_token 管理
# ══════════════════════════════════════════════════════════

_token_cache = {"token": None, "expires_at": 0}


def get_access_token():
    """获取微信access_token，自动缓存刷新（有效期2小时）"""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]

    url = "https://api.weixin.qq.com/cgi-bin/token"
    params = {
        "grant_type": "client_credential",
        "appid": WECHAT_APP_ID,
        "secret": WECHAT_APP_SECRET,
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()

    if "access_token" not in data:
        raise Exception(f"获取access_token失败: {data}")

    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 7200) - 300  # 提前5分钟刷新
    print(f"  access_token刷新成功，有效期至: {datetime.fromtimestamp(_token_cache['expires_at']).strftime('%H:%M:%S')}")
    return data["access_token"]


# ══════════════════════════════════════════════════════════
#  签名验证
# ══════════════════════════════════════════════════════════

def check_signature(signature, timestamp, nonce):
    """验证微信服务器签名"""
    items = sorted([WECHAT_TOKEN, timestamp, nonce])
    sha1 = hashlib.sha1("".join(items).encode()).hexdigest()
    return sha1 == signature


# ══════════════════════════════════════════════════════════
#  Supabase 用户管理
# ══════════════════════════════════════════════════════════

def _supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def upsert_user(open_id, nickname=""):
    """新增或更新用户（关注时调用）"""
    if not SUPABASE_URL:
        print("  Supabase未配置，跳过用户存储")
        return

    now = datetime.utcnow().isoformat()
    trial_end = (datetime.utcnow() + timedelta(days=7)).isoformat()

    data = {
        "open_id": open_id,
        "nickname": nickname,
        "is_subscribed": True,
        "plan": "trial",
        "trial_start": now,
        "trial_end": trial_end,
    }

    # 先查询是否已存在
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/users?open_id=eq.{open_id}",
        headers=_supabase_headers(),
        timeout=10,
    )
    existing = resp.json()

    if existing and len(existing) > 0:
        # 已存在，更新关注状态
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/users?open_id=eq.{open_id}",
            headers=_supabase_headers(),
            json={"is_subscribed": True},
            timeout=10,
        )
    else:
        # 新用户，插入
        requests.post(
            f"{SUPABASE_URL}/rest/v1/users",
            headers=_supabase_headers(),
            json=data,
            timeout=10,
        )


def unsubscribe_user(open_id):
    """用户取关时更新状态"""
    if not SUPABASE_URL:
        return

    requests.patch(
        f"{SUPABASE_URL}/rest/v1/users?open_id=eq.{open_id}",
        headers=_supabase_headers(),
        json={"is_subscribed": False},
        timeout=10,
    )


def get_subscriber_count():
    """获取有效订阅用户数"""
    if not SUPABASE_URL:
        return 0

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/users?is_subscribed=eq.true&select=open_id",
        headers=_supabase_headers(),
        timeout=10,
    )
    users = resp.json()
    return len(users) if users else 0


# ══════════════════════════════════════════════════════════
#  图文HTML模板
# ══════════════════════════════════════════════════════════

def build_article_html(oamv_status, signals, industry_stats, is_intraday=False):
    """生成公众号图文HTML（精简卡片式，产品级排版）"""
    today = datetime.now().strftime("%Y-%m-%d")
    weekday_cn = ["周一","周二","周三","周四","周五","周六","周日"][datetime.now().weekday()]

    # 市场情绪
    if oamv_status:
        can_open = oamv_status["can_open_position"]
        sentiment = "偏多" if can_open else "偏空"
        sentiment_color = "#52c41a" if can_open else "#f5222d"
        sentiment_bg = "#f6ffed" if can_open else "#fff2f0"
    else:
        can_open = True
        sentiment = "中性"
        sentiment_color = "#faad14"
        sentiment_bg = "#fffbe6"

    mode_tag = "盘中" if is_intraday else "盘后"

    # 季节性
    current_month = datetime.now().month
    seasonal_html = ""
    danger_months = [3, 5]
    golden_months = [8, 10]
    if current_month in danger_months:
        seasonal_html = f'<div style="background:#fff2f0;border-left:3px solid #f5222d;padding:8px 12px;margin:10px 0;border-radius:4px;font-size:13px;color:#cf1322;">⚠️ {current_month}月历史表现偏弱，建议控制仓位</div>'
    elif current_month in golden_months:
        seasonal_html = f'<div style="background:#f6ffed;border-left:3px solid #52c41a;padding:8px 12px;margin:10px 0;border-radius:4px;font-size:13px;color:#389e0d;">✅ {current_month}月历史表现较好，信号可信度较高</div>'

    # 行业风向
    industry_html = ""
    if industry_stats:
        hot = [s for s in industry_stats if s["momentum"] > 0]
        rotation_in = [s for s in industry_stats if s["rotation"] == "轮入"]
        hot_names = "、".join(s["name"] for s in hot[:5]) if hot else "无"
        rot_names = "、".join(s["name"] for s in rotation_in[:3]) if rotation_in else ""
        industry_html = f'''
        <div style="background:#fff;border:1px solid #f0f0f0;border-radius:8px;padding:12px;margin:10px 0;">
            <div style="font-size:14px;font-weight:bold;margin-bottom:8px;">🔥 行业风向</div>
            <div style="font-size:13px;color:#333;">热门: {hot_names}</div>
            {"<div style='font-size:13px;color:#1890ff;margin-top:4px;'>🔄 轮入: " + rot_names + "</div>" if rot_names else ""}
        </div>'''

    # 信号分级
    priority_signals = []
    normal_signals = []
    ptr = {
        "score_8_golden": True,
        "vol_extreme_shrink": 0.3,
        "price_sweet_spot": (10, 20),
        "score_7_downgrade": True,
    }

    for s in signals:
        eq = s.get("entry_quality_score", 0)
        vr = s.get("vol_ratio", 1.0)
        price = s.get("price", 0)

        is_priority = False
        if ptr["score_8_golden"] and eq >= 8:
            is_priority = True
        if vr < ptr["vol_extreme_shrink"]:
            is_priority = True
        p_lo, p_hi = ptr["price_sweet_spot"]
        if p_lo <= price <= p_hi:
            is_priority = True
        if ptr["score_7_downgrade"] and eq == 7:
            is_priority = False

        if is_priority:
            priority_signals.append(s)
        else:
            normal_signals.append(s)

    # 信号卡片HTML
    signals_html = ""
    if not signals:
        if can_open:
            signals_html = '<div style="text-align:center;padding:30px;color:#999;font-size:14px;">📭 今日暂无符合条件的标的<br>耐心等待，不追涨是纪律</div>'
        else:
            signals_html = '<div style="text-align:center;padding:30px;color:#999;font-size:14px;">🛡️ 市场偏弱，建议观望</div>'
    else:
        # 优先挡
        if priority_signals:
            signals_html += '<div style="font-size:15px;font-weight:bold;margin:15px 0 8px 0;color:#f5222d;">🔴 重点推荐</div>'
            for s in priority_signals:
                eq = s.get("entry_quality_score", 0)
                vr = s.get("vol_ratio", 1.0)
                j = s.get("J", 99)

                tags_html = ""
                if eq >= 8:
                    tags_html += '<span style="background:#fff7e6;color:#fa8c16;padding:2px 6px;border-radius:3px;font-size:11px;margin-right:4px;">⭐8分</span>'
                if vr < 0.3:
                    tags_html += '<span style="background:#e6f7ff;color:#1890ff;padding:2px 6px;border-radius:3px;font-size:11px;margin-right:4px;">💧极度缩量</span>'
                if 10 <= s.get("price", 0) <= 20:
                    tags_html += '<span style="background:#f6ffed;color:#52c41a;padding:2px 6px;border-radius:3px;font-size:11px;">💰最佳区间</span>'

                pos = "仓位15%" if (vr < 0.3 or eq >= 8) else "仓位10%"
                change_sign = "+" if s["change_pct"] >= 0 else ""
                change_color = "#f5222d" if s["change_pct"] >= 0 else "#52c41a"

                signals_html += f'''
                <div style="background:#fff;border:1px solid #ffccc7;border-radius:8px;padding:12px;margin:8px 0;">
                    <div style="font-size:15px;font-weight:bold;margin-bottom:4px;">{s["name"]} <span style="color:#999;font-size:12px;font-weight:normal;">{s["code"]}</span></div>
                    <div style="font-size:18px;font-weight:bold;color:{change_color};margin:4px 0;">{s["price"]:.2f} <span style="font-size:13px;">{change_sign}{s["change_pct"]:.2f}%</span></div>
                    <div style="margin:6px 0;">{tags_html}</div>
                    <div style="font-size:12px;color:#666;display:flex;justify-content:space-between;">
                        <span>评分:{eq}/8 | J:{j:.0f} | 量比:{vr:.2f}</span>
                        <span style="color:#1890ff;">{pos}</span>
                    </div>
                    <div style="font-size:12px;color:#999;margin-top:4px;">止损:{s["hard_stop"]:.2f} | 持仓10天</div>
                </div>'''

        # 普通挡
        if normal_signals:
            signals_html += '<div style="font-size:15px;font-weight:bold;margin:15px 0 8px 0;color:#8c8c8c;">⚪ 关注标的</div>'
            for s in normal_signals:
                eq = s.get("entry_quality_score", 0)
                vr = s.get("vol_ratio", 1.0)
                j = s.get("J", 99)
                change_sign = "+" if s["change_pct"] >= 0 else ""
                change_color = "#f5222d" if s["change_pct"] >= 0 else "#52c41a"

                signals_html += f'''
                <div style="background:#fff;border:1px solid #f0f0f0;border-radius:8px;padding:12px;margin:8px 0;">
                    <div style="font-size:15px;font-weight:bold;margin-bottom:4px;">{s["name"]} <span style="color:#999;font-size:12px;font-weight:normal;">{s["code"]}</span></div>
                    <div style="font-size:16px;font-weight:bold;color:{change_color};margin:4px 0;">{s["price"]:.2f} <span style="font-size:13px;">{change_sign}{s["change_pct"]:.2f}%</span></div>
                    <div style="font-size:12px;color:#666;display:flex;justify-content:space-between;">
                        <span>评分:{eq}/8 | J:{j:.0f} | 量比:{vr:.2f}</span>
                        <span style="color:#8c8c8c;">仓位5%</span>
                    </div>
                    <div style="font-size:12px;color:#999;margin-top:4px;">止损:{s["hard_stop"]:.2f} | 持仓10天</div>
                </div>'''

    # 完整HTML
    html = f'''
    <div style="max-width:600px;margin:0 auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#333;padding:10px;">

        <!-- 头部 -->
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:20px;margin-bottom:15px;color:#fff;">
            <div style="font-size:20px;font-weight:bold;margin-bottom:8px;">📊 量化潜伏 | {mode_tag}版</div>
            <div style="font-size:13px;opacity:0.8;">{today} {weekday_cn}</div>
        </div>

        <!-- 市场情绪 -->
        <div style="background:{sentiment_bg};border-radius:8px;padding:12px;margin:10px 0;display:flex;align-items:center;justify-content:space-between;">
            <div>
                <div style="font-size:14px;font-weight:bold;color:{sentiment_color};">📈 市场情绪: {sentiment}</div>
                <div style="font-size:12px;color:#666;margin-top:2px;">OAMV活筹择时 | 信号:{len(signals)}只</div>
            </div>
            <div style="font-size:24px;">{"🟢" if can_open else "🔴"}</div>
        </div>

        {seasonal_html}
        {industry_html}

        <!-- 信号区域 -->
        {signals_html}

        <!-- 仓位管理 -->
        <div style="background:#fafafa;border-radius:8px;padding:12px;margin:15px 0;">
            <div style="font-size:14px;font-weight:bold;margin-bottom:6px;">📦 仓位管理</div>
            <div style="font-size:12px;color:#666;line-height:1.8;">
                单只 ≤15%(推荐) / ≤10%(关注) | 总敞口 ≤50%<br>
                持仓周期: 10个交易日 | 4级退出机制
            </div>
        </div>

        <!-- 底部 -->
        <div style="text-align:center;padding:15px 0;border-top:1px solid #f0f0f0;margin-top:15px;">
            <div style="font-size:11px;color:#999;">⚠️ 量化系统自动输出，不构成投资建议</div>
            <div style="font-size:11px;color:#bbb;margin-top:4px;">量化潜伏 Pro | V6.4</div>
        </div>

    </div>
    '''

    return html


# ══════════════════════════════════════════════════════════
#  微信素材上传 + 群发
# ══════════════════════════════════════════════════════════

def upload_news_article(title, content_html, thumb_media_id=""):
    """
    上传图文素材到微信服务器

    参数:
      title: 文章标题
      content_html: 文章HTML内容
      thumb_media_id: 封面图片素材ID（可选）

    返回:
      media_id: 图文素材ID，用于群发
    """
    token = get_access_token()
    url = f"https://api.weixin.qq.com/cgi-bin/material/add_news?access_token={token}"

    article = {
        "articles": [{
            "title": title,
            "author": "量化潜伏Pro",
            "digest": f"今日发现潜伏信号，点击查看详情",
            "content": content_html,
            "content_source_url": "",
            "need_open_comment": 0,
            "only_fans_can_comment": 0,
        }]
    }

    resp = requests.post(url, json=article, timeout=30)
    data = resp.json()

    if "media_id" not in data:
        raise Exception(f"上传图文素材失败: {data}")

    print(f"  图文素材上传成功: media_id={data['media_id']}")
    return data["media_id"]


def mass_send_news(media_id):
    """
    群发图文消息给所有关注者

    参数:
      media_id: 图文素材ID

    返回:
      msg_id: 群发消息ID
    """
    token = get_access_token()
    url = f"https://api.weixin.qq.com/cgi-bin/message/mass/sendall?access_token={token}"

    data = {
        "filter": {
            "is_to_all": True,
        },
        "mpnews": {
            "media_id": media_id,
        },
        "msgtype": "mpnews",
        "send_ignore_reprint": 0,
    }

    resp = requests.post(url, json=data, timeout=30)
    result = resp.json()

    if result.get("errcode", 0) != 0:
        raise Exception(f"群发失败: {result}")

    print(f"  群发成功: msg_id={result.get('msg_id')}, msg_data_id={result.get('msg_data_id')}")
    return result


def push_signals_to_wechat(oamv_status, signals, industry_stats, is_intraday=False):
    """
    主入口：接收信号数据 → 生成图文 → 上传 → 群发

    参数:
      oamv_status: OAMV择时状态
      signals: 信号列表
      industry_stats: 行业统计
      is_intraday: 是否盘中

    返回:
      dict: 推送结果
    """
    if not WECHAT_APP_ID or not WECHAT_APP_SECRET:
        print("  微信公众号未配置，跳过群发")
        return {"success": False, "error": "not_configured"}

    try:
        # 1. 生成标题
        today = datetime.now().strftime("%m月%d日")
        if oamv_status:
            can_open = oamv_status["can_open_position"]
            sentiment = "偏多" if can_open else "偏空"
        else:
            sentiment = "中性"
        mode_tag = "盘中" if is_intraday else "盘后"
        title = f"量化潜伏 | {today}{mode_tag} | {len(signals)}只信号 | {sentiment}"

        # 2. 生成图文HTML
        print("  生成公众号图文HTML...", flush=True)
        html = build_article_html(oamv_status, signals, industry_stats, is_intraday)

        # 3. 上传素材
        print("  上传图文素材...", flush=True)
        media_id = upload_news_article(title, html)

        # 4. 群发
        print("  群发推文...", flush=True)
        result = mass_send_news(media_id)

        return {
            "success": True,
            "media_id": media_id,
            "msg_id": result.get("msg_id"),
            "msg_data_id": result.get("msg_data_id"),
        }

    except Exception as e:
        print(f"  公众号群发失败: {e}", flush=True)
        return {"success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════
#  微信事件处理
# ══════════════════════════════════════════════════════════

def handle_wechat_event(xml_data):
    """
    处理微信事件推送

    事件类型:
      subscribe: 用户关注
      unsubscribe: 用户取关
      CLICK: 菜单点击
      text: 文本消息
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_data)
    except Exception:
        return ""

    msg_type = root.find("MsgType").text if root.find("MsgType") is not None else ""
    from_user = root.find("FromUserName").text if root.find("FromUserName") is not None else ""
    to_user = root.find("ToUserName").text if root.find("ToUserName") is not None else ""

    reply = ""

    if msg_type == "event":
        event = root.find("Event").text if root.find("Event") is not None else ""

        if event == "subscribe":
            # 用户关注 → 自动回复 + 记录用户
            upsert_user(from_user)
            reply = (
                "🎉 欢迎关注量化潜伏Pro！\n\n"
                "您已获得7天免费试用，每日22:00将收到盘后潜伏信号推送。\n\n"
                "📖 核心策略：威科夫LPS + VPA量价分析\n"
                "📐 8分评分体系 | OAMV择时 | 行业动量过滤\n\n"
                "回复【信号】查看今日最新信号\n"
                "回复【帮助】查看使用说明"
            )

        elif event == "unsubscribe":
            # 用户取关
            unsubscribe_user(from_user)

    elif msg_type == "text":
        content = root.find("Content").text if root.find("Content") is not None else ""

        if "信号" in content:
            reply = (
                "📊 今日信号将在22:00盘后推送，请留意公众号消息。\n\n"
                "如需查看历史信号，请回复【历史】"
            )
        elif "帮助" in content or "help" in content.lower():
            reply = (
                "📖 量化潜伏Pro 使用说明\n\n"
                "1️⃣ 每日22:00自动推送盘后信号\n"
                "2️⃣ 信号分为【重点推荐】和【关注标的】\n"
                "3️⃣ 建议持仓10个交易日\n"
                "4️⃣ 单只仓位≤15%，总敞口≤50%\n\n"
                "回复【信号】查看最新信号\n"
                "回复【订阅】查看订阅状态"
            )
        elif "订阅" in content:
            reply = (
                "📋 订阅状态\n\n"
                "当前方案：7天免费试用\n"
                "到期后可续费：\n"
                "💰 月度 29元/月\n"
                "💰 季度 79元/季\n"
                "💰 年度 199元/年\n\n"
                "如需订阅请联系管理员"
            )
        else:
            reply = "收到您的消息！回复【信号】【帮助】【订阅】获取对应信息。"

    # 构造XML回复
    if reply:
        reply_xml = f"""<xml>
<ToUserName><![CDATA[{from_user}]]></ToUserName>
<FromUserName><![CDATA[{to_user}]]></FromUserName>
<CreateTime>{int(time.time())}</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[{reply}]]></Content>
</xml>"""
        return reply_xml

    return ""
