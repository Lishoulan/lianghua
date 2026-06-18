"""
推送通道模块 — Server酱微信推送

从 v63_daily_push.py 提取的公共推送逻辑，供各版本推送脚本复用。
增强功能：重试机制 + 定时投递降级策略 + pushid记录
"""
import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

# 推送分组配置
SERVERCHAN_KEYS_ADMIN = [k.strip() for k in os.getenv("SERVERCHAN_KEY", "").split(",") if k.strip()]
SERVERCHAN_KEYS_BETA = [k.strip() for k in os.getenv("SERVERCHAN_KEY_BETA", "").split(",") if k.strip()]

# 兼容旧变量名
SERVERCHAN_KEYS = SERVERCHAN_KEYS_ADMIN

# 最近一次推送的 pushid 记录（用于状态查询）
_last_push_ids = []


def _do_send(key, title, desp, scheduled=None):
    """执行单次Server酱推送请求
    
    Returns:
        tuple: (success: bool, pushid: str or None, error: str or None)
    """
    import requests
    try:
        url = f"https://sctapi.ftqq.com/{key}.send"
        data = {"title": title, "desp": desp}
        if scheduled:
            data["scheduled"] = scheduled
        resp = requests.post(url, data=data, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("code") == 0:
                pushid = result.get("data", {}).get("pushid", "")
                return True, pushid, None
            else:
                return False, None, f"API返回错误: {result}"
        else:
            return False, None, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, None, str(e)


def send_serverchan(title, desp, keys=None, scheduled=None):
    """Server酱推送，支持指定key列表（分组推送）和定时发送
    
    增强功能：
    - 单个Key推送失败后间隔5秒重试一次
    - 如果 scheduled 推送两次都失败，降级为立即发送
    - 记录成功推送的 pushid

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
        logger.info("Server酱推送: 无可用Key，跳过")
        return False

    success_count = 0
    push_ids = []

    for key in keys:
        # 第一次尝试
        ok, pushid, err = _do_send(key, title, desp, scheduled=scheduled)
        if ok:
            success_count += 1
            if pushid:
                push_ids.append(pushid)
            if scheduled:
                logger.info(f"Server酱定时发送: 已设定 {scheduled} 投递 (pushid={pushid})")
            continue

        logger.warning(f"Server酱推送失败(第1次): {err}")
        
        # 间隔5秒重试
        time.sleep(5)
        ok, pushid, err = _do_send(key, title, desp, scheduled=scheduled)
        if ok:
            success_count += 1
            if pushid:
                push_ids.append(pushid)
            logger.info(f"Server酱重试成功 (pushid={pushid})")
            continue

        logger.warning(f"Server酱推送失败(第2次): {err}")

        # 降级策略：如果是定时发送失败，尝试立即发送
        if scheduled:
            logger.warning("定时投递失败，降级为立即发送（宁可早到也不丢失）")
            ok, pushid, err = _do_send(key, title, desp, scheduled=None)
            if ok:
                success_count += 1
                if pushid:
                    push_ids.append(pushid)
                logger.info(f"Server酱降级立即发送成功 (pushid={pushid})")
            else:
                logger.error(f"Server酱推送彻底失败: {err}")

    _last_push_ids = push_ids
    logger.info(f"Server酱推送: {success_count}/{len(keys)} 成功")
    return success_count > 0


def get_last_push_ids():
    """获取最近一次推送的 pushid 列表"""
    return _last_push_ids.copy()


def send_group_push(admin_title, admin_desp, beta_title, beta_desp, scheduled=None):
    """分组推送：管理员组（完整技术版）+ 内测组（精简卡片版）

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

    # 管理员组推送
    if SERVERCHAN_KEYS_ADMIN:
        logger.info(f"推送管理员组 ({len(SERVERCHAN_KEYS_ADMIN)}个Key)...")
        results["admin"] = send_serverchan(admin_title, admin_desp, keys=SERVERCHAN_KEYS_ADMIN, scheduled=scheduled)
    else:
        logger.info("管理员组: 无Key配置，跳过")
        results["admin"] = False

    # 内测组推送
    if SERVERCHAN_KEYS_BETA:
        logger.info(f"推送内测组 ({len(SERVERCHAN_KEYS_BETA)}个Key)...")
        results["beta"] = send_serverchan(beta_title, beta_desp, keys=SERVERCHAN_KEYS_BETA, scheduled=scheduled)
    else:
        logger.info("内测组: 无Key配置，跳过")
        results["beta"] = False

    return results
