"""
推送通道模块 — Server酱微信推送

从 v63_daily_push.py 提取的公共推送逻辑，供各版本推送脚本复用。
"""
import os
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


def send_serverchan(title, desp, keys=None, scheduled=None):
    """Server酱推送，支持指定key列表（分组推送）和定时发送

    Args:
        title: 推送标题
        desp: 推送内容（Markdown格式）
        keys: Server酱Key列表，默认使用管理员组
        scheduled: 定时发送时间，格式 "YYYY-MM-DD HH:MM:SS"（北京时间）
                   消息会缓存在Server酱服务器，到指定时间才投递到微信

    Returns:
        bool: 是否至少有一个Key推送成功
    """
    if keys is None:
        keys = SERVERCHAN_KEYS_ADMIN
    if not keys:
        logger.info("Server酱推送: 无可用Key，跳过")
        return False

    import requests
    success_count = 0
    for key in keys:
        try:
            url = f"https://sctapi.ftqq.com/{key}.send"
            data = {"title": title, "desp": desp}
            if scheduled:
                data["scheduled"] = scheduled
            resp = requests.post(url, data=data, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                if result.get("code") == 0:
                    success_count += 1
                    if scheduled:
                        logger.info(f"Server酱定时发送: 已设定 {scheduled} 投递")
                else:
                    logger.warning(f"Server酱推送失败: {result}")
            else:
                logger.warning(f"Server酱推送HTTP错误: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Server酱推送异常: {e}")

    logger.info(f"Server酱推送: {success_count}/{len(keys)}")
    return success_count > 0


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
