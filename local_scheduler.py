"""
本地定时调度器 — 基于 APScheduler 的常驻后台服务

在工作日（周一至周五）自动执行扫描推送：
  - 13:30  盘中扫描
  - 17:30  盘后分析

用法:
    python local_scheduler.py              # 启动调度器（前台运行）
    python local_scheduler.py --once       # 立即执行一次扫描（测试用）
    python local_scheduler.py --test       # 仅打印下次运行时间，不实际执行

Docker 内通过 entrypoint 启动，配合 restart: unless-stopped 实现开机自启。
"""
import os
import sys
import subprocess
import argparse
import signal
import logging
from pathlib import Path
from datetime import datetime, timedelta

# ── 项目根目录加入 sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 日志配置 ──
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_DIR / "scheduler.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("scheduler")

# ── 加载 .env ──
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


# ══════════════════════════════════════════════════════════
#  A股交易日历判断
# ══════════════════════════════════════════════════════════

# 2026 年法定节假日（节假日 + 调休补班日）
# 格式: "YYYYMMDD" -> True=休市, False=补班交易日
# 每年初更新一次即可
_HOLIDAY_MAP_2026 = {
    # 元旦
    "20260101": True, "20260102": True, "20260103": True,
    # 春节（2月17日除夕至2月23日初七）
    "20260217": True, "20260218": True, "20260219": True,
    "20260220": True, "20260221": True, "20260222": True, "20260223": True,
    # 春节前周末补班
    "20260207": False,
    # 清明节
    "20260404": True, "20260405": True, "20260406": True,
    # 劳动节
    "20250501": True, "20260501": True, "20260502": True, "20260503": True,
    "20260504": True, "20260505": True,
    # 劳动节前周末补班
    "20260426": False,
    # 端午节
    "20260619": True, "20260620": True, "20260621": True,
    # 中秋节
    "20260925": True, "20260926": True, "20260927": True,
    # 国庆节
    "20261001": True, "20261002": True, "20261003": True,
    "20261004": True, "20261005": True, "20261006": True, "20261007": True,
    # 国庆前周末补班
    "20260927": False,
}


def is_trading_day(dt: datetime | None = None) -> bool:
    """判断是否为 A股交易日

    规则:
      1. 周末默认非交易日
      2. 法定节假日非交易日
      3. 调休补班日为交易日（周末但开市）
    """
    if dt is None:
        dt = datetime.now()

    date_str = dt.strftime("%Y%m%d")
    weekday = dt.weekday()  # 0=周一, 6=周日

    # 查节假日表
    holiday_flag = _HOLIDAY_MAP_2026.get(date_str)
    if holiday_flag is not None:
        # True=休市, False=补班交易日
        return not holiday_flag

    # 周末非交易日
    if weekday >= 5:
        return False

    return True


# ══════════════════════════════════════════════════════════
#  扫描任务执行
# ══════════════════════════════════════════════════════════

# 全局标志：防止任务重叠执行
_task_running = False
_data_update_running = False


def run_data_update():
    """独立数据更新任务：拉取当日全市场日线，更新 DuckDB 缓存

    在收盘后 16:00 执行，提前把当日数据拉好，
    这样 18:30 盘后推送的 prewarm_data 中 fetch_today_bars 可以秒级返回（缓存已最新）。
    即使此任务失败，18:30 推送时 prewarm_data 仍会重试。
    """
    global _data_update_running

    if _data_update_running:
        logger.warning("[data_update] 上一次更新尚未结束，跳过")
        return

    now = datetime.now()
    if not is_trading_day(now):
        logger.info(f"[data_update] 今日非交易日（{now.strftime('%Y-%m-%d')}），跳过")
        return

    _data_update_running = True
    logger.info(f"[data_update] 开始拉取当日最新数据 @ {now.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        cmd = [
            sys.executable, "-u", "-c",
            "from scripts.fetch_today_bars import fetch_today_bars; "
            "r = fetch_today_bars(); "
            "print(f'result: {r}')"
        ]
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
        )

        # 写入日志
        log_path = LOG_DIR / f"data_update_{now.strftime('%Y%m%d')}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== data_update {now.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"exit_code: {result.returncode}\n\n")
            f.write("--- stdout ---\n")
            f.write(result.stdout)
            f.write("\n--- stderr ---\n")
            f.write(result.stderr)

        if result.returncode == 0:
            logger.info(f"[data_update] 完成（exit=0）")
            # 打印关键结果行
            for line in result.stdout.strip().split("\n"):
                if "result:" in line or "merged" in line.lower():
                    logger.info(f"  {line.strip()}")
        else:
            logger.error(f"[data_update] 失败（exit={result.returncode}）")
            for line in result.stderr.strip().split("\n")[-3:]:
                logger.error(f"  {line}")

    except subprocess.TimeoutExpired:
        logger.error("[data_update] 超时（10分钟），强制终止")
    except Exception as e:
        logger.error(f"[data_update] 执行异常: {e}", exc_info=True)
    finally:
        _data_update_running = False


def run_scan_task(slot_name: str):
    """执行一次扫描推送

    Args:
        slot_name: 时段名称（"intraday" / "after_hours"），仅用于日志标记
    """
    global _task_running

    if _task_running:
        logger.warning(f"[{slot_name}] 上一次任务尚未结束，跳过本次执行")
        return

    now = datetime.now()
    if not is_trading_day(now):
        logger.info(f"[{slot_name}] 今日非交易日（{now.strftime('%Y-%m-%d')}），跳过")
        return

    _task_running = True
    logger.info(f"[{slot_name}] 开始执行扫描推送 @ {now.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 调用 trigger_push.py --local
        cmd = [sys.executable, "-u", "trigger_push.py", "--local"]
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 分钟超时
        )

        # 写入任务日志
        task_log = LOG_DIR / f"{slot_name}_{now.strftime('%Y%m%d')}.log"
        with open(task_log, "w", encoding="utf-8") as f:
            f.write(f"=== {slot_name} {now.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"exit_code: {result.returncode}\n\n")
            f.write("--- stdout ---\n")
            f.write(result.stdout)
            f.write("\n--- stderr ---\n")
            f.write(result.stderr)

        if result.returncode == 0:
            logger.info(f"[{slot_name}] 扫描推送完成（exit=0）")
        else:
            logger.error(f"[{slot_name}] 扫描推送失败（exit={result.returncode}）")
            # 打印最后几行 stderr 帮助排查
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-5:]:
                logger.error(f"  {line}")

    except subprocess.TimeoutExpired:
        logger.error(f"[{slot_name}] 扫描超时（30分钟），强制终止")
    except Exception as e:
        logger.error(f"[{slot_name}] 执行异常: {e}", exc_info=True)
    finally:
        _task_running = False


# ══════════════════════════════════════════════════════════
#  APScheduler 调度器
# ══════════════════════════════════════════════════════════

def create_scheduler():
    """创建并配置 APScheduler 调度器"""
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")

    # 数据更新：工作日 16:00（收盘后拉取当日全市场日线）
    # 提前更新 DuckDB 缓存，18:30 盘后推送可直接用最新数据
    scheduler.add_job(
        func=run_data_update,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=16,
            minute=0,
            timezone="Asia/Shanghai",
        ),
        id="data_update",
        name="收盘数据更新",
        misfire_grace_time=600,  # 错过 10 分钟内仍补执行
        coalesce=True,
        max_instances=1,
    )

    # 盘中扫描：工作日 14:00
    scheduler.add_job(
        func=run_scan_task,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=14,
            minute=0,
            timezone="Asia/Shanghai",
        ),
        args=["intraday"],
        id="intraday_scan",
        name="盘中扫描推送",
        misfire_grace_time=300,  # 如果错过 5 分钟内仍可补执行
        coalesce=True,           # 多次错过只执行一次
        max_instances=1,
    )

    # 盘后分析：工作日 18:30
    scheduler.add_job(
        func=run_scan_task,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=18,
            minute=30,
            timezone="Asia/Shanghai",
        ),
        args=["after_hours"],
        id="after_hours_scan",
        name="盘后分析推送",
        misfire_grace_time=300,
        coalesce=True,
        max_instances=1,
    )

    return scheduler


def print_next_runs(scheduler, count=5):
    """打印接下来的 N 次任务运行时间"""
    from apscheduler.schedulers.background import BackgroundScheduler

    jobs = scheduler.get_jobs()
    if not jobs:
        print("无已注册任务")
        return

    print(f"\n{'='*60}")
    print(f"  TradingAgents 本地调度器 — 接下来 {count} 次运行")
    print(f"{'='*60}")

    all_runs = []
    for job in jobs:
        for i in range(count):
            next_time = job.trigger.get_next_fire_time(None, datetime.now())
            if next_time and i == 0:
                all_runs.append((next_time, job.name))
                # 连续获取后续时间
                prev = next_time
                for _ in range(count - 1):
                    prev = job.trigger.get_next_fire_time(prev, prev)
                    if prev:
                        all_runs.append((prev, job.name))

    all_runs.sort(key=lambda x: x[0])
    for i, (run_time, name) in enumerate(all_runs[:count], 1):
        trading = is_trading_day(run_time)
        status = "交易日" if trading else "非交易日(将跳过)"
        print(f"  {i}. {run_time.strftime('%Y-%m-%d %H:%M:%S %a')} | {name} | {status}")

    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TradingAgents 本地定时调度器")
    parser.add_argument("--once", action="store_true", help="立即执行一次扫描（测试用）")
    parser.add_argument("--test", action="store_true", help="仅打印调度计划，不启动")
    args = parser.parse_args()

    # 立即执行一次
    if args.once:
        slot = "intraday" if 9 <= datetime.now().hour < 15 else "after_hours"
        logger.info(f"手动触发单次执行（slot={slot}）")
        run_scan_task(slot)
        return

    # 创建调度器
    scheduler = create_scheduler()

    # 打印调度计划
    print_next_runs(scheduler)

    if args.test:
        print("--test 模式：仅打印计划，不启动调度器")
        return

    # 启动调度器
    scheduler.start()
    logger.info("调度器已启动，等待任务触发...")
    logger.info("数据更新: 工作日 16:00 | 盘中扫描: 工作日 14:00 | 盘后分析: 工作日 18:30")
    logger.info("按 Ctrl+C 停止")

    # 优雅关闭
    def shutdown(signum, frame):
        logger.info(f"收到信号 {signum}，正在关闭调度器...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 主线程保持运行
    try:
        while True:
            signal.pause() if hasattr(signal, "pause") else None
            # Windows 没有 signal.pause，用 sleep 替代
            import time
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        shutdown(None, None)


if __name__ == "__main__":
    main()
