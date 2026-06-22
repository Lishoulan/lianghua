"""
全市场扫描引擎模块

提供同步和异步两种扫描模式：
  - SyncScanner: 基于ThreadPoolExecutor的同步扫描（兼容旧逻辑）
  - AsyncScanner: 基于asyncio+aiohttp的异步扫描（性能优化）

两种模式均支持：
  - akshare批量预筛选（减少扫描范围）
  - 断点续传（中断后可恢复）
  - 盘中实时K线拼接
  - 行业过滤 + 入场质量评分过滤
"""
import os
import sys
import json
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

logger = logging.getLogger(__name__)

# 重试机制
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False


# ══════════════════════════════════════════════════════════
#  信号处理核心逻辑
# ══════════════════════════════════════════════════════════

def _fetch_and_process_one_core(ts_code, name, industry, best_params, realtime_quote=None):
    """获取单只股票数据并计算指标的核心逻辑（不含重试和异常捕获）

    Returns:
        tuple: (ts_code, name, industry, df_or_None)
    """
    from classic_ta.stock_data_duckdb import get_stock_data_readonly, _pending_write_queue
    from classic_ta.v60_ambush_model import IndicatorCalcBase
    from classic_ta.v63_ambush_model import add_micro_confirm_indicators, Detect_AmbushSignal_V63
    from classic_ta.v64_ambush_model import add_inst_support_indicators, Detect_AmbushSignal_V64

    df = get_stock_data_readonly(ts_code, min_rows=130)
    if df is None:
        # 缓存缺失，放入待更新队列，跳过本次扫描
        _pending_write_queue.put(ts_code)
        return (ts_code, name, industry, None)

    # 盘中模式：拼接实时K线
    if realtime_quote is not None:
        from classic_ta.common.stock_pool import append_realtime_bar
        df = append_realtime_bar(df, realtime_quote)

    df = IndicatorCalcBase(df)
    df = add_micro_confirm_indicators(df)
    df = add_inst_support_indicators(df, best_params)
    df = Detect_AmbushSignal_V64(df, best_params)

    if df is None or len(df) < 130:
        return (ts_code, name, industry, None)

    return (ts_code, name, industry, df)


def _fetch_and_process_one_with_retry(ts_code, name, industry, best_params, realtime_quote=None):
    """带重试的单只股票处理"""
    if TENACITY_AVAILABLE:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
               retry=retry_if_exception_type((ConnectionError, TimeoutError)))
        def _retry_wrapper():
            return _fetch_and_process_one_core(ts_code, name, industry, best_params, realtime_quote)
        return _retry_wrapper()
    else:
        return _fetch_and_process_one_core(ts_code, name, industry, best_params, realtime_quote)


def _fetch_and_process_one(ts_code, name, industry, best_params, realtime_quote=None):
    """获取单只股票数据并计算指标，返回 (ts_code, name, industry, df_or_None)"""
    try:
        return _fetch_and_process_one_with_retry(ts_code, name, industry, best_params, realtime_quote)
    except Exception as e:
        logger.warning(f"股票处理异常 {ts_code}({name}): {e}")
        return (ts_code, name, industry, None)


def _extract_signal_info(ts_code, name, industry, df, best_params):
    """从已处理的DataFrame中提取信号信息

    Args:
        ts_code: 股票代码
        name: 股票名称
        industry: 行业
        df: 已计算指标的DataFrame
        best_params: 策略参数

    Returns:
        dict or None: 信号信息，如果不是信号日则返回None
    """
    from classic_ta.common.signal_analyzer import analyze_signal_detail
    from classic_ta.common.t_trading import analyze_t_trading

    latest = df.iloc[-1]
    if pd.isna(latest.get("yellow_line")) or pd.isna(latest.get("white_line")):
        return None

    if not latest.get("ambush_signal", False):
        return None

    prev = df.iloc[-2]
    change_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
    vol_ratio = latest["Volume"] / latest["volume_ma"] if latest["volume_ma"] > 0 else 0

    # 个股N日累计收益率（与行业动量同窗口，用于行业滞涨股识别）
    mom_days = best_params.get("industry_momentum_days", 10)
    stock_ret_n = None
    if len(df) > mom_days:
        close_now = float(latest["Close"])
        close_prev = float(df.iloc[-(mom_days + 1)]["Close"])
        if close_prev > 0:
            stock_ret_n = round((close_now - close_prev) / close_prev, 4)

    detail = analyze_signal_detail(df, len(df) - 1, best_params)
    t_info = analyze_t_trading(df, len(df) - 1)

    window = best_params["ambush_window"]
    sos_dates = []
    for j in range(max(0, len(df) - window), len(df)):
        if df.iloc[j].get("tag_sos_anchor", False):
            sos_dates.append(df.index[j].strftime("%m-%d"))

    eq_score = int(latest.get("entry_quality_score", 0)) if "entry_quality_score" in df.columns else 0
    hard_stop = round(float(latest["Close"] * 0.95), 2)
    chandelier_init = round(float(latest["Close"] - 3 * latest["atr14"]), 2)

    signal_info = {
        "code": ts_code,
        "name": name,
        "industry": industry,
        "price": round(float(latest["Close"]), 2),
        "change_pct": round(float(change_pct), 2),
        "white_line": round(float(latest["white_line"]), 2),
        "yellow_line": round(float(latest["yellow_line"]), 2),
        "J": round(float(latest["J"]), 1),
        "atr14": round(float(latest["atr14"]), 2),
        "vol_ratio": round(float(vol_ratio), 2),
        "sos_dates": sos_dates,
        "analysis": detail,
        "signal_date": df.index[-1].strftime("%Y-%m-%d"),
        "entry_quality_score": eq_score,
        "eq_j_score": int(latest.get("eq_j_score", 0)),
        "eq_vol_score": int(latest.get("eq_vol_score", 0)),
        "eq_candle_score": int(latest.get("eq_candle_score", 0)),
        "eq_ma_score": int(latest.get("eq_ma_score", 0)),
        "hard_stop": hard_stop,
        "chandelier_init": chandelier_init,
        "inst_support_score": int(latest.get("inst_support_score", 0)),
        "factor_a": bool(latest.get("factor_a_vol_stable", False)),
        "factor_b": bool(latest.get("factor_b_vp_divergence", False)),
        "factor_c": bool(latest.get("factor_c_support_hold", False)),
        "factor_d": bool(latest.get("factor_d_intraday_accum", False)),
        "stock_ret_n": stock_ret_n,
        "t_trading": t_info,
    }
    return signal_info


# ══════════════════════════════════════════════════════════
#  同步扫描器
# ══════════════════════════════════════════════════════════

class SyncScanner:
    """基于ThreadPoolExecutor的同步全市场扫描器

    Features:
      - 并发获取股票数据
      - akshare批量预筛选
      - 断点续传
      - 盘中实时K线拼接
      - 行业过滤 + 入场质量评分过滤
    """

    def __init__(self, best_params, result_dir=None, max_workers=10, scan_timeout_sec=900):
        self.best_params = best_params
        self.max_workers = max_workers
        self.scan_timeout_sec = scan_timeout_sec  # 全局扫描超时（秒），默认15分钟
        self.result_dir = result_dir or Path(__file__).parent.parent.parent / "results" / "v63_daily"
        self.result_dir = Path(self.result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self._status_file = self.result_dir / "scan_status.json"

    def _load_scan_status(self):
        """加载断点续传状态"""
        if self._status_file.exists():
            try:
                with open(self._status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                completed = set(data.get("completed", []))
                logger.info(f"断点续传: 发现{len(completed)}只已完成股票")
                return completed
            except Exception as e:
                logger.warning(f"断点续传状态加载失败: {e}")
        return set()

    def _save_scan_status(self, completed_set):
        """保存断点续传状态"""
        try:
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._status_file, "w", encoding="utf-8") as f:
                json.dump({"completed": list(completed_set)}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"断点续传状态保存失败: {e}")

    def _clear_scan_status(self):
        """扫描完成后删除状态文件"""
        try:
            if self._status_file.exists():
                self._status_file.unlink()
                logger.info("断点续传状态文件已清理")
        except Exception as e:
            logger.warning(f"断点续传状态文件清理失败: {e}")

    def _flush_pending_writes(self):
        """扫描结束后，单线程批量补全缓存缺失的股票数据"""
        from classic_ta.stock_data_duckdb import _pending_write_queue, batch_update_stocks

        pending_codes = []
        while not _pending_write_queue.empty():
            try:
                ts_code = _pending_write_queue.get_nowait()
                pending_codes.append(ts_code)
            except Exception:
                break

        if pending_codes:
            # 去重
            pending_codes = list(set(pending_codes))
            logger.info(f"缓存补全: {len(pending_codes)}只股票需要更新")
            batch_update_stocks(pending_codes)
        else:
            logger.info("缓存完整，无需补全")

    def scan(self, industry_allow_matrix=None, industry_map=None,
             prefilter_df=None, realtime_quotes=None,
             oamv_weekly_allowed_dates=None):
        """全市场扫描潜伏信号

        Args:
            industry_allow_matrix: 行业允许买入矩阵
            industry_map: {ts_code: industry_name}
            prefilter_df: 预筛选后的行情DataFrame
            realtime_quotes: {ts_code: quote_dict} 实时行情
            oamv_weekly_allowed_dates: OAMV允许开仓的日期集合

        Returns:
            tuple: (signals, all_signals_data)
        """
        from classic_ta.common.stock_pool import get_all_a_stocks

        all_stocks = get_all_a_stocks()
        if not all_stocks:
            logger.warning("无法获取股票列表")
            return [], {}

        # 使用批量预筛选结果过滤股票列表
        if prefilter_df is not None:
            prefilter_codes = set(prefilter_df["ts_code"].tolist())
            original_count = len(all_stocks)
            all_stocks = [(tc, n, ind) for tc, n, ind in all_stocks if tc in prefilter_codes]
            logger.info(f"预筛选后股票数: {len(all_stocks)}/{original_count}")

        # 断点续传
        completed_set = self._load_scan_status()
        if completed_set:
            before_count = len(all_stocks)
            all_stocks = [(tc, n, ind) for tc, n, ind in all_stocks if tc not in completed_set]
            logger.info(f"断点续传: 跳过{before_count - len(all_stocks)}只已完成股票，剩余{len(all_stocks)}只")

        is_intraday = realtime_quotes is not None and len(realtime_quotes) > 0
        total = len(all_stocks)
        # 全局扫描超时（秒）：防止在GitHub Actions中因缓存冷启动导致扫描超30分钟被取消
        scan_timeout_sec = getattr(self, 'scan_timeout_sec', 900)  # 默认15分钟

        print(f"  🔍 扫描开始: {total}只股票 | 并发:{self.max_workers} | "
              f"模式:{'盘中实时' if is_intraday else '盘后完整'} | "
              f"超时:{scan_timeout_sec//60}min", flush=True)
        logger.info(f"扫描股票数: {total} | 并发数: {self.max_workers} | 模式: {'盘中实时' if is_intraday else '盘后完整'}")

        signals = []
        all_signals_data = {}
        processed = 0
        errors = 0
        cache_miss = 0
        start_time = time.time()
        timed_out = False

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for ts_code, name, industry in all_stocks:
                rt_quote = realtime_quotes.get(ts_code) if is_intraday else None
                future = executor.submit(_fetch_and_process_one, ts_code, name, industry,
                                         self.best_params, rt_quote)
                futures[future] = (ts_code, name, industry)

            for future in as_completed(futures):
                ts_code, name, industry = futures[future]
                processed += 1

                # 进度输出（每100只 + flush，确保GitHub Actions可见）
                if processed % 100 == 0 or processed == total:
                    elapsed = time.time() - start_time
                    speed = processed / elapsed if elapsed > 0 else 0
                    eta = (total - processed) / speed if speed > 0 else 0
                    print(f"  📊 进度: {processed}/{total} ({processed/total*100:.0f}%) | "
                          f"信号:{len(signals)} 缓存缺失:{cache_miss} 错误:{errors} | "
                          f"速度:{speed:.0f}只/s | ETA:{eta:.0f}s | 已耗时:{elapsed:.0f}s", flush=True)

                # 超时检查
                if time.time() - start_time > scan_timeout_sec:
                    print(f"  ⚠️ 扫描超时({scan_timeout_sec//60}min)，已处理{processed}/{total}只，"
                          f"提前终止（已发现{len(signals)}只信号）", flush=True)
                    timed_out = True
                    # 取消尚未开始的futures
                    for f in futures:
                        f.cancel()
                    break

                try:
                    result_ts_code, result_name, result_industry, df = future.result(timeout=30)
                except Exception:
                    errors += 1
                    completed_set.add(ts_code)
                    continue

                if df is None:
                    if result_ts_code:  # 缓存缺失（非异常）
                        cache_miss += 1
                    else:
                        errors += 1
                    completed_set.add(ts_code)
                    continue

                all_signals_data[ts_code] = df
                signal_info = _extract_signal_info(ts_code, name, industry, df, self.best_params)
                if signal_info is not None:
                    # 行业过滤
                    if industry_allow_matrix is not None and industry and industry in industry_allow_matrix.columns:
                        try:
                            signal_date = pd.Timestamp(signal_info["signal_date"])
                            ind_val = industry_allow_matrix[industry].reindex([signal_date])
                            if not ind_val.empty and not ind_val.iloc[0]:
                                continue
                        except Exception:
                            pass

                    # OAMV日期过滤
                    if oamv_weekly_allowed_dates is not None:
                        signal_date = pd.Timestamp(signal_info["signal_date"])
                        if signal_date not in oamv_weekly_allowed_dates:
                            continue

                    signals.append(signal_info)
                    print(f"  🎯 潜伏信号: {name}({ts_code}) [{industry}] "
                          f"{signal_info['price']:.2f} J:{signal_info['J']:.1f} "
                          f"量比:{signal_info['vol_ratio']:.2f} 评分:{signal_info['entry_quality_score']}", flush=True)

        elapsed = time.time() - start_time
        timeout_note = " (超时截断)" if timed_out else ""
        print(f"  ✅ 扫描完成{timeout_note}: 耗时{elapsed/60:.1f}min | "
              f"处理{processed}/{total} | 信号:{len(signals)}只 | "
              f"缓存缺失:{cache_miss} | 错误:{errors}", flush=True)
        logger.info(f"扫描完成! 耗时: {elapsed / 60:.1f}min | 信号: {len(signals)}只 | 错误: {errors}")

        # 清理线程本地DuckDB连接池
        try:
            from classic_ta.stock_data_duckdb import close_thread_local_conns
            close_thread_local_conns()
        except Exception:
            pass

        self._clear_scan_status()
        # 扫描完成后，单线程批量补全缓存缺失
        self._flush_pending_writes()
        return signals, all_signals_data


# ══════════════════════════════════════════════════════════
#  异步扫描器
# ══════════════════════════════════════════════════════════

class AsyncScanner:
    """基于asyncio+aiohttp的异步全市场扫描器

    相比SyncScanner的优势：
      1. 异步I/O：数据获取期间不占线程，同样并发数下内存占用更低
      2. 智能预筛选：先批量获取实时行情，快速排除不满足条件的股票
      3. 分批处理：避免一次性提交过多任务导致内存溢出
      4. 实时进度：更细粒度的进度报告

    用法:
        scanner = AsyncScanner(best_params, max_concurrent=20)
        signals, all_data = await scanner.scan()
    """

    def __init__(self, best_params, result_dir=None, max_concurrent=20, batch_size=500):
        self.best_params = best_params
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.result_dir = result_dir or Path(__file__).parent.parent.parent / "results" / "v63_daily"
        self.result_dir = Path(self.result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def _flush_pending_writes(self):
        """扫描结束后，单线程批量补全缓存缺失的股票数据"""
        from classic_ta.stock_data_duckdb import _pending_write_queue, batch_update_stocks

        pending_codes = []
        while not _pending_write_queue.empty():
            try:
                ts_code = _pending_write_queue.get_nowait()
                pending_codes.append(ts_code)
            except Exception:
                break

        if pending_codes:
            # 去重
            pending_codes = list(set(pending_codes))
            logger.info(f"缓存补全: {len(pending_codes)}只股票需要更新")
            batch_update_stocks(pending_codes)
        else:
            logger.info("缓存完整，无需补全")

    async def _async_fetch_one(self, ts_code, name, industry, session=None):
        """异步获取单只股票数据

        由于akshare/tushare本身是同步库，这里使用run_in_executor
        将同步I/O放到线程池中执行，同时用asyncio.Semaphore控制并发。
        """
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,  # 默认线程池
                _fetch_and_process_one,
                ts_code, name, industry, self.best_params, None
            )
            return result
        except Exception as e:
            logger.warning(f"异步获取异常 {ts_code}({name}): {e}")
            return (ts_code, name, industry, None)

    async def _process_batch(self, stocks, semaphore, progress_callback=None):
        """处理一批股票

        Args:
            stocks: [(ts_code, name, industry), ...]
            semaphore: asyncio.Semaphore 控制并发数
            progress_callback: 可选的进度回调函数
        """
        tasks = []
        for ts_code, name, industry in stocks:
            task = self._process_one_with_semaphore(ts_code, name, industry, semaphore)
            tasks.append(task)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _process_one_with_semaphore(self, ts_code, name, industry, semaphore):
        """带信号量控制的单只股票处理"""
        async with semaphore:
            return await self._async_fetch_one(ts_code, name, industry)

    async def scan(self, industry_allow_matrix=None, industry_map=None,
                   prefilter_df=None, realtime_quotes=None,
                   oamv_weekly_allowed_dates=None):
        """异步全市场扫描

        Args:
            与SyncScanner.scan()相同

        Returns:
            tuple: (signals, all_signals_data)
        """
        from classic_ta.common.stock_pool import get_all_a_stocks

        all_stocks = get_all_a_stocks()
        if not all_stocks:
            logger.warning("无法获取股票列表")
            return [], {}

        # 预筛选
        if prefilter_df is not None:
            prefilter_codes = set(prefilter_df["ts_code"].tolist())
            original_count = len(all_stocks)
            all_stocks = [(tc, n, ind) for tc, n, ind in all_stocks if tc in prefilter_codes]
            logger.info(f"预筛选后股票数: {len(all_stocks)}/{original_count}")

        total = len(all_stocks)
        logger.info(f"异步扫描股票数: {total} | 最大并发: {self.max_concurrent} | 批次大小: {self.batch_size}")

        signals = []
        all_signals_data = {}
        processed = 0
        errors = 0
        start_time = time.time()

        semaphore = asyncio.Semaphore(self.max_concurrent)

        # 分批处理
        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch = all_stocks[batch_start:batch_end]

            results = await self._process_batch(batch, semaphore)

            for result in results:
                processed += 1
                if isinstance(result, Exception):
                    errors += 1
                    continue

                ts_code, name, industry, df = result
                if df is None:
                    errors += 1
                    continue

                all_signals_data[ts_code] = df
                signal_info = _extract_signal_info(ts_code, name, industry, df, self.best_params)
                if signal_info is not None:
                    # 行业过滤
                    if industry_allow_matrix is not None and industry and industry in industry_allow_matrix.columns:
                        try:
                            signal_date = pd.Timestamp(signal_info["signal_date"])
                            ind_val = industry_allow_matrix[industry].reindex([signal_date])
                            if not ind_val.empty and not ind_val.iloc[0]:
                                continue
                        except Exception:
                            pass

                    signals.append(signal_info)
                    logger.info(f"潜伏信号: {name}({ts_code}) [{industry}] "
                                f"{signal_info['price']:.2f} J:{signal_info['J']:.1f} "
                                f"量比:{signal_info['vol_ratio']:.2f} 评分:{signal_info['entry_quality_score']}")

            # 批次进度
            elapsed = time.time() - start_time
            eta = elapsed / processed * (total - processed) if processed > 0 else 0
            logger.info(f"进度: {processed}/{total} ({processed / total * 100:.1f}%) | "
                        f"信号:{len(signals)} | 失败:{errors} | ETA:{eta:.0f}s")

        elapsed = time.time() - start_time
        logger.info(f"异步扫描完成! 耗时: {elapsed / 60:.1f}min | 信号: {len(signals)}只 | 错误: {errors}")

        # 异步扫描完成后，单线程批量补全缓存缺失
        self._flush_pending_writes()
        return signals, all_signals_data


def apply_dynamic_score_filter(signals, oamv_status, dynamic_score_params):
    """精细动态评分过滤

    Args:
        signals: 信号列表
        oamv_status: OAMV择时状态
        dynamic_score_params: 动态评分参数

    Returns:
        list: 过滤后的信号列表
    """
    if not signals:
        return signals

    is_bull = oamv_status and oamv_status.get("can_open_position", False)
    dsp = dynamic_score_params

    filtered = []
    for s in signals:
        j = s.get("J", 99)
        eq = s.get("entry_quality_score", 0)
        vr = s.get("vol_ratio", 1.0)

        # J值硬上限
        if j >= dsp.get("j_hard_cap", 5):
            continue

        if is_bull:
            if eq >= dsp.get("bull_min_score", 5):
                filtered.append(s)
            elif eq == 4 and j < dsp.get("bull_score4_j_max", 5) and vr < dsp.get("bull_score4_vol_ratio_max", 0.60):
                filtered.append(s)
        else:
            if eq >= dsp.get("bear_min_score", 6):
                filtered.append(s)

    return filtered
