"""全市场活跃市值缓存模块

通过 tushare daily_basic(trade_date=...) 逐日聚合全市场活跃市值，
结果缓存到本地CSV文件，后续运行只增量获取新日期。

活跃市值 = Σ(每只股票的 circ_mv × turnover_rate / 100)
"""
import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

# 缓存文件路径
CACHE_DIR = Path(__file__).parent.parent / "results" / "oamv_cache"
CACHE_FILE = CACHE_DIR / "market_amv_cache.csv"


def _get_pro():
    import tushare as ts
    from dotenv import load_dotenv
    import os
    # 显式加载 .env 中的 token
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)
    token = os.getenv("TUSHARE_TOKEN")
    if token and token != "your_tushare_token_here":
        ts.set_token(token)
    return ts.pro_api()


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache():
    """加载缓存数据"""
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, index_col='trade_date', parse_dates=True)
        logger.info(f"加载活跃市值缓存: {len(df)} 天, {df.index[0]} ~ {df.index[-1]}")
        return df
    return None


def save_cache(df):
    """保存缓存数据"""
    _ensure_cache_dir()
    df.to_csv(CACHE_FILE)
    logger.info(f"活跃市值缓存已保存: {len(df)} 天")


def fetch_market_amv(start_date=None, end_date=None, existing_cache=None, max_age_days=730):
    """获取全市场活跃市值时间序列

    参数:
        start_date: 起始日期 (YYYYMMDD), 默认2年前
        end_date: 结束日期 (YYYYMMDD), 默认今天
        existing_cache: 已有缓存DataFrame, 用于增量更新
        max_age_days: 缓存最大保留天数, 默认730(2年), 设为0保留全部

    返回:
        pd.DataFrame, 索引为trade_date, 列包含 amv_circ (全市场活跃市值)
    """
    pro = _get_pro()

    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=730)).strftime('%Y%m%d')

    # 获取交易日历
    cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
    cal = cal[cal['is_open'] == 1]
    trade_dates = sorted(cal['cal_date'].tolist())

    if len(trade_dates) == 0:
        logger.warning("交易日历为空")
        return existing_cache

    # 确定需要获取的日期 (增量)
    if existing_cache is not None and len(existing_cache) > 0:
        cached_dates = set(existing_cache.index.strftime('%Y%m%d'))
        # 也接受 Timestamp 格式
        cached_dates.update(set(pd.to_datetime(existing_cache.index).strftime('%Y%m%d')))
        dates_to_fetch = [d for d in trade_dates if d not in cached_dates]
    else:
        dates_to_fetch = trade_dates

    if len(dates_to_fetch) == 0:
        logger.info("活跃市值缓存已是最新，无需增量获取")
        return existing_cache

    logger.info(f"需要获取 {len(dates_to_fetch)} 天的活跃市值数据")

    # 逐日获取
    new_results = []
    for i, td in enumerate(dates_to_fetch):
        try:
            db = pro.daily_basic(
                trade_date=td,
                fields='ts_code,trade_date,circ_mv,turnover_rate_f'
            )
            if db is not None and len(db) > 0:
                db['circ_mv'] = pd.to_numeric(db['circ_mv'], errors='coerce')
                db['turnover_rate_f'] = pd.to_numeric(db['turnover_rate_f'], errors='coerce')
                amv = (db['circ_mv'] * db['turnover_rate_f'] / 100.0).sum()
                new_results.append({
                    'trade_date': td,
                    'amv': amv,
                    'stock_count': len(db),
                })
            else:
                logger.warning(f"  {td}: daily_basic 返回空数据")

            # 每50天打印进度
            if (i + 1) % 50 == 0:
                logger.info(f"  活跃市值获取进度: {i+1}/{len(dates_to_fetch)}")

            # 限速
            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"  {td}: 获取失败 {e}")
            time.sleep(1)

    if len(new_results) == 0:
        logger.warning("未获取到任何新数据")
        return existing_cache

    # 合并新数据
    new_df = pd.DataFrame(new_results)
    new_df['trade_date'] = pd.to_datetime(new_df['trade_date'], format='%Y%m%d')
    new_df = new_df.set_index('trade_date')

    if existing_cache is not None and len(existing_cache) > 0:
        # 确保索引类型一致
        existing_cache.index = pd.to_datetime(existing_cache.index)
        # 合并: 用新数据覆盖旧数据
        combined = pd.concat([existing_cache, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        combined = new_df

    # 限制缓存最大天数 (max_age_days=0 保留全部)
    if max_age_days > 0:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
        combined = combined[combined.index >= cutoff]

    # 保存缓存
    save_cache(combined)

    return combined


def _merge_amv_columns(df):
    """合并 amv 和 amv_circ 列，优先用 amv，缺失时用 amv_circ"""
    if 'amv' in df.columns and 'amv_circ' in df.columns:
        df['amv'] = df['amv'].fillna(df['amv_circ'])
    elif 'amv_circ' in df.columns and 'amv' not in df.columns:
        df['amv'] = df['amv_circ']
    return df


def get_market_amv_series():
    """获取全市场活跃市值时间序列 (带缓存)

    返回:
        pd.Series, 索引为trade_date, 值为全市场活跃市值
    """
    # 尝试加载缓存
    cache = load_cache()

    # 增量更新
    df = fetch_market_amv(existing_cache=cache)

    if df is None or len(df) == 0:
        return None

    df = _merge_amv_columns(df)
    return df['amv'].dropna()


def get_market_amv_from_duckdb(start_date="20200101", end_date=None):
    """从 DuckDB 缓存聚合全市场日成交额作为 OAMV 代理序列

    原理: AMV = Σ(circ_mv × turnover_rate_f / 100)
        其中 circ_mv = close × 流通股本, turnover_rate_f ≈ volume / 流通股本
        故 AMV ≈ Σ(close × volume) = 全市场日成交额
    OAMV 滤波器做 z-score 标准化, 绝对值不重要, 故成交额代理完全等价。

    优势: 无需 tushare, 无限流; eltdx 缓存即前复权日线; 毫秒级查询。

    数据修复: eltdx 数据源存在 volume 单位不一致问题 (部分股票/时段为"手",
    另一些为"股", 差100倍)。通过检测 AMV 序列的日环比突变点 (>10倍) 并
    缩放归一化, 使序列连续, 不受单位变化影响。

    参数:
        start_date: 起始日期 (YYYYMMDD), 默认 2020-01-01
        end_date: 结束日期 (YYYYMMDD), 默认今天

    返回:
        pd.Series, 索引为 trade_date, 值为归一化后的全市场日成交额
    """
    try:
        import duckdb
        from classic_ta.stock_data_duckdb import DUCKDB_PATH
        if not DUCKDB_PATH.exists():
            logger.warning("DuckDB 缓存文件不存在, 无法获取 OAMV 代理序列")
            return None

        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y%m%d')

        start_ts = pd.Timestamp(start_date).strftime('%Y-%m-%d')
        end_ts = pd.Timestamp(end_date).strftime('%Y-%m-%d')

        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        rows = conn.execute("""
            SELECT date, SUM(close * volume) AS amv
            FROM daily_data
            WHERE date BETWEEN ? AND ?
              AND volume > 0 AND close IS NOT NULL
            GROUP BY date
            ORDER BY date
        """, [start_ts, end_ts]).fetchall()
        conn.close()

        if not rows:
            logger.warning("DuckDB 聚合结果为空")
            return None

        dates = [r[0] for r in rows]
        amvs = [float(r[1]) for r in rows]
        series = pd.Series(amvs, index=pd.to_datetime(dates), name='amv')

        # ── 修复 volume 单位不一致 ──
        # eltdx 数据源部分股票/时段 volume 单位为"股"而非"手"(差100倍),
        # 导致 AMV 序列出现 10-100 倍的日环比跳变。检测并缩放归一化。
        series = _normalize_amv_unit_jumps(series)

        logger.info(f"DuckDB OAMV 代理序列: {len(series)}天, "
                    f"{series.index[0].strftime('%Y-%m-%d')} ~ "
                    f"{series.index[-1].strftime('%Y-%m-%d')}")
        return series
    except Exception as e:
        logger.warning(f"DuckDB OAMV 代理序列获取失败: {e}")
        return None


def _normalize_amv_unit_jumps(series, jump_threshold=10.0):
    """修复 AMV 序列中的单位跳变

    检测日环比超过 jump_threshold 倍的突变点, 将突变后的序列缩放,
    使其与突变前的水平一致。这样得到的序列连续, 不受 volume 单位变化影响。

    参数:
        series: pd.Series, AMV 时间序列
        jump_threshold: 突变检测阈值 (日环比>10倍视为单位跳变)

    返回:
        pd.Series, 归一化后的连续序列
    """
    if len(series) < 3:
        return series

    values = series.values.astype(float).copy()
    corrections = []

    # 逐日检测环比跳变
    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]
        if prev <= 0 or curr <= 0:
            continue
        ratio = curr / prev
        if ratio > jump_threshold:
            # 突然放大: 缩小后续数据
            scale = 1.0 / ratio
            values[i:] *= scale
            corrections.append((series.index[i], ratio, 'down', scale))
        elif ratio < 1.0 / jump_threshold:
            # 突然缩小: 放大后续数据
            scale = 1.0 / ratio
            values[i:] *= scale
            corrections.append((series.index[i], ratio, 'up', scale))

    if corrections:
        for date, ratio, direction, scale in corrections:
            logger.info(f"AMV单位跳变修复: {date.strftime('%Y-%m-%d')} "
                        f"ratio={ratio:.1f}x {direction} scale={scale:.4f}")

    return pd.Series(values, index=series.index, name=series.name)


def get_market_amv_series_for_backtest(start_date="20200101", end_date=None):
    """获取全市场活跃市值时间序列 (回测专用, 保留全部历史)

    参数:
        start_date: 起始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD), 默认今天

    返回:
        pd.Series, 索引为trade_date, 值为全市场活跃市值
    """
    cache = load_cache()

    # 增量更新, max_age_days=0 保留全部历史
    df = fetch_market_amv(
        start_date=start_date, end_date=end_date,
        existing_cache=cache, max_age_days=0,
    )

    if df is None or len(df) == 0:
        return None

    df = _merge_amv_columns(df)
    return df['amv'].dropna()
