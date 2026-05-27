import os
import sys
import gc
import json
import warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "pip_libs"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import tushare as ts

warnings.filterwarnings('ignore')

from ml_strategy.oamv_filter import OAMVHysteresisFilter
from ml_strategy.feature_engine import FeatureDiscretizer
from ml_strategy.catboost_predictor import CatBoostPredictor
from ml_strategy.panic_breaker import MarketPanicCircuitBreaker

load_dotenv(Path(__file__).parent / ".env")

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
pro = None
if TUSHARE_TOKEN:
    try:
        pro = ts.pro_api(TUSHARE_TOKEN)
    except Exception as e:
        print(f"Tushare init failed: {e}")
        sys.exit(1)

OAMV_UPPER = 4.0
OAMV_LOWER = -2.3
CATBOOST_BUY_THRESHOLD = 0.65
MAX_PORTFOLIO_STOCKS = 3
COMMISSION_RATE = 0.0003
STAMP_DUTY_RATE = 0.001
SLIPPAGE_RATE = 0.0005
FRICTION_COST_PCT = (COMMISSION_RATE * 2 + STAMP_DUTY_RATE + SLIPPAGE_RATE * 2) * 100
MIN_AMPLITUDE_MULT = 1.5
MIN_PRICE = 3.0
MIN_HOLD_DAYS = 3
COOLDOWN_DAYS = 5
PANIC_BREADTH_THRESHOLD = 0.85
PANIC_LIMIT_DOWN_THRESHOLD = 150
PWVC_VETO_THRESHOLD = 1.5
J_OVERSOLD_THRESHOLD = 13
MODEL_DIR = Path(__file__).parent / "ml_strategy" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_DIR = Path(__file__).parent / "portfolio_state"
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)


def is_st_stock(name: str) -> bool:
    for kw in ['ST', '*ST', 'S*ST', 'SST']:
        if name.startswith(kw):
            return True
    return False


def get_index_daily(ts_code='000001.SH', start_date='20210101', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    try:
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'vol': 'Volume', 'amount': 'amount',
                   'trade_date': 'Date'}
        for old, new in col_map.items():
            if old in df.columns:
                df[new] = df[old]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        return df
    except Exception:
        return None


def get_industry_daily(ts_code, start_date='20210101', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    try:
        if ts_code.endswith('.SH') or ts_code.endswith('.SZ'):
            df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'vol': 'Volume', 'trade_date': 'Date'}
        for old, new in col_map.items():
            if old in df.columns:
                df[new] = df[old]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        return df
    except Exception:
        return None


def get_stock_daily(ts_code, start_date='20210101', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'vol': 'Volume', 'trade_date': 'Date'}
        for old, new in col_map.items():
            if old in df.columns:
                df[new] = df[old]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)
        return df
    except Exception:
        return None


def compute_indicators(df):
    df = df.copy()
    if 'Close' not in df.columns:
        return None

    df['white_line'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['white_line'] = df['white_line'].ewm(span=10, adjust=False).mean()

    df['ma14'] = df['Close'].rolling(window=14).mean()
    df['ma28'] = df['Close'].rolling(window=28).mean()
    df['ma57'] = df['Close'].rolling(window=57).mean()
    df['ma114'] = df['Close'].rolling(window=114).mean()
    df['yellow_line'] = (df['ma14'] + df['ma28'] + df['ma57'] + df['ma114']) / 4

    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(window=14, min_periods=1).mean()

    df['Vol_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()

    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema_fast - ema_slow
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2

    df['MACD_cross_up'] = False
    df['MACD_cross_down'] = False
    for i in range(1, len(df)):
        if pd.notna(df['MACD'].iloc[i]) and pd.notna(df['MACD'].iloc[i-1]):
            if df['MACD'].iloc[i] >= 0 and df['MACD'].iloc[i-1] < 0:
                df.iloc[i, df.columns.get_loc('MACD_cross_up')] = True
            if df['MACD'].iloc[i] < 0 and df['MACD'].iloc[i-1] >= 0:
                df.iloc[i, df.columns.get_loc('MACD_cross_down')] = True

    df['white_above_yellow'] = df['white_line'] > df['yellow_line']

    low_list = df['Low'].rolling(window=9, min_periods=1).min()
    high_list = df['High'].rolling(window=9, min_periods=1).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D_val'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D_val']

    df['J_below_0_recent5'] = df['J'].rolling(window=5, min_periods=1).min() < 0
    df['J_rising'] = False
    for i in range(1, len(df)):
        if pd.notna(df['J'].iloc[i]) and pd.notna(df['J'].iloc[i-1]):
            df.iloc[i, df.columns.get_loc('J_rising')] = df['J'].iloc[i] > df['J'].iloc[i-1]

    df['yellow_rising'] = False
    for i in range(1, len(df)):
        if pd.notna(df['yellow_line'].iloc[i]) and pd.notna(df['yellow_line'].iloc[i-1]):
            df.iloc[i, df.columns.get_loc('yellow_rising')] = df['yellow_line'].iloc[i] > df['yellow_line'].iloc[i-1]

    dist_pct = (df['Close'] - df['yellow_line']).abs() / df['yellow_line'] * 100
    near_yellow = (dist_pct < 2.0).astype(int)
    df['sideways'] = near_yellow.rolling(window=8, min_periods=1).sum() >= 6
    df['not_sideways'] = ~df['sideways']

    df['vol_ma5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
    df['vol_above_ma5'] = df['Volume'] > df['vol_ma5']
    df['low_above_yellow'] = df['Low'] > df['yellow_line']

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['amplitude_20'] = ((df['High'] - df['Low']) / df['Close'] * 100).rolling(window=20).mean()

    return df


def compute_industry_j(industry_df):
    if industry_df is None or len(industry_df) < 20:
        return None
    low_9 = industry_df['Low'].rolling(window=9, min_periods=9).min()
    high_9 = industry_df['High'].rolling(window=9, min_periods=9).max()
    denom = high_9 - low_9
    rsv = pd.Series(np.where(denom == 0, 50, (industry_df['Close'] - low_9) / denom * 100),
                     index=industry_df.index, dtype=float)
    rsv = rsv.fillna(50)
    k = pd.Series(np.nan, index=industry_df.index, dtype=float)
    d = pd.Series(np.nan, index=industry_df.index, dtype=float)
    k.iloc[0] = 50.0
    d.iloc[0] = 50.0
    for i in range(1, len(industry_df)):
        k.iloc[i] = 2.0 / 3.0 * k.iloc[i - 1] + 1.0 / 3.0 * rsv.iloc[i]
        d.iloc[i] = 2.0 / 3.0 * d.iloc[i - 1] + 1.0 / 3.0 * k.iloc[i]
    j = 3.0 * k - 2.0 * d
    return j


INDUSTRY_ETF_MAP = {
    '银行': '512800.SH', '房地产': '512200.SH', '保险': '512070.SH',
    '证券': '512880.SH', '多元金融': '512880.SH',
    '食品饮料': '515170.SH', '家用电器': '159996.SZ', '纺织服饰': '159993.SZ',
    '汽车': '516110.SH', '商贸零售': '516180.SH', '社会服务': '159766.SZ',
    '医药生物': '512010.SH', '农林牧渔': '159825.SZ',
    '电子': '159997.SZ', '计算机': '512720.SH', '通信': '515880.SH', '传媒': '159805.SZ',
    '电力设备': '159611.SZ', '机械设备': '159886.SZ', '国防军工': '512660.SH',
    '基础化工': '159870.SZ', '石油石化': '159861.SZ', '钢铁': '159861.SZ',
    '有色金属': '512400.SH', '煤炭': '515220.SH', '建筑材料': '159745.SZ',
    '建筑装饰': '159745.SZ', '环保': '159861.SZ', '公用事业': '159825.SZ',
    '交通运输': '159662.SZ', '综合': '159993.SZ',
    '轻工制造': '159993.SZ', '美容护理': '159993.SZ',
}


def check_realtime_panic(trade_date=None):
    print("[Panic Check] 实时市场宽度检查...")
    if trade_date is None:
        trade_date = datetime.now().strftime('%Y%m%d')

    try:
        all_daily = pro.daily(trade_date=trade_date)
        if all_daily is None or all_daily.empty:
            print("  无法获取全市场日线数据，跳过熔断检查")
            return False, 0, 0.0

        total = len(all_daily)
        if 'pct_chg' not in all_daily.columns:
            print("  缺少pct_chg列，跳过熔断检查")
            return False, 0, 0.0

        limit_downs = (all_daily['pct_chg'] <= -9.5).sum()
        declining = (all_daily['pct_chg'] < 0).sum()
        declining_pct = declining / total if total > 0 else 0.0

        is_panic = (limit_downs >= PANIC_LIMIT_DOWN_THRESHOLD) or (declining_pct >= PANIC_BREADTH_THRESHOLD)

        print(f"  日期: {trade_date}")
        print(f"  全市场股票数: {total}")
        print(f"  跌停数: {limit_downs} (阈值: {PANIC_LIMIT_DOWN_THRESHOLD})")
        print(f"  下跌比例: {declining_pct:.1%} (阈值: {PANIC_BREADTH_THRESHOLD:.0%})")

        if is_panic:
            print(f"  >>> 熔断触发！市场进入恐慌状态！ <<<")
        else:
            print(f"  市场情绪正常")

        return is_panic, int(limit_downs), float(declining_pct)

    except Exception as e:
        print(f"  熔断检查异常: {e}，默认不触发")
        return False, 0, 0.0


def load_universe_amv(stock_limit=300):
    print("[Universe AMV] 加载股票池计算Universe AMV...")
    try:
        stock_basic = pro.stock_basic(exchange='', list_status='L',
                                       fields='ts_code,symbol,name,industry,list_date')
        a_stocks = stock_basic[
            (stock_basic['ts_code'].str.endswith('.SH')) |
            (stock_basic['ts_code'].str.endswith('.SZ'))
        ]
        a_stocks = a_stocks[~a_stocks['name'].apply(is_st_stock)]
        sample_stocks = a_stocks.head(stock_limit)

        all_stock_data = {}
        total = len(sample_stocks)
        for idx, (_, row) in enumerate(sample_stocks.iterrows()):
            ts_code = row['ts_code']
            if idx % 100 == 0:
                print(f"  AMV数据加载: {idx}/{total}")
            try:
                df = get_stock_daily(ts_code, '20240101')
                if df is not None and len(df) > 50:
                    all_stock_data[ts_code] = {'data': df, 'name': row['name'], 'industry': row.get('industry', '')}
            except Exception:
                continue

        index_df = get_index_daily('000001.SH', '20240101')
        if index_df is None:
            return None, None
        index_df = compute_indicators(index_df)

        oamv_filter = OAMVHysteresisFilter(
            upper_threshold=OAMV_UPPER,
            lower_threshold=OAMV_LOWER,
            cost_ma_period=34,
            weekly_ema_period=5,
            weekly_use_ema=True,
        )
        oamv_filter.fit(index_df, all_stock_data=all_stock_data)
        print(f"  Universe AMV 数据源: {oamv_filter.data_source}")
        return oamv_filter, index_df

    except Exception as e:
        print(f"  Universe AMV加载失败: {e}，降级为指数代理")
        index_df = get_index_daily('000001.SH', '20240101')
        if index_df is None:
            return None, None
        index_df = compute_indicators(index_df)
        oamv_filter = OAMVHysteresisFilter(
            upper_threshold=OAMV_UPPER,
            lower_threshold=OAMV_LOWER,
            cost_ma_period=34,
            weekly_ema_period=5,
            weekly_use_ema=True,
        )
        oamv_filter.fit(index_df)
        print(f"  降级数据源: {oamv_filter.data_source}")
        return oamv_filter, index_df


def find_latest_model():
    model_files = sorted(MODEL_DIR.glob("catboost_v6*.cbm"), reverse=True)
    if model_files:
        return model_files[0]
    fallback = MODEL_DIR / "catboost_latest.cbm"
    if fallback.exists():
        return fallback
    return None


def run_daily_scan(stock_limit=None, quick_mode=False):
    print("=" * 90)
    print("v8.1 每日选股扫描器 - 四层风控流水线 + 四大共识硬过滤")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    print(f"\n  Layer 1: Universe AMV 迟滞状态机 (EMA-5周线) [+4%/-2.3%]")
    print(f"  Layer 2: 全市场跌停潮熔断器 (breadth>={PANIC_BREADTH_THRESHOLD:.0%} / 跌停>={PANIC_LIMIT_DOWN_THRESHOLD})")
    print(f"  Layer 3: CatBoost概率触发器 + 四大共识硬过滤:")
    print(f"           - PWVC>{PWVC_VETO_THRESHOLD} 一票否决 (高位放量大阴线=主力出货)")
    print(f"           - white_above_yellow=True (白线在黄线上方=多头趋势)")
    print(f"           - J<{J_OVERSOLD_THRESHOLD} 情绪冰点 (左侧便宜筹码)")
    print(f"  Layer 4: 动态交易成本阀门 (振幅>={FRICTION_COST_PCT * MIN_AMPLITUDE_MULT:.2f}%)")

    print("\n" + "=" * 90)
    print("[Layer 1] 检查0AMV大势状态 + EMA-5周线过滤器")
    print("-" * 90)

    amv_stock_limit = 200 if quick_mode else 300
    oamv_filter, index_df = load_universe_amv(amv_stock_limit)

    if oamv_filter is None:
        print("0AMV初始化失败！")
        return None

    print(oamv_filter.summary())

    oamv_state_df = oamv_filter.get_state_df()
    latest_date = oamv_state_df.index[-1]
    latest_state = int(oamv_state_df.iloc[-1]['oamv_state'])
    latest_x = float(oamv_state_df.iloc[-1]['oamv_x'])
    weekly_ok = oamv_filter.is_trading_allowed(latest_date, require_weekly=True)
    daily_ok = oamv_filter.is_trading_allowed(latest_date, require_weekly=False)

    state_label = "BULL (做多绿灯)" if latest_state == 1 else "BEAR (全部清仓)"
    weekly_label = "BULL" if weekly_ok else "BEAR"
    print(f"\n  最新日期: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  日线0AMV: {state_label} (X={latest_x:+.2f}%)")
    print(f"  周线EMA-5: {weekly_label}")

    if not daily_ok:
        print("\n  >>> 日线0AMV = BEAR，系统关闭！所有持仓建议清仓！ <<<")
        transitions = oamv_filter.get_transition_dates()
        if transitions:
            last_t = transitions[-1]
            print(f"  最近转换: {last_t['date'].strftime('%Y-%m-%d')} "
                  f"State {last_t['from']} -> {last_t['to']} (X={last_t['x_value']:.2f}%)")
        return {'oamv_state': 0, 'oamv_x': latest_x, 'weekly_ok': False, 'panic': False, 'candidates': []}

    if not weekly_ok:
        print("\n  >>> 周线EMA-5 = BEAR，禁止新开仓！仅允许持仓管理！ <<<")

    print("\n" + "=" * 90)
    print("[Layer 2] 全市场跌停潮熔断器")
    print("-" * 90)

    is_panic, limit_downs, declining_pct = check_realtime_panic(latest_date.strftime('%Y%m%d'))

    if is_panic:
        print("\n  >>> 熔断触发！强制关闭选股通道！所有持仓建议减仓/清仓！ <<<")
        return {
            'oamv_state': latest_state,
            'oamv_x': latest_x,
            'weekly_ok': weekly_ok,
            'panic': True,
            'limit_downs': limit_downs,
            'declining_pct': declining_pct,
            'candidates': [],
        }

    if not weekly_ok:
        print("\n  >>> 周线BEAR，跳过选股扫描 <<<")
        return {
            'oamv_state': latest_state,
            'oamv_x': latest_x,
            'weekly_ok': False,
            'panic': False,
            'candidates': [],
        }

    print("\n" + "=" * 90)
    print("[Layer 3] CatBoost 概率触发器 - 全市场扫描")
    print("-" * 90)

    print("\n  加载行业ETF数据...")
    industry_j_cache = {}
    for ind_name, etf_code in INDUSTRY_ETF_MAP.items():
        try:
            ind_df = get_industry_daily(etf_code, '20240101')
            if ind_df is not None and len(ind_df) > 20:
                ind_j = compute_industry_j(ind_df)
                if ind_j is not None:
                    industry_j_cache[ind_name] = ind_j
        except Exception:
            pass
    print(f"  加载 {len(industry_j_cache)} 个行业ETF")

    model_path = find_latest_model()
    if model_path is None:
        print("  模型文件不存在！请先运行 run_v6_2.py 训练模型")
        return None

    model = CatBoostPredictor(buy_threshold=CATBOOST_BUY_THRESHOLD, l2_leaf_reg=8, max_depth=4)
    model.load_model(str(model_path))
    print(f"  模型加载成功: {model_path.name}")

    stock_basic = pro.stock_basic(exchange='', list_status='L',
                                   fields='ts_code,symbol,name,industry,list_date')
    a_stocks = stock_basic[
        (stock_basic['ts_code'].str.endswith('.SH')) |
        (stock_basic['ts_code'].str.endswith('.SZ'))
    ]
    a_stocks = a_stocks[~a_stocks['name'].apply(is_st_stock)]
    if stock_limit:
        a_stocks = a_stocks.head(stock_limit)
    print(f"  扫描股票数: {len(a_stocks)} (已过滤ST)")

    discretizer = FeatureDiscretizer()
    feature_cols = CatBoostPredictor.FEATURE_COLS
    buy_candidates = []
    scan_count = 0
    error_count = 0

    for idx, (_, row) in enumerate(a_stocks.iterrows()):
        ts_code = row['ts_code']
        name = row['name']
        industry = row.get('industry', '')

        if idx % 200 == 0:
            print(f"  扫描进度: {idx}/{len(a_stocks)}, 候选: {len(buy_candidates)}")

        try:
            df = get_stock_daily(ts_code, '20240101')
            if df is None or len(df) < 200:
                continue

            if df['Close'].iloc[-1] < MIN_PRICE:
                continue

            df = compute_indicators(df)
            if df is None:
                continue

            featured_df = discretizer.transform(df)
            industry_j = industry_j_cache.get(industry)
            featured_df = discretizer.add_market_context(featured_df, index_df, None, industry_j)

            latest_row = df.iloc[-1]
            latest_feat = featured_df.iloc[-1]

            available_cols = [c for c in feature_cols if c in latest_feat.index]
            if any(pd.isna(latest_feat.get(c)) for c in available_cols):
                continue

            amp = latest_row.get('amplitude_20', 0)
            if pd.notna(amp) and amp > 0 and amp < FRICTION_COST_PCT * MIN_AMPLITUDE_MULT:
                continue

            clean_data = latest_feat[available_cols].to_frame().T
            for col in ['price_zone', 'j_zone', 'k_pattern']:
                if col in clean_data.columns:
                    clean_data[col] = clean_data[col].astype(int)

            prob = model.predict_proba(clean_data)[0]

            if prob >= CATBOOST_BUY_THRESHOLD:
                white_above = latest_row.get('white_above_yellow', False)
                if pd.notna(white_above) and not white_above:
                    continue

                j_val = latest_row.get('J', 100)
                if pd.notna(j_val) and j_val >= J_OVERSOLD_THRESHOLD:
                    continue

                pwvc_val = latest_feat.get('pwvc', 0.0)
                if pd.notna(pwvc_val) and pwvc_val > PWVC_VETO_THRESHOLD:
                    continue

                rsi_val = latest_row.get('RSI', 0)
                white_above = latest_row.get('white_above_yellow', False)
                yellow_rise = latest_row.get('yellow_rising', False)
                dist_to_yellow = (latest_row['Close'] - latest_row['yellow_line']) / latest_row['yellow_line'] * 100
                atr_val = latest_row.get('ATR14', 0)
                stop_loss_price = latest_row['Close'] - 1.5 * atr_val if atr_val > 0 else 0
                trailing_stop_price = latest_row['Close'] * 0.92
                amplitude_val = amp if pd.notna(amp) else 0

                buy_candidates.append({
                    'code': ts_code,
                    'name': name,
                    'industry': industry,
                    'prob': float(prob),
                    'price': float(latest_row['Close']),
                    'J': float(j_val) if not pd.isna(j_val) else 0,
                    'RSI': float(rsi_val) if not pd.isna(rsi_val) else 0,
                    'white_above_yellow': bool(white_above),
                    'yellow_rising': bool(yellow_rise),
                    'dist_to_yellow': float(dist_to_yellow),
                    'atr': float(atr_val) if not pd.isna(atr_val) else 0,
                    'amplitude_20': float(amplitude_val),
                    'stop_loss': float(max(stop_loss_price, trailing_stop_price)),
                    'pwvc': float(pwvc_val) if not pd.isna(pwvc_val) else 0,
                    'accumulation_score': float(latest_feat.get('accumulation_score', 0)),
                })

            scan_count += 1
            del df, featured_df

        except Exception:
            error_count += 1
            continue

        if idx % 500 == 0:
            gc.collect()

    print(f"\n  扫描完成: {scan_count} 成功, {error_count} 失败")
    print(f"  符合条件的候选股: {len(buy_candidates)}")

    buy_candidates.sort(key=lambda x: -x['prob'])
    top_candidates = buy_candidates[:10]

    print("\n" + "=" * 90)
    print("v6.2 每日选股结果")
    print("=" * 90)

    if not top_candidates:
        print("\n  今日无符合条件的股票，建议空仓观望。")
    else:
        print(f"\n  Top-{len(top_candidates)} 候选股 (按CatBoost概率排序):")
        print()
        print(f"  {'#':>3s}  {'代码':>10s}  {'名称':<8s}  {'行业':<8s}  "
              f"{'概率':>6s}  {'现价':>7s}  {'J值':>6s}  {'RSI':>5s}  "
              f"{'白>黄':>4s}  {'黄升':>4s}  {'距黄线':>6s}  {'振幅':>5s}  {'PWVC':>5s}  {'建仓分':>5s}  {'止损价':>7s}")
        print("  " + "-" * 115)

        for i, c in enumerate(top_candidates, 1):
            wy = 'Y' if c['white_above_yellow'] else 'N'
            yr = 'Y' if c['yellow_rising'] else 'N'
            print(f"  {i:>3d}  {c['code']:>10s}  {c['name']:<8s}  {c['industry']:<8s}  "
                  f"{c['prob']:>5.1%}  {c['price']:>7.2f}  {c['J']:>6.1f}  {c['RSI']:>5.1f}  "
                  f"{wy:>4s}  {yr:>4s}  {c['dist_to_yellow']:>+5.1f}%  {c['amplitude_20']:>4.1f}%  "
                  f"{c['pwvc']:>5.2f}  {c['accumulation_score']:>5.2f}  {c['stop_loss']:>7.2f}")

        print(f"\n  操作建议:")
        print(f"  - 最多买入前{MAX_PORTFOLIO_STOCKS}只 (概率最高的)")
        print(f"  - 每只仓位不超过总资金的25%")
        print(f"  - 止损价: max(买入价 - 1.5xATR, 买入价 x 0.92)")
        print(f"  - 止盈: 从最高点回撤8%时止盈")
        print(f"  - 清仓: 0AMV日线=BEAR 或 周线=BEAR 或 熔断触发 或 收盘跌破黄线")
        print(f"  - 四大共识硬过滤: PWVC>{PWVC_VETO_THRESHOLD}否决 / 白>黄 / J<{J_OVERSOLD_THRESHOLD}")

    result = {
        'scan_date': latest_date.strftime('%Y-%m-%d'),
        'scan_time': datetime.now().strftime('%H:%M:%S'),
        'oamv_state': latest_state,
        'oamv_x': latest_x,
        'weekly_ok': weekly_ok,
        'panic': is_panic,
        'limit_downs': limit_downs,
        'declining_pct': declining_pct,
        'total_scanned': scan_count,
        'total_candidates': len(buy_candidates),
        'top_candidates': top_candidates,
        'config': {
            'version': 'v8.1',
            'oamv_source': oamv_filter.data_source,
            'weekly_filter': 'EMA-5',
            'panic_threshold': f'breadth>={PANIC_BREADTH_THRESHOLD:.0%} OR limit_down>={PANIC_LIMIT_DOWN_THRESHOLD}',
            'catboost': f'l2_reg=8, depth=4, threshold={CATBOOST_BUY_THRESHOLD}',
            'friction_gate': f'amplitude>={FRICTION_COST_PCT * MIN_AMPLITUDE_MULT:.2f}%',
            'consensus_filters': f'PWVC>{PWVC_VETO_THRESHOLD} veto / white_above_yellow / J<{J_OVERSOLD_THRESHOLD}',
        }
    }

    output_dir = Path(__file__).parent / "daily_scan_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_file = output_dir / f"scan_v81_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存: {result_file}")
    print("=" * 90)

    return result


if __name__ == "__main__":
    limit = None
    quick = False
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'quick':
            quick = True
        else:
            limit = int(arg)
    run_daily_scan(stock_limit=limit, quick_mode=quick)
