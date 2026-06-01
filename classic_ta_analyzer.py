import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from data.akshare_data import get_stock_data_with_indicators
from classic_ta.candlestick_patterns import run_candlestick_detection
from classic_ta.volume_price_analysis import run_vpa_analysis
from classic_ta.wyckoff_analysis import run_wyckoff_analysis
from classic_ta.buy_signal_detector import run_buy_signal_detection
from classic_ta.ai_reporter import build_feature_summary, generate_analysis_report

HOT_STOCKS = [
    ("600519.SH", "贵州茅台"),
    ("601318.SH", "中国平安"),
    ("600036.SH", "招商银行"),
    ("000001.SZ", "平安银行"),
    ("000333.SZ", "美的集团"),
    ("002415.SZ", "海康威视"),
    ("300750.SZ", "宁德时代"),
    ("300059.SZ", "东方财富"),
    ("600570.SH", "恒生电子"),
    ("600362.SH", "江西铜业"),
    ("601899.SH", "紫金矿业"),
    ("601668.SH", "中国建筑"),
    ("000858.SZ", "五粮液"),
    ("000651.SZ", "格力电器"),
    ("601888.SH", "中国中免"),
]

SEP = "=" * 70


def analyze_stock(ts_code, stock_name):
    print(SEP)
    print(f"  正在分析: {stock_name} ({ts_code})")
    print(SEP)

    df = get_stock_data_with_indicators(ts_code)
    if df is None:
        print(f"  ❌ 获取 {stock_name} ({ts_code}) 数据失败，跳过")
        return None

    df = run_candlestick_detection(df)
    df = run_vpa_analysis(df)
    df = run_wyckoff_analysis(df)
    df = run_buy_signal_detection(df)

    latest = df.iloc[-1]
    if latest.get("reversal_signal", False):
        print("  🔄 ★ 检测到【抄底/反转信号】★")
    if latest.get("uptrend_signal", False):
        print("  🚀 ★ 检测到【主升浪接力信号】★")

    feature_text = build_feature_summary(df, stock_name, ts_code)

    print("-" * 70)
    print("  📋 客观特征摘要")
    print("-" * 70)
    print(feature_text)

    report = generate_analysis_report(feature_text)

    print("-" * 70)
    print("  🤖 AI 分析报告")
    print("-" * 70)
    print(report)

    return report


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print(f"  python {sys.argv[0]} single <ts_code> <stock_name>  — 分析单只股票")
        print(f"  python {sys.argv[0]} batch                         — 批量分析热门股票")
        print(f"  python {sys.argv[0]} scan [数量]                   — 全市场扫描买入信号")
        print(f"  python {sys.argv[0]} backtest [模式] [参数...]      — 回测信号收益")
        print()
        print("回测模式:")
        print(f"  python {sys.argv[0]} backtest              — 单票独立回测(20只热门股)")
        print(f"  python {sys.argv[0]} backtest portfolio    — 组合回测(等权分配+资金管理)")
        print(f"  python {sys.argv[0]} backtest oos [数量]    — 样本外测试(随机冷门股盲测)")
        print()
        print("其他示例:")
        print(f"  python {sys.argv[0]} single 600519.SH 贵州茅台")
        print(f"  python {sys.argv[0]} scan 100              — 仅扫描前100只")
        return

    mode = sys.argv[1]

    if mode == "single":
        if len(sys.argv) < 4:
            print("❌ single 模式需要提供 ts_code 和 stock_name")
            print(f"  用法: python {sys.argv[0]} single <ts_code> <stock_name>")
            return
        ts_code = sys.argv[2]
        stock_name = sys.argv[3]
        analyze_stock(ts_code, stock_name)

    elif mode == "batch":
        print(SEP)
        print("  🚀 批量分析热门A股")
        print(SEP)
        for ts_code, stock_name in HOT_STOCKS:
            analyze_stock(ts_code, stock_name)
            print()
        print(SEP)
        print("  ✅ 批量分析完成")
        print(SEP)

    elif mode == "scan":
        from classic_ta.market_scanner import scan_market
        max_n = None
        if len(sys.argv) > 2:
            try:
                max_n = int(sys.argv[2])
            except ValueError:
                pass
        scan_market(max_stocks=max_n)

    elif mode == "backtest":
        from classic_ta.backtest import run_backtest, run_oos_test
        bt_mode = "single"
        n_stocks = None
        start = "20200101"
        end = "20251231"
        if len(sys.argv) > 2:
            arg2 = sys.argv[2]
            if arg2 == "portfolio":
                bt_mode = "portfolio"
            elif arg2 == "oos":
                bt_mode = "oos"
                if len(sys.argv) > 3:
                    try:
                        n_stocks = int(sys.argv[3])
                    except ValueError:
                        pass
            else:
                try:
                    n_stocks = int(arg2)
                except ValueError:
                    pass
        if len(sys.argv) > 3 and bt_mode != "oos":
            start = sys.argv[2] if n_stocks is None else sys.argv[3]
        if len(sys.argv) > 4:
            end = sys.argv[4]

        if bt_mode == "oos":
            run_oos_test(start, end, n_stocks=n_stocks or 50)
        else:
            stocks = None if n_stocks is None else None
            run_backtest(start_date=start, end_date=end, use_market_filter=True, mode=bt_mode)

    else:
        print(f"❌ 未知模式: {mode}")
        print(f"  支持的模式: single, batch, scan, backtest")


if __name__ == "__main__":
    main()
