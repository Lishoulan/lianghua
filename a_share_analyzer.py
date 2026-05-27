"""
TradingAgents A 股分析使用指南
================================

### A 股代码格式说明

- **上海证券交易所 (上交所)**: .SS 后缀
  - 主板: 600xxx, 601xxx, 603xxx, 605xxx
  - 科创板: 688xxx
  - 例子: 600519.SS (贵州茅台), 688981.SS (中芯国际)

- **深圳证券交易所 (深交所)**: .SZ 后缀
  - 主板: 000xxx, 001xxx
  - 中小板: 002xxx
  - 创业板: 300xxx
  - 例子: 000001.SZ (平安银行), 300750.SZ (宁德时代)

### A 股数据源建议

TradingAgents 默认使用 yfinance，但 yfinance 对 A 股支持有限。建议：

1. **AkShare** (推荐): 免费开源的 A 股数据接口
2. **Tushare**: 需要注册 Token，免费额度够用
3. **东方财富/同花顺**: 需相应的 API 接入

### 快速开始

1. 确保已安装 TradingAgents (已完成)
2. 配置 DeepSeek API Key (已完成)
3. 使用下方脚本运行 A 股分析
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from tradingagents import TradingAgentsGraph, TradingAgentsConfig

# A 股热门股票列表
A_SHARE_TICKERS = [
    ("600519.SS", "贵州茅台"),
    ("601318.SS", "中国平安"),
    ("600036.SS", "招商银行"),
    ("000001.SZ", "平安银行"),
    ("000333.SZ", "美的集团"),
    ("002415.SZ", "海康威视"),
    ("300750.SZ", "宁德时代"),
    ("300059.SZ", "东方财富"),
    ("688981.SS", "中芯国际"),
    ("601888.SS", "中国中免"),
]

def analyze_a_share(ticker, ticker_name, date_str="2024-12-31"):
    """使用 TradingAgents 分析单只 A 股"""
    print(f"\n{'='*60}")
    print(f"正在分析: {ticker_name} ({ticker})")
    print(f"{'='*60}")

    try:
        config = TradingAgentsConfig(
            llm_provider="litellm",
            deep_think_llm="deepseek/deepseek-chat",
            quick_think_llm="deepseek/deepseek-chat",
            max_debate_rounds=2,
            max_risk_discuss_rounds=1,
            max_recur_limit=100,
            response_language="zh-CN",
        )

        ta = TradingAgentsGraph(debug=True, config=config)
        _, decision = ta.propagate(ticker, date_str)

        print(f"\n{'='*60}")
        print("分析完成!")
        print(f"{'='*60}")
        print(str(decision))

        # 保存结果
        result = {
            "ticker": ticker,
            "ticker_name": ticker_name,
            "date": date_str,
            "decision": str(decision),
            "status": "success",
        }

        output_dir = Path(__file__).parent / "results" / "a_share_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{ticker.replace('.', '_')}_{date_str}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_file}")

        return result

    except Exception as e:
        print(f"\n分析出错: {e}")
        print("\n💡 提示: 可能是 yfinance 数据限制问题。")
        print("   如需完整分析，请考虑:")
        print("   1. 等待一段时间后重试 (yfinance 有限流)")
        print("   2. 使用 A 股专门数据源 (如 AkShare)")

        return {
            "ticker": ticker,
            "ticker_name": ticker_name,
            "date": date_str,
            "error": str(e),
            "status": "error",
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "single" and len(sys.argv) >= 4:
            ticker = sys.argv[2]
            name = sys.argv[3] if len(sys.argv) > 4 else ticker
            date = sys.argv[4] if len(sys.argv) > 5 else "2024-12-31"
            analyze_a_share(ticker, name, date)
        elif sys.argv[1] == "list":
            print("热门 A 股列表:")
            for ticker, name in A_SHARE_TICKERS:
                print(f"  {ticker}: {name}")
    else:
        print("TradingAgents A 股分析工具")
        print("\n使用方法:")
        print("  python a_share_analyzer.py list                  # 查看热门 A 股列表")
        print("  python a_share_analyzer.py single <代码> <名称>  # 分析单只股票")
        print("\n例子:")
        print("  python a_share_analyzer.py single 600519.SS 贵州茅台")
        print("  python a_share_analyzer.py single 300750.SZ 宁德时代")

