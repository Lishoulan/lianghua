import os
import time
import requests
from typing import Dict, List, Optional


class LLMStockAnalyzer:
    def __init__(self, api_key=None, base_url=None, model=None):
        self._api_key = api_key
        self._base_url = base_url
        self.model = model or 'deepseek-v4-pro'
        self.timeout = 60
        self.max_retries = 3
        self.retry_delay = 2

    @property
    def api_key(self):
        key = self._api_key or os.getenv('DEEPSEEK_API_KEY', '')
        if not key:
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                env_path = Path(__file__).resolve().parent.parent / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=env_path, override=True)
                    key = os.getenv('DEEPSEEK_API_KEY', '')
            except Exception:
                pass
        return key

    @property
    def base_url(self):
        url = self._base_url or os.getenv('OPENAI_BASE_URL', '')
        if not url:
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                env_path = Path(__file__).resolve().parent.parent / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=env_path, override=True)
                    url = os.getenv('OPENAI_BASE_URL', 'https://api.deepseek.com')
            except Exception:
                url = 'https://api.deepseek.com'
        return url

    def analyze_stock(self, ts_code: str, name: str, stock_data: dict,
                      holding_info: Optional[dict] = None,
                      buy_signal: Optional[dict] = None) -> Optional[str]:
        context = self._build_context(ts_code, name, stock_data, holding_info, buy_signal)
        prompt = self._build_prompt(ts_code, name, context)
        return self._call_api(prompt)

    def analyze_event(self, ts_code: str, name: str, news_list: List[str]) -> Optional[str]:
        if not news_list:
            return None
        prompt = self._build_event_prompt(ts_code, name, news_list)
        return self._call_api(prompt)

    def analyze_batch(self, stocks: List[Dict], stock_data_map: dict) -> Dict[str, str]:
        results = {}
        for stock in stocks:
            ts_code = stock['ts_code']
            name = stock.get('name', ts_code)
            data = stock_data_map.get(ts_code)
            if data is None:
                continue
            holding_info = stock.get('holding_info')
            buy_signal = stock.get('buy_signal')
            try:
                analysis = self.analyze_stock(ts_code, name, data, holding_info, buy_signal)
                if analysis:
                    results[ts_code] = analysis
            except Exception as e:
                results[ts_code] = f"分析失败: {str(e)}"
        return results

    def _build_context(self, ts_code: str, name: str, stock_data: dict,
                       holding_info: Optional[dict], buy_signal: Optional[dict]) -> str:
        parts = []
        parts.append(f"股票: {name}({ts_code})")

        if 'data' in stock_data:
            df = stock_data['data']
            if len(df) > 0:
                last = df.iloc[-1]
                parts.append(f"最新收盘价: {last.get('Close', 0):.2f}")

                if len(df) >= 5:
                    recent_5d = df.tail(5)
                    ret_5d = (recent_5d['Close'].iloc[-1] / recent_5d['Close'].iloc[0] - 1) * 100
                    parts.append(f"近5日涨跌幅: {ret_5d:+.2f}%")

                if len(df) >= 20:
                    recent_20d = df.tail(20)
                    ret_20d = (recent_20d['Close'].iloc[-1] / recent_20d['Close'].iloc[0] - 1) * 100
                    parts.append(f"近20日涨跌幅: {ret_20d:+.2f}%")

                if len(df) >= 60:
                    recent_60d = df.tail(60)
                    ret_60d = (recent_60d['Close'].iloc[-1] / recent_60d['Close'].iloc[0] - 1) * 100
                    parts.append(f"近60日涨跌幅: {ret_60d:+.2f}%")

                if 'Volume' in df.columns and len(df) >= 20:
                    vol_5 = df['Volume'].tail(5).mean()
                    vol_20 = df['Volume'].tail(20).mean()
                    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0
                    parts.append(f"量比(5日/20日): {vol_ratio:.2f}")

                if 'ATR14' in df.columns and 'Close' in df.columns:
                    atr = float(last.get('ATR14', 0))
                    close = float(last.get('Close', 0))
                    if close > 0 and atr > 0:
                        parts.append(f"ATR14: {atr:.2f} (波动率{atr/close*100:.1f}%)")

                if 'white_line' in df.columns and 'yellow_line' in df.columns:
                    white = last.get('white_line')
                    yellow = last.get('yellow_line')
                    if white is not None and yellow is not None:
                        w_above = "白线在黄线上方" if white > yellow else "白线在黄线下方"
                        parts.append(f"均线状态: {w_above}")

                if 'MACD' in df.columns and 'DIF' in df.columns:
                    macd = last.get('MACD')
                    dif = last.get('DIF')
                    if macd is not None and dif is not None:
                        parts.append(f"DIF: {dif:.3f}, MACD: {macd:.3f}")

                if 'J' in df.columns:
                    j_val = last.get('J')
                    if j_val is not None:
                        parts.append(f"KDJ-J值: {j_val:.1f}")

                if len(df) >= 20:
                    high_20 = df['High'].tail(20).max()
                    low_20 = df['Low'].tail(20).min()
                    close = float(last.get('Close', 0))
                    if high_20 > low_20:
                        pos_20 = (close - low_20) / (high_20 - low_20) * 100
                        parts.append(f"20日价格位置: {pos_20:.0f}% (0=最低,100=最高)")

                if len(df) >= 60:
                    high_60 = df['High'].tail(60).max()
                    low_60 = df['Low'].tail(60).min()
                    close = float(last.get('Close', 0))
                    if high_60 > low_60:
                        pos_60 = (close - low_60) / (high_60 - low_60) * 100
                        parts.append(f"60日价格位置: {pos_60:.0f}%")

                if len(df) >= 120:
                    high_120 = df['High'].tail(120).max()
                    low_120 = df['Low'].tail(120).min()
                    close = float(last.get('Close', 0))
                    if high_120 > low_120:
                        pos_120 = (close - low_120) / (high_120 - low_120) * 100
                        parts.append(f"120日价格位置: {pos_120:.0f}%")

                if len(df) >= 5:
                    recent = df.tail(5)
                    closes = recent['Close'].values
                    if len(closes) >= 2:
                        trend = "连续上涨" if all(closes[i] <= closes[i+1] for i in range(len(closes)-1)) else \
                                "连续下跌" if all(closes[i] >= closes[i+1] for i in range(len(closes)-1)) else "震荡"
                        parts.append(f"近5日趋势: {trend}")

        industry = stock_data.get('industry', '')
        if industry:
            parts.append(f"所属行业: {industry}")

        if holding_info:
            parts.append(f"\n--- 持仓信息 ---")
            parts.append(f"买入价: {holding_info.get('entry_price', 0):.2f}")
            parts.append(f"持仓天数: {holding_info.get('hold_days', 0)}天")
            parts.append(f"当前收益: {holding_info.get('profit_pct', 0):+.2f}%")
            parts.append(f"峰值回撤: {holding_info.get('dd_pct', 0):.1f}%")

        if buy_signal:
            parts.append(f"\n--- 买入信号 ---")
            parts.append(f"RADE概率: {buy_signal.get('prob', 0):.1%}")
            parts.append(f"J值: {buy_signal.get('j_val', 0):.1f}")
            parts.append(f"PWVC: {buy_signal.get('pwvc', 0):.2f}")
            parts.append(f"累积分数: {buy_signal.get('accumulation_score', 0):.2f}")

        return "\n".join(parts)

    def _build_prompt(self, ts_code: str, name: str, context: str) -> str:
        return f"""你是一位专业的A股投资分析师，请根据以下技术指标数据，对{name}({ts_code})进行中长期（1-3个月）走势分析。

{context}

请从以下维度进行简要分析（总共不超过150字）：
1. 趋势判断：当前处于上升/下降/震荡趋势中的哪个阶段？
2. 关键支撑/压力位：基于均线和价格位置
3. 中长期展望：看多/看空/中性，一句话理由

请用简洁的中文回答，格式如下：
趋势：[上升/下降/震荡]
支撑：[价格] 压力：[价格]
展望：[看多/看空/中性] - [一句话理由]"""

    def _build_event_prompt(self, ts_code: str, name: str, news_list: List[str]) -> str:
        news_text = "\n".join(f"{i+1}. {title}" for i, title in enumerate(news_list[:5]))
        return f"""你是A股风控分析师。以下是{name}({ts_code})的近期新闻：
{news_text}

请判断：
1. 是否存在重大利空风险（减持/立案/业绩暴雷/退市风险）？
2. 是否存在强政策催化？

回答格式（不超过80字）：
风险：[无/低/中/高] - [一句话说明]
催化：[无/弱/强] - [一句话说明]"""

    def _call_api(self, prompt: str) -> Optional[str]:
        api_key = self.api_key
        if not api_key:
            print("  [LLM] API Key为空，跳过分析")
            return None

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,
            'max_tokens': 200,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    result = resp.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    return content.strip()
                elif resp.status_code == 429:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"  [LLM] 限流，等待{wait}秒后重试 ({attempt+1}/{self.max_retries})")
                    time.sleep(wait)
                else:
                    print(f"  [LLM] HTTP {resp.status_code}: {resp.text[:200]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
            except requests.exceptions.Timeout:
                print(f"  [LLM] 超时，重试 ({attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
            except Exception as e:
                print(f"  [LLM] 异常: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

        print(f"  [LLM] 全部{self.max_retries}次重试失败")
        return None
