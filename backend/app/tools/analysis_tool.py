import yfinance as yf
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Type, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from GoogleNews import GoogleNews
import json

class DateTimeEncoder(json.JSONEncoder):
    """datetime 객체를 JSON으로 직렬화하기 위한 인코더"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class MarketAnalysisInput(BaseModel):
    """Market Analysis Tool의 입력 모델"""
    symbol: Optional[str] = Field(None, description="분석할 주식 심볼 (예: AAPL)")
    company_name: Optional[str] = Field(default="", description="회사명 (예: 'Apple')")
    period_days: int = Field(default=180, description="데이터 조회 기간 (일)")
    rsi_period: int = Field(default=14, description="RSI 계산 기간")
    bb_period: int = Field(default=20, description="볼린저 밴드 계산 기간")
    ma_periods: List[int] = Field(default=[50, 200], description="이동평균선 계산 기간 리스트")

class MarketAnalysis:
    """시장 분석을 위한 통합 클래스"""
    
    INDICES = {
        '^GSPC': {'name': 'S&P 500', 'description': '미국 대형주 500개 기업을 포함하는 대표적인 주가지수'},
        '^DJI': {'name': '다우존스', 'description': '미국 30대 우량 기업을 대표하는 산업평균지수'},
        '^IXIC': {'name': '나스닥', 'description': '기술주 중심의 미국 전자주식시장 지수'}
    }

    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_market_news(self) -> list:
        """시장 뉴스를 비동기적으로 가져오는 메서드"""
        try:
            def fetch_news():
                googlenews = GoogleNews(lang='en', period='1d')
                googlenews.search("US stock market")
                news = googlenews.results()[:5]
                googlenews.clear()
                return news
            return await asyncio.to_thread(fetch_news)
        except Exception as e:
            print(f"뉴스 조회 실패: {str(e)}")
            return []

    async def fetch_company_data(self, symbol: str, company_name: str = "") -> Dict[str, Any]:
        """회사 기본 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return {"error": "주가 데이터를 가져올 수 없습니다."}
            
            current_price = float(hist['Close'].iloc[-1])
            open_price = float(hist['Open'].iloc[-1])
            
            return {
                "basic_info": {
                    "symbol": symbol,
                    "company_name": company_name or info.get('longName', 'N/A'),
                    "sector": info.get('sector', 'N/A'),
                    "industry": info.get('industry', 'N/A'),
                    "market_cap": info.get('marketCap', 'N/A'),
                    "website": info.get('website', 'N/A'),
                    "business_summary": info.get('longBusinessSummary', 'N/A')
                },
                "stock_data": {
                    "current_price": current_price,
                    "open": open_price,
                    "high": float(hist['High'].iloc[-1]),
                    "low": float(hist['Low'].iloc[-1]),
                    "close": float(hist['Close'].iloc[-1]),
                    "volume": int(hist['Volume'].iloc[-1]),
                    "day_change": float(((current_price - open_price) / open_price) * 100),
                    "timestamp": datetime.now().isoformat()
                },
                "financial_metrics": {
                    "pe_ratio": info.get('trailingPE', 'N/A'),
                    "forward_pe": info.get('forwardPE', 'N/A'),
                    "dividend_yield": info.get('dividendYield', 'N/A'),
                    "beta": info.get('beta', 'N/A'),
                    "eps": info.get('trailingEps', 'N/A'),
                    "peg_ratio": info.get('pegRatio', 'N/A'),
                    "profit_margins": info.get('profitMargins', 'N/A'),
                    "revenue_growth": info.get('revenueGrowth', 'N/A')
                }
            }
        except Exception as e:
            return {"error": f"회사 데이터 조회 중 오류 발생: {str(e)}"}

    def _convert_days_to_period(self, days: int) -> str:
        """일수를 yfinance 지원 기간 형식으로 변환"""
        if days <= 7: return "5d"
        elif days <= 30: return "1mo"
        elif days <= 90: return "3mo"
        elif days <= 180: return "6mo"
        elif days <= 365: return "1y"
        elif days <= 730: return "2y"
        elif days <= 1825: return "5y"
        elif days <= 3650: return "10y"
        else: return "max"

    async def fetch_price_data(self, symbol: str, period_days: int) -> Optional[pd.DataFrame]:
        """가격 데이터 조회"""
        try:
            ticker = yf.Ticker(symbol)
            period = self._convert_days_to_period(period_days)
            hist = ticker.history(period=period)
            return None if hist.empty else hist
        except Exception as e:
            print(f"가격 데이터 조회 실패: {str(e)}")
            return None

    async def get_technical_analysis(self, df: pd.DataFrame, 
                                   rsi_period: int = 14,
                                   bb_period: int = 20,
                                   ma_periods: List[int] = [50, 200]) -> Dict[str, Any]:
        """기술적 지표 계산"""
        try:
            # 지표 계산
            rsi_indicator = RSIIndicator(close=df['Close'], window=rsi_period)
            bb_indicator = BollingerBands(close=df['Close'], window=bb_period)
            macd_indicator = MACD(close=df['Close'])
            
            # 이동평균 계산
            moving_averages = {
                f'MA{period}': round(float(df['Close'].rolling(window=period).mean().iloc[-1]), 2)
                for period in ma_periods
            }

            # 거래량 분석
            volume_analysis = {
                "current_volume": int(df['Volume'].iloc[-1]),
                "avg_volume_5d": float(df['Volume'].rolling(window=5).mean().iloc[-1]),
                "avg_volume_20d": float(df['Volume'].rolling(window=20).mean().iloc[-1])
            }

            technical_data = {
                'rsi': {
                    'value': round(float(rsi_indicator.rsi().iloc[-1]), 2),
                    'period': rsi_period
                },
                'bollinger_bands': {
                    'upper': round(float(bb_indicator.bollinger_hband().iloc[-1]), 2),
                    'middle': round(float(bb_indicator.bollinger_mavg().iloc[-1]), 2),
                    'lower': round(float(bb_indicator.bollinger_lband().iloc[-1]), 2)
                },
                'macd': {
                    'macd': round(float(macd_indicator.macd().iloc[-1]), 2),
                    'signal': round(float(macd_indicator.macd_signal().iloc[-1]), 2),
                    'histogram': round(float(macd_indicator.macd_diff().iloc[-1]), 2)
                },
                'moving_averages': moving_averages,
                'volume_analysis': volume_analysis
            }

            technical_data['analysis'] = await self._analyze_indicators(technical_data)
            return technical_data

        except Exception as e:
            return {'error': f'기술적 지표 계산 중 오류 발생: {str(e)}'}

    async def _analyze_indicators(self, data: Dict[str, Any]) -> Dict[str, str]:
        """기술적 지표 분석"""
        analysis = {}
        
        # RSI 분석
        rsi = data['rsi']['value']
        analysis['rsi'] = "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립"
        
        # MACD 분석
        analysis['macd'] = "상승신호" if data['macd']['macd'] > data['macd']['signal'] else "하락신호"
        
        # 볼린저 밴드 분석
        bb = data['bollinger_bands']
        current_price = bb['middle']
        
        if current_price > bb['upper']:
            analysis['bollinger'] = "상단밴드 상향돌파"
        elif current_price < bb['lower']:
            analysis['bollinger'] = "하단밴드 하향돌파"
        else:
            analysis['bollinger'] = "밴드 내 움직임"
        
        return analysis

class MarketAnalysisTool(BaseTool):
    """시장 분석을 위한 통합 Tool"""
    
    name: str = "market_analysis"
    description: str = "시장 데이터, 기업 정보, 기술적 지표(RSI, 볼린저 밴드, MACD 등)를 분석합니다."
    args_schema: Type[BaseModel] = MarketAnalysisInput
    
    async def _run(
        self,
        symbol: Optional[str] = None,
        company_name: str = "",
        period_days: int = 180,
        rsi_period: int = 14,
        bb_period: int = 20,
        ma_periods: List[int] = [50, 200],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """동기 실행 메서드"""
        return await self._arun(
            symbol=symbol,
            company_name=company_name,
            period_days=period_days,
            rsi_period=rsi_period,
            bb_period=bb_period,
            ma_periods=ma_periods
        )

    async def _arun(
        self,
        symbol: Optional[str] = None,
        company_name: str = "",
        period_days: int = 180,
        rsi_period: int = 14,
        bb_period: int = 20,
        ma_periods: List[int] = [50, 200],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[str, Dict[str, Any]]:
        """비동기 실행 메서드"""
        try:
            analyzer = MarketAnalysis()
            result = await self._collect_market_data(
                analyzer, symbol, company_name, period_days, rsi_period, bb_period, ma_periods
            )
            return self._format_market_analysis(result)
            
        except Exception as e:
            error_result = {'error': f'시장 분석 중 오류 발생: {str(e)}'}
            return json.loads(json.dumps(error_result, cls=DateTimeEncoder))

    async def _collect_market_data(
        self,
        analyzer: MarketAnalysis,
        symbol: Optional[str],
        company_name: str,
        period_days: int,
        rsi_period: int,
        bb_period: int,
        ma_periods: List[int]
    ) -> Dict[str, Any]:
        """시장 데이터 수집 메서드"""
        market_overview = {
            'timestamp': datetime.now().isoformat(),
            'market_news': await analyzer.fetch_market_news(),
            'major_indices': {}
        }

        # 3대 지수 데이터 수집
        for idx_symbol, info in MarketAnalysis.INDICES.items():
            ticker = yf.Ticker(idx_symbol)
            hist = ticker.history(period="1d")
            
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                open_price = float(hist['Open'].iloc[-1])
                day_change = float(((current_price - open_price) / open_price) * 100)
                
                market_overview['major_indices'][info['name']] = {
                    'description': info['description'],
                    'current_price': round(current_price, 2),
                    'open': round(open_price, 2),
                    'high': round(float(hist['High'].iloc[-1]), 2),
                    'low': round(float(hist['Low'].iloc[-1]), 2),
                    'close': round(float(hist['Close'].iloc[-1]), 2),
                    'volume': int(hist['Volume'].iloc[-1]),
                    'day_change': round(day_change, 2)
                }

        result = {'market_overview': market_overview}

        if symbol:
            company_data = await analyzer.fetch_company_data(symbol, company_name)
            df = await analyzer.fetch_price_data(symbol, period_days)
            
            if df is not None and 'error' not in company_data:
                technical_analysis = await analyzer.get_technical_analysis(
                    df, rsi_period, bb_period, ma_periods
                )
                result.update({
                    'company_data': company_data,
                    'technical_analysis': technical_analysis
                })
            else:
                return {
                    'market_overview': market_overview,
                    'error': f'개별 종목 데이터 조회 실패: {symbol}'
                }

        return result
    
    def _format_market_analysis(self, data: Dict[str, Any]) -> str:
        """분석 결과를 마크다운 형식으로 포맷팅"""
        output = ["## 시장 개요"]
        
        # 주요 지수 섹션
        if "market_overview" in data and "major_indices" in data["market_overview"]:
            output.append("\n### 주요 지수")
            for index_name, index_data in data["market_overview"]["major_indices"].items():
                output.extend([
                    f"\n#### {index_name}",
                    f"- 현재가: ${index_data['current_price']:,.2f} ({index_data['day_change']:+.2f}%)",
                    f"- 거래량: {index_data['volume']:,}",
                    f"- 일중: 고가 ${index_data['high']:,.2f} / 저가 ${index_data['low']:,.2f}"
                ])

        # 뉴스 섹션
        if "market_overview" in data and "market_news" in data["market_overview"]:
            output.append("\n### 주요 뉴스")
            output.extend([f"- [{news['title']}]({news['link']}) - {news['media']}" 
                         for news in data["market_overview"]["market_news"]])

        # 개별 종목 분석
        if "company_data" in data:
            company = data["company_data"]["basic_info"]
            stock = data["company_data"]["stock_data"]
            metrics = data["company_data"]["financial_metrics"]
            tech = data["technical_analysis"]
            
            output.extend([
                f"\n## 개별 종목 분석: {company['symbol']}",
                "\n### 기본 정보",
                f"- 현재가: ${stock['current_price']:,.2f} ({stock['day_change']:+.2f}%)",
                f"- 거래량: {stock['volume']:,}",
                f"- 시가총액: ${company['market_cap']:,.0f}",
                f"- 섹터: {company['sector']}",
                f"- 산업: {company['industry']}",
                "\n### 재무 지표",
                f"- P/E 비율: {metrics['pe_ratio']}",
                f"- EPS: ${metrics['eps']}",
                f"- 수익성장률: {metrics['revenue_growth']*100 if metrics['revenue_growth'] != 'N/A' else 'N/A'}%",
                "\n### 기술적 분석",
                f"- RSI({tech['rsi']['period']}): {tech['rsi']['value']:.2f} ({tech['analysis']['rsi']})",
                f"- MACD: {tech['macd']['macd']:.2f} ({tech['analysis']['macd']})"
            ])
            
            bb = tech['bollinger_bands']
            output.append(f"- 볼린저 밴드: 상단 ${bb['upper']:.2f} / 중단 ${bb['middle']:.2f} / 하단 ${bb['lower']:.2f}")
            
            for ma_type, ma_value in tech['moving_averages'].items():
                output.append(f"- {ma_type}: ${ma_value:.2f}")

            vol = tech['volume_analysis']
            vol_vs_avg = (vol['current_volume'] / vol['avg_volume_20d'] - 1) * 100
            output.append(f"- 거래량: {vol['current_volume']:,} (20일 평균 대비: {vol_vs_avg:+.1f}%)")

        return "\n".join(output)