import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional, Type, List, Union
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
import asyncio
import os
import numpy as np

class MarketAnalysisInput(BaseModel):
    """마켓 분석 도구의 입력 파라미터를 정의하는 클래스"""
    
    symbol: str = Field(...,description="분석할 주식 심볼 (예: 'AAPL', 'MSFT')",)
    
    period_days: int = Field(
        default=180,
        description="분석할 기간 (일)",
        ge=1,
        le=3650
    )
    
    rsi_period: int = Field(
        default=14,
        description="RSI 계산 기간",
        ge=1,
        le=50
    )
    
    bb_period: int = Field(
        default=20,
        description="볼린저 밴드 계산 기간",
        ge=1,
        le=50
    )
    
    ma_periods: List[int] = Field(
        default=[50, 200],
        description="이동평균선 계산 기간 리스트"
    )
    
    company_name: Optional[str] = Field(
        default="",
        description="회사명 (선택사항)"
    )
    
    stoch_k_period: int = Field(
        default=14,
        description="스토캐스틱 K 기간",
        ge=1,
        le=50
    )
    stoch_d_period: int = Field(
        default=3,
        description="스토캐스틱 D 기간",
        ge=1,
        le=20
    )
    
    obv_enabled: bool = Field(
        default=True,
        description="OBV(On Balance Volume) 표시 여부"
    )

    @field_validator('symbol')
    def validate_symbol(cls, v):
        if not v.isalnum():
            raise ValueError("심볼은 알파벳과 숫자만 포함할 수 있습니다")
        if not any(c.isalpha() for c in v):
            raise ValueError("심볌은 최소 하나의 알파벳을 포함해야 합니다")
        return v.upper()

class ComprehensiveStockAnalyzer:
    """종합 주식 분석 클래스"""
    
    def __init__(self):
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    async def get_stock_data(self, symbol: str, period_days: int) -> pd.DataFrame:
        """주식 데이터 조회"""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=period_days)
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        return df, stock.info

    async def get_basic_price_info(self, symbol: str, company_name: str = "") -> Dict[str, Any]:
        """기본 주가 정보 조회"""
        try:
            df, info = await self.get_stock_data(symbol, 1)
            
            if df.empty:
                return {"error": "주가 데이터를 가져올 수 없습니다."}
            
            current_price = float(df['Close'].iloc[-1])
            open_price = float(df['Open'].iloc[-1])
            
            return {
                "stock_info": {
                    "symbol": symbol,
                    "company_name": company_name or info.get('longName', 'N/A'),
                },
                "price_data": {
                    "current_price": round(current_price, 2),
                    "open": round(open_price, 2),
                    "high": round(float(df['High'].iloc[-1]), 2),
                    "low": round(float(df['Low'].iloc[-1]), 2),
                    "close": round(float(df['Close'].iloc[-1]), 2),
                    "volume": int(df['Volume'].iloc[-1]),
                    "day_change": round(float(((current_price - open_price) / open_price) * 100), 2),
                    "timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {"error": f"주가 데이터 조회 중 오류 발생: {str(e)}"}

    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """RSI 계산"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int, d_period: int) -> tuple:
        """스토캐스틱 계산"""
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """OBV(On Balance Volume) 계산"""
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        return obv

    async def analyze_market(self, input_data: MarketAnalysisInput) -> Dict[str, Any]:
        """종합 시장 분석 수행"""
        try:
            df, info = await self.get_stock_data(input_data.symbol, input_data.period_days)
            
            # 기술적 지표 계산
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            bb_middle = df['Close'].rolling(window=input_data.bb_period).mean()
            bb_std = df['Close'].rolling(window=input_data.bb_period).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            rsi = self._calculate_rsi(df, input_data.rsi_period)

            # 차트 생성
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 20),
                                                         gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
            
            # 메인 차트
            ax1.plot(df.index, df['Close'], label='Close', color='blue', alpha=0.7)
            ax1.plot(df.index, bb_upper, 'r--', alpha=0.3, label='BB Upper')
            ax1.plot(df.index, bb_middle, 'g--', alpha=0.3, label='BB Middle')
            ax1.plot(df.index, bb_lower, 'r--', alpha=0.3, label='BB Lower')
            
            for period in input_data.ma_periods:
                ma = df['Close'].rolling(window=period).mean()
                ax1.plot(df.index, ma, label=f'MA{period}', alpha=0.7)

            # 거래량 차트
            ax2.bar(df.index, df['Volume'], label='Volume', color='darkgray', alpha=0.7)
            
            # MACD 차트
            ax3.plot(df.index, macd, label='MACD', color='blue')
            ax3.plot(df.index, signal, label='Signal', color='orange')
            ax3.bar(df.index, macd - signal, label='MACD Histogram', color='gray', alpha=0.3)
            
            # 스토캐스틱 차트 추가
            k, d = self._calculate_stochastic(df, input_data.stoch_k_period, input_data.stoch_d_period)
            ax4.plot(df.index, k, label='%K', color='blue')
            ax4.plot(df.index, d, label='%D', color='red')
            ax4.axhline(y=80, color='r', linestyle='--', alpha=0.3)
            ax4.axhline(y=20, color='g', linestyle='--', alpha=0.3)
            ax4.set_ylabel('Stochastic')
            ax4.legend(loc='upper left')
            
            # OBV 차트 추가 (선택적)
            if input_data.obv_enabled:
                obv = self._calculate_obv(df)
                ax5.plot(df.index, obv, label='OBV', color='darkgreen')
                ax5.set_ylabel('OBV')
                ax5.legend(loc='upper left')
            
            # RSI 차트를 ax5로 이동
            ax5.plot(df.index, rsi, label='RSI', color='purple')
            ax5.axhline(y=70, color='r', linestyle='--', alpha=0.3)
            ax5.axhline(y=30, color='g', linestyle='--', alpha=0.3)
            ax5.set_ylabel('RSI')
            ax5.set_ylim([0, 100])
            
            # 차트 스타일링
            ax1.set_title(f'{input_data.symbol} Comprehensive Analysis Chart')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            ax3.legend(loc='upper left')
            ax4.legend(loc='upper left')
            ax5.legend(loc='upper left')
            
            ax5.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax2.set_ylabel('Volume')
            ax3.set_ylabel('MACD')
            ax4.set_ylabel('Stochastic')
            ax5.set_ylabel('RSI')
            ax5.set_ylim([0, 100])

            plt.tight_layout()
            
            # 차트 저장
            save_path = 'charts'
            os.makedirs(save_path, exist_ok=True)
            chart_path = os.path.join(save_path, f"{input_data.symbol}_analysis.png")
            plt.savefig(chart_path)
            plt.close()

            return {
                "symbol": input_data.symbol,
                "company_name": input_data.company_name or info.get('longName', 'N/A'),
                "charts": {
                    "path": chart_path
                },
                "technical_analysis": {
                    "rsi": {
                        "value": round(rsi[-1], 2),
                        "period": input_data.rsi_period
                    },
                    "macd": {
                        "value": round(macd[-1], 2),
                        "signal": round(signal[-1], 2),
                        "histogram": round((macd[-1] - signal[-1]), 2)
                    },
                    "bollinger_bands": {
                        "upper": round(bb_upper[-1], 2),
                        "middle": round(bb_middle[-1], 2),
                        "lower": round(bb_lower[-1], 2)
                    }
                }
            }
        except Exception as e:
            plt.close()
            return {
                "symbol": input_data.symbol,
                "error": f"분석 중 오류 발생: {str(e)}"
            }

class ComprehensiveStockTool(BaseTool):
    """종합 주식 분석 도구"""
    
    name: str = "comprehensive_stock_analysis"
    description: str = """
    주식의 기본 정보와 기술적 분석을 제공하는 종합 분석 도구입니다.
    기본 가격 정보 조회와 기술적 분석(RSI, MACD, 볼린저 밴드 등)을 수행합니다.
    """
    args_schema: Type[BaseModel] = MarketAnalysisInput

    def _run(
        self,
        symbol: str,
        period_days: int = 180,
        rsi_period: int = 14,
        bb_period: int = 20,
        ma_periods: List[int] = None,
        company_name: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """동기 실행 메서드"""
        return asyncio.run(self._arun(
            symbol=symbol,
            period_days=period_days,
            rsi_period=rsi_period,
            bb_period=bb_period,
            ma_periods=ma_periods,
            company_name=company_name
        ))

    async def _arun(
        self,
        symbol: str,
        period_days: int = 180,
        rsi_period: int = 14,
        bb_period: int = 20,
        ma_periods: List[int] = None,
        company_name: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """비동기 실행 메서드"""
        try:
            analyzer = ComprehensiveStockAnalyzer()
            
            # 기본 가격 정보 조회
            price_info = await analyzer.get_basic_price_info(
                symbol=symbol,
                company_name=company_name
            )
            
            # 기술적 분석 수행
            analysis_input = MarketAnalysisInput(
                symbol=symbol,
                period_days=period_days,
                rsi_period=rsi_period,
                bb_period=bb_period,
                ma_periods=ma_periods or [50, 200],
                company_name=company_name
            )
            market_analysis = await analyzer.analyze_market(analysis_input)
            
            # 결과 통합
            return {
                "basic_info": price_info,
                "market_analysis": market_analysis
            }
        except Exception as e:
            return {
                "error": f"분석 중 오류 발생: {str(e)}"
            }