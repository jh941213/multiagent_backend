import yfinance as yf
from datetime import datetime
from typing import Dict, Any, Optional, Type, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
import asyncio

class StockPriceInput(BaseModel):
    """Stock Price Tool의 입력 모델"""
    symbol: str = Field(..., description="주식 심볼 (예: AAPL)")
    company_name: Optional[str] = Field(default="", description="회사명 (예: 'Apple')")

class StockPriceAnalyzer:
    """주식 가격 정보 분석 클래스"""
    async def get_stock_price(self, symbol: str, company_name: str = "") -> Dict[str, Any]:
        """주식 기본 가격 정보 조회"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return {"error": "주가 데이터를 가져올 수 없습니다."}
            
            current_price = float(hist['Close'].iloc[-1])
            open_price = float(hist['Open'].iloc[-1])
            
            return {
                "stock_info": {
                    "symbol": symbol,
                    "company_name": company_name or info.get('longName', 'N/A'),
                },
                "price_data": {
                    "current_price": round(current_price, 2),
                    "open": round(open_price, 2),
                    "high": round(float(hist['High'].iloc[-1]), 2), 
                    "low": round(float(hist['Low'].iloc[-1]), 2),
                    "close": round(float(hist['Close'].iloc[-1]), 2),
                    "volume": int(hist['Volume'].iloc[-1]),
                    "day_change": round(float(((current_price - open_price) / open_price) * 100), 2),
                    "timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {"error": f"주가 데이터 조회 중 오류 발생: {str(e)}"}

class StockPriceTool(BaseTool):
    """주식 가격 조회를 위한 Tool"""
    
    name: str = "stock_price"
    description: str = "기본적인 주가 정보를 조회합니다. 주가 분석에는 활용하지 않습니다. 주가 분석에는 MarketAnalysisTool을 사용합니다."
    args_schema: Type[BaseModel] = StockPriceInput
    
    def _run(
        self,
        symbol: str,
        company_name: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """동기 실행 메서드"""
        return asyncio.run(self._arun(symbol=symbol, company_name=company_name))

    async def _arun(
        self,
        symbol: str,
        company_name: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """비동기 실행 메서드"""
        try:
            analyzer = StockPriceAnalyzer()
            result = await analyzer.get_stock_price(
                symbol=symbol,
                company_name=company_name
            )
            return result
        except Exception as e:
            return {'error': f'가격 데이터 조회 중 오류 발생: {str(e)}'}