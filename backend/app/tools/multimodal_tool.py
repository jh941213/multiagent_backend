import asyncio
import nest_asyncio
import json
import boto3
from PIL import Image
import os
import base64
from io import BytesIO
from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from .ticker_tool import ComprehensiveStockTool
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class StockQueryInput(BaseModel):
    """주식 분석 쿼리 입력 모델"""
    symbol: str = Field(..., description="주식 심볼 (예: AAPL)")

class MultimodalTool(BaseTool):
    """멀티모달 주식 분석 도구"""
    
    name: str = "multimodal_stock_analysis"
    description: str = "주식 차트와 데이터를 종합적으로 분석하여 투자 인사이트를 제공합니다."
    args_schema: Type[BaseModel] = StockQueryInput
    
    # Pydantic 필드로 클래스 변수 정의
    bedrock_runtime: Any = Field(default=None, exclude=True)
    base_tool: Any = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        try:
            nest_asyncio.apply()
        except:
            pass
        
        # 클래스 변수 초기화
        object.__setattr__(self, 'bedrock_runtime', boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        ))
        object.__setattr__(self, 'base_tool', ComprehensiveStockTool())

    async def analyze_chart_with_bedrock(self, chart_path: str) -> str:
        """차트 이미지 분석"""
        try:
            if not os.path.exists(chart_path):
                return "차트 이미지를 찾을 수 없습니다."

            with Image.open(chart_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_str
                                }
                            },
                            {
                                "type": "text",
                                "text": "이 주식 차트를 분석하여 주요 기술적 패턴과 트렌드를 설명해주세요."
                            }
                        ]
                    }
                ]
            }

            response = self.bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except Exception as e:
            return f"차트 분석 중 오류 발생: {str(e)}"

    async def get_final_analysis(self, stock_data: Dict[str, Any], chart_analysis: str) -> str:
        """종합 분석 실행"""
        try:
            analysis_prompt = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
                        다음 주식 데이터와 차트 분석을 바탕으로 종합적인 투자 인사이트를 제공해주세요:

                        기본 정보:
                        {stock_data['basic_info']}

                        시장 분석:
                        {stock_data['market_analysis']}

                        차트 분석 결과:
                        {chart_analysis}

                        위 정보를 종합적으로 분석하여 현재 시장 상황과 투자 전략에 대한 인사이트를 제공해주세요.
                        """
                    }
                ]
            }

            response = self.bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(analysis_prompt)
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except Exception as e:
            return f"종합 분석 중 오류 발생: {str(e)}"

    def _run(
        self,
        symbol: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """동기 실행 메서드"""
        return asyncio.run(self._arun(symbol=symbol))

    async def _arun(
        self,
        symbol: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """비동기 실행 메서드"""
        try:
            # 기본 주식 분석 실행
            stock_data = await self.base_tool._arun(
                symbol=symbol,
                period_days=180,  # 기본값 사용
                rsi_period=14,
                bb_period=20,
                ma_periods=[50, 200],
                company_name=""
            )

            # 차트 분석 실행
            if 'market_analysis' in stock_data and 'charts' in stock_data['market_analysis']:
                chart_path = stock_data['market_analysis']['charts']['path']
                chart_analysis = await self.analyze_chart_with_bedrock(chart_path)
            else:
                chart_analysis = "차트 분석을 수행할 수 없습니다."

            # 종합 분석 실행
            final_analysis = await self.get_final_analysis(stock_data, chart_analysis)

            # 결과 통합
            return {
                **stock_data,
                "ai_analysis": {
                    "chart_analysis": chart_analysis,
                    "final_analysis": final_analysis
                }
            }
        except Exception as e:
            return {
                "error": f"분석 중 오류 발생: {str(e)}"
            }

# 사용 예시:
"""
# 도구 초기화
tool = MultimodalTool()

# 분석 실행
result = tool._run("AAPL")

print(result['ai_analysis']['final_analysis'])
"""