from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
import asyncio
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 환경 변수 로드
load_dotenv()

class VectorDBSearchInput(BaseModel):
    """벡터 DB 검색 도구의 입력 모델"""
    query: str = Field(..., description="검색할 질문")
    category: Optional[str] = Field(
        default="",
        description="검색 카테고리 (계좌개설, 수수료/증거금, 매매안내, 인증센터, 거래 리스크 관리)"
    )

class VectorDBSearcher:
    """벡터 데이터베이스 검색 클래스"""
    async def search(self, query: str, category: str = "") -> Dict[str, Any]:
        """벡터 데이터베이스 검색 실행"""
        try:
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
            vector_db_path = os.path.join("/Users/kdb/Desktop/stockelper_v3.5/backend/app/vector_db", category) if category else "/Users/kdb/Desktop/stockelper_v3/vector_db"
            vectorstore = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
            documents = vectorstore.similarity_search(query, k=3)
            
            return {
                "search_info": {
                    "query": query,
                    "category": category or "전체",
                },
                "results": documents
            }
        except Exception as e:
            return {"error": f"검색 중 오류 발생: {str(e)}"}

class VectorDBSearchTool(BaseTool):
    """벡터 데이터베이스 검색을 위한 Tool"""
    
    name: str = "vector_db_search"
    description: str = """벡터 데이터베이스에서 관련 정보를 검색합니다.
    카테고리: '계좌개설', '수수료/증거금', '매매안내', '인증센터', '거래 리스크 관리'"""
    args_schema: Type[BaseModel] = VectorDBSearchInput
    
    llm: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    ))
    
    category_prompt: ChatPromptTemplate = Field(
        default_factory=lambda: ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자의 질문을 분석하여 적절한 카테고리를 선택하는 도우미입니다.
            다음 카테고리 중 하나만 선택하세요:
            - 계좌개설: 계좌 개설, 신규 계좌 등과 관련된 문의
            - 수수료/증거금: 거래 수수료, 증거금, 자금 관련 문의
            - 매매안내: 주식 거래 방법, 매수/매도 방법 등 거래 관련 문의
            - 인증센터: 보안, 인증, 로그인 관련 문의
            - 거래 리스크 관리: 투자 위험, 리스크 관리 관련 문의
            
            입력된 질문을 분석하여 가장 적절한 카테고리만 응답하세요."""),
            ("human", "{query}"),
        ])
    )

    async def _get_category(self, query: str) -> str:
        """LLM을 사용하여 쿼리에서 카테고리 추출"""
        try:
            result = await self.llm.ainvoke(
                self.category_prompt.format_messages(query=query)
            )
            # 응답에서 카테고리 추출
            category = result.content.strip()
            # 유효한 카테고리인지 확인
            valid_categories = ['계좌개설', '수수료/증거금', '매매안내', '인증센터', '거래 리스크 관리']
            return category if category in valid_categories else ""
        except Exception as e:
            print(f"카테고리 추출 중 오류 발생: {str(e)}")
            return ""

    async def _arun(
        self,
        query: str,
        category: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """비동기 실행 메서드"""
        try:
            # 카테고리가 지정되지 않은 경우 LLM을 통해 카테고리 추출
            if not category:
                category = await self._get_category(query)
            
            searcher = VectorDBSearcher()
            result = await searcher.search(
                query=query,
                category=category
            )
            if "results" in result:
                result["results"] = [doc.page_content for doc in result["results"]]
            return result
        except Exception as e:
            return {'error': f'검색 중 오류 발생: {str(e)}'}

    def _run(
        self,
        query: str,
        category: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """동기 실행 메서드"""
        try:
            # asyncio.run() 대신 기존 이벤트 루프 사용
            return asyncio.get_event_loop().run_until_complete(
                self._arun(query, category, run_manager)
            )
        except Exception as e:
            return {'error': f'검색 중 오류 발생: {str(e)}'}