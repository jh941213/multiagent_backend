import asyncio
import os
from dotenv import load_dotenv
from pprint import pprint
from langchain_core.tools import BaseTool
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from datetime import datetime
class YouTubeSearchInput(BaseModel):
    """YouTube 검색 입력 모델"""
    query: str = Field(..., description="검색할 키워드나 주제")
    max_results: int = Field(default=1, description="검색할 최대 동영상 수")

class YouTubeSearchTool(BaseTool):
    """YouTube 검색 및 자막 추출 Tool"""
    
    name: str = "youtube_search"
    description: str = "주식/투자 관련 YouTube 동영상을 검색하고 자막을 추출합니다."
    _api_key: str = PrivateAttr()
    _llm: ChatOpenAI = PrivateAttr()
    args_schema: Type[BaseModel] = YouTubeSearchInput
    
    def __init__(self, **kwargs):
        """Pydantic과 호환되게 API 키와 LLM 초기화"""
        super().__init__(**kwargs)
        load_dotenv()  # .env 파일 로드
        self._api_key = os.getenv('YOUTUBE_API_KEY')
        if not self._api_key:
            raise ValueError("YouTube API Key not found in environment variables")
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def _optimize_search_query(self, query: str) -> str:
        """LLM을 사용하여 검색 쿼리 최적화"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""
        다음 검색어를 YouTube에서 주식/투자 관련 동영상을 찾기 위한 최적화된 검색어로 변환해주세요.
        원본 검색어: {query} + 현재 시간: {current_time}
        
        규칙:
        1. 주식/투자와 관련된 키워드를 한국어로 포함
        2. 한국어로만 검색어 작성
        3. 최근 정보를 찾기 위한 시간 관련 키워드 포함
        4. 간단하고 명확하게 작성
        5. 현재시간을 참고하여 최근 정보를 찾기 위한 키워드 포함
        
        한국어 검색어만 반환하세요.
        """
        
        optimized_query = await self._llm.ainvoke(prompt)
        print(f"최적화된 검색어: {optimized_query}")
        return optimized_query

    async def _get_transcript(self, video_id: str) -> str:
        """YouTube 동영상의 자막을 추출"""
        try:
            transcript = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
            )
            # 자막 텍스트를 1000자로 제한
            full_text = "\n".join([item['text'] for item in transcript])
            return full_text[:1000] + ("..." if len(full_text) > 1000 else "")
        except Exception as e:
            return f"자막 없음: {str(e)}"

    def _extract_video_id(self, url: str) -> str:
        """URL에서 video ID 추출"""
        if 'youtu.be' in url:
            return url.split('/')[-1]
        parsed_url = urlparse(url)
        return parse_qs(parsed_url.query)['v'][0]

    def _run(
        self,
        query: str,
        max_results: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """동기 메서드는 비동기 메서드를 실행"""
        return asyncio.run(self._arun(query, max_results, run_manager))

    async def _arun(
        self,
        query: str,
        max_results: int = 1,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """메인 비동기 실행 메서드"""
        try:
            # 검색어 최적화
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            query_with_time = f"{query} {current_time}"
            optimized_query = await self._optimize_search_query(query_with_time)
            optimized_query = optimized_query.content.strip('"')  # LLM 반환값 처리
            
            # YouTube API 클라이언트 생성과 검색 실행
            youtube_search = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: build('youtube', 'v3', developerKey=self._api_key).search().list(
                    q=optimized_query,
                    part='id,snippet',
                    maxResults=max_results,
                    type='video',
                    order='relevance',
                    relevanceLanguage='ko'
                ).execute()
            )
            
            # 검색 결과를 순차적으로 처리
            results = []
            all_transcripts = []  # 모든 자막을 저장할 리스트
            
            for item in youtube_search.get('items', []):
                video_id = item['id']['videoId']
                video_info = {
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'description': item['snippet']['description'],
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                }
                
                # 자막 가져오기 (순차적으로 실행)
                transcript = await self._get_transcript(video_id)
                video_info['transcript'] = transcript
                results.append(video_info)
                
                # 자막을 컨텍스트용 리스트에 추가
                all_transcripts.append(
                    f"\n=== 동영상 #{len(results)} ===\n"
                    f"제목: {video_info['title']}\n"
                    f"채널: {video_info['channel']}\n"
                    f"URL: {video_info['url']}\n"
                    f"자막 내용:\n{transcript}\n"
                    f"=== 동영상 #{len(results)} 끝 ===\n"
                )
            
            return {
                'videos': results,
                'context': "\n".join(all_transcripts)  # 모든 자막을 하나의 문자열로 결합
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'query': query
            }