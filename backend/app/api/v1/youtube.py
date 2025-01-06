from fastapi import APIRouter, HTTPException, Query
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

router = APIRouter(prefix="/youtube")
logger.info(f"YouTube 라우터 초기화됨: prefix={router.prefix}")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    logger.error("YouTube API key not found in environment variables")
    raise ValueError("YouTube API key is not set in environment variables")

def youtube_search(query: str, max_results: int = 3):
    """YouTube API를 사용하여 동영상을 검색합니다."""
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    
    try:
        request = youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=max_results,
            type="video",
            regionCode="KR",
            relevanceLanguage="ko"
        )
        
        response = request.execute()
        
        results = []
        for item in response["items"]:
            video_id = item["id"]["videoId"]
            results.append({
                "id": video_id,
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "description": item["snippet"]["description"],
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                "url": f"https://www.youtube.com/watch?v={video_id}"
            })
        return results
    except Exception as e:
        logger.error(f"YouTube 검색 중 오류: {str(e)}")
        raise e

@router.get("/search")
async def search_videos(q: str = Query(..., description="검색어")):
    logger.info(f"검색 요청 받음: query={q}")
    try:
        results = youtube_search(q)
        logger.info(f"검색 결과: {len(results)} 개의 비디오")
        return {"items": results}
    except HttpError as e:
        logger.error(f"YouTube API HTTP 오류: {str(e)}")
        raise HTTPException(status_code=e.resp.status, detail=str(e))
    except Exception as e:
        logger.error(f"일반 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))