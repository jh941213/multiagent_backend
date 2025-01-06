from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional
from pydantic import BaseModel

# API 라우터 설정 (prefix 제거)
router = APIRouter(
    tags=["news"]
)

# PostgreSQL 연결 설정
DB_CONFIG = {
    "dbname": "stockhelper",
    "user": "kdb",
    "password": "1234",
    "host": "localhost"
}

class NewsResponse(BaseModel):
    id: int
    title: str
    source: str
    time: datetime
    company: str
    stockCode: str
    originUrl: str
    summary: str

@router.get("/news", response_model=List[NewsResponse])
async def get_news(
    stock_code: Optional[str] = Query(None, description="종목 코드"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=100, description="페이지당 뉴스 수")
):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        offset = (page - 1) * limit
        
        query = """
        SELECT 
            id::integer,
            title,
            press as source,
            timestamp as time,
            company,
            stock_code as "stockCode",
            origin_url as "originUrl",
            summary
        FROM news 
        """
        
        params = []
        if stock_code:
            query += " WHERE stock_code = %s"
            params.append(stock_code)
        
        query += """
        ORDER BY timestamp DESC 
        LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])
        
        cur.execute(query, params)
        news_items = cur.fetchall()
        formatted_news = [dict(row) for row in news_items]
        
        return formatted_news
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()
        except:
            pass