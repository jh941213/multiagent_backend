from app.api.v1 import chat, health, market, charts, news, youtube, hil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
import asyncio
from app.agents.super_agent import postgres_manager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_event_loop():
    """기본 이벤트 루프 설정"""
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("기본 이벤트 루프가 성공적으로 설정되었습니다.")
    except Exception as e:
        logger.error(f"이벤트 루프 설정 중 오류 발생: {e}")
        raise

def setup_charts_directory():
    """차트 디렉토리 설정"""
    current_dir = Path(__file__).parent.parent.parent
    charts_dir = current_dir / "backend" / "charts"
    
    try:
        charts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"차트 디렉토리 생성 완료: {charts_dir}")
    except Exception as e:
        logger.warning(f"차트 디렉토리 생성 실패: {e}")
        charts_dir = Path(tempfile.gettempdir()) / "stockelper_charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"임시 차트 디렉토리 사용: {charts_dir}")
    
    return charts_dir

def create_app():
    """FastAPI 애플리케이션 생성"""
    app = FastAPI(
        title="Stockelper AI",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS 미들웨어 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://frontend:3000", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 차트 디렉토리 설정 및 마운트
    charts_dir = setup_charts_directory()
    app.mount("/charts", StaticFiles(directory=str(charts_dir)), name="charts")
    
    # API 라우터 등록
    routers = [
        (chat.router, "/v1"),
        (health.router, "/v1"),
        (market.router, "/v1"),
        (charts.router, "/v1"),
        (news.router, "/v1"),
        (youtube.router, "/v1"),
        (hil.router, "/v1"),
    ]
    
    for router, prefix in routers:
        app.include_router(router, prefix=prefix)
    
    return app

# 이벤트 루프 설정
setup_event_loop()

# FastAPI 앱 생성
app = create_app()

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트 핸들러"""
    try:
        await postgres_manager.get_pool()
        logger.info("데이터베이스 연결 초기화 완료")
    except Exception as e:
        logger.error(f"데이터베이스 연결 초기화 실패: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트 핸들러"""
    await postgres_manager.close_pool()
    logger.info("데이터베이스 연결 종료")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8010, 
        reload=True,
        loop="asyncio"
    )
