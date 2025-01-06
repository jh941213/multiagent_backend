from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
router = APIRouter()

# 전역 ThreadPoolExecutor 생성
executor = ThreadPoolExecutor(max_workers=4)
chat_service = ChatService()

@router.post("/chat", response_model=ChatResponse)
async def agent_chat_endpoint(request: ChatRequest):
    """에이전트 채팅 API 엔드포인트"""
    try:
        # ThreadPoolExecutor를 사용하여 별도의 스레드에서 처리
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: chat_service.sync_process_chat(
                message=request.message,
                chat_history=request.chat_history
            )
        )
        
        return ChatResponse(
            response=result["response"],
            chat_history=result["chat_history"]
        )
            
    except Exception as e:
        logger.error(f"에이전트 채팅 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"에이전트 채팅 처리 중 오류가 발생했습니다: {str(e)}"
        )
