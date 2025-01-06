from fastapi import APIRouter, HTTPException
from ...schemas.chat import ChatRequest, ChatResponse
from ...services.chat_service import ChatService
from ...routers.chat_router import router as chat_router

router = APIRouter()
router.include_router(chat_router, tags=["chat"])
