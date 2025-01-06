from fastapi import APIRouter
from ..schemas.response import StandardResponse

router = APIRouter()

@router.get("/health", response_model=StandardResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    return StandardResponse(
        status="success",
        message="Service is healthy"
    ) 