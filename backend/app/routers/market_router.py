from fastapi import APIRouter, HTTPException
from ..services.market_service import MarketService
from ..schemas.market import MarketResponse
from ..schemas.response import StandardResponse

router = APIRouter()

@router.get("/market-indices", response_model=StandardResponse)
async def get_indices():
    """시장 지수 데이터 조회"""
    try:
        market_data = await MarketService.get_market_data()
        return StandardResponse(
            status="success",
            message="Market data retrieved successfully",
            data={"indices": market_data}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 