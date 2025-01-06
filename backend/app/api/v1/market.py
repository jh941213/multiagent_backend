from fastapi import APIRouter
from ...routers import market_router

router = APIRouter()
router.include_router(market_router.router, tags=["market"]) 