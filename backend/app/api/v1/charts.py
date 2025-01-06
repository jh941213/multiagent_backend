from fastapi import APIRouter
from ...routers.chart_router import router as chart_router

router = APIRouter()
router.include_router(chart_router, tags=["charts"])
