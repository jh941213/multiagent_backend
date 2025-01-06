from fastapi import APIRouter
from ...routers import health_router

router = APIRouter()
router.include_router(health_router.router, tags=["health"])
