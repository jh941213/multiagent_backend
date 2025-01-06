from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/charts")
async def get_charts():
    """차트 이미지 목록을 반환하는 엔드포인트"""
    try:
        # 차트 디렉토리 경로
        charts_dir = Path("/Users/kdb/Desktop/stockelper_v3.5/backend/charts")
        
        # 지원하는 이미지 확장자
        image_extensions = {'.png', '.jpg', '.jpeg'}
        
        # 차트 이미지 목록 생성
        chart_images = []
        for file in charts_dir.iterdir():
            if file.suffix.lower() in image_extensions:
                # 파일 이름에서 '_analysis' 제거하고 보기 좋게 표시
                name = file.stem.replace('_analysis', '')
                name = name.upper()  # 대문자로 변환
                
                chart_images.append({
                    "src": f"/charts/{file.name}",
                    "alt": f"Chart {file.stem}",
                    "name": name  # 처리된 이름 추가
                })
        
        return JSONResponse(content=chart_images)
        
    except Exception as e:
        logger.error(f"차트 이미지 로드 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="차트 이미지 로드 중 오류가 발생했습니다."
        ) 