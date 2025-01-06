from pydantic import BaseModel
from typing import Any, Optional

class StandardResponse(BaseModel):
    """표준 응답 스키마"""
    status: str
    message: str
    data: Optional[Any] = None 