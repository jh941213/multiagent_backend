from pydantic import BaseModel
from typing import List, Dict

class ChartData(BaseModel):
    time: str
    value: float

class MarketData(BaseModel):
    name: str
    symbol: str
    current: float
    change: float
    changePercent: float
    flag: str
    color: str
    data: List[ChartData]

class MarketResponse(BaseModel):
    indices: List[MarketData] 