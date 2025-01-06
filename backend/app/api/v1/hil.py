from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from ...agents.hil_agent import (
    research_graph,
    KDBanalyst,
    GenerateAnalystsState,
    ResearchGraphState
)

router = APIRouter(
    prefix="/hil",
    tags=["hil"]
)

class ResearchRequest(BaseModel):
    topic: str
    max_analysts: int = 3
    human_analyst_feedback: Optional[str] = None

class ResearchResponse(BaseModel):
    final_report: str
    analysts: List[KDBanalyst]

@router.post("/research", response_model=ResearchResponse)
async def create_research_report(request: ResearchRequest):
    try:
        # 초기 상태 설정
        initial_state = ResearchGraphState(
            topic=request.topic,
            max_analysts=request.max_analysts,
            human_analyst_feedback=request.human_analyst_feedback,
            analysts=[],
            sections=[],
            introduction="",
            content="",
            conclusion="",
            final_report=""
        )

        # 리서치 그래프 실행
        result = research_graph.invoke(initial_state)

        return {
            "final_report": result["final_report"],
            "analysts": result["analysts"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 