from typing import Dict, Any
from ..agents.finance_agent import create_finance_agent
from ..agents.market_agent import create_market_agent
from langgraph.checkpoint.memory import MemorySaver

class AgentService:
    """에이전트 서비스"""
    def __init__(self):
        self.memory = MemorySaver()
        self.finance_agent = create_finance_agent(self.memory)
        self.market_agent = create_market_agent(self.memory)

    async def get_finance_analysis(self, query: str) -> Dict[str, Any]:
        """금융 분석 수행"""
        return await self.finance_agent.ainvoke({"messages": [{"role": "user", "content": query}]})

    async def get_market_analysis(self, query: str) -> Dict[str, Any]:
        """시장 분석 수행"""
        return await self.market_agent.ainvoke({"messages": [{"role": "user", "content": query}]}) 