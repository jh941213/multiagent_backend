from typing import Annotated, Sequence, TypedDict, Dict, Any, List
import json
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from tools.basic_tool import StockPriceTool
from tools.analysis_tool import MarketAnalysisTool
from tools.youtube_tool import YouTubeSearchTool
from tools.multimodal_tool import MultimodalTool
import matplotlib
class MarketAgentState(TypedDict):
    """시장 분석 에이전트의 상태"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    chat_history: List[Dict[str, str]]

def create_market_agent(memory=None):
    """시장 분석 에이전트 생성"""
    # 도구 초기화
    stock_price_tool = StockPriceTool()
    market_analysis_tool = MarketAnalysisTool()
    youtube_search_tool = YouTubeSearchTool()
    multimodal_tool = MultimodalTool()
    tools = [stock_price_tool, market_analysis_tool, youtube_search_tool, multimodal_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    
    # LLM 초기화
    model = ChatOpenAI(model="gpt-4o-mini")
    model = model.bind_tools(tools)
    
    # 도구 노드
    def tool_node(state: MarketAgentState) -> Dict:
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            # matplotlib 사용 전에 백엔드를 'Agg'로 설정
        
            matplotlib.use('Agg')
            
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            if asyncio.iscoroutine(tool_result):
                tool_result = asyncio.run(tool_result)
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs, "chat_history": state["chat_history"]}
    
    # LLM 노드
    def call_model(
        state: MarketAgentState,
        config: RunnableConfig,
    ) -> Dict:
        system_prompt = SystemMessage(content="""
        주식 시장 분석 Stockelper AI 어시스턴트입니다.
        다음 도구들을 사용하여 분석을 제공합니다:

        1. 주가 정보 조회 (stock_price) -> 기본적인 주가 정보 조회인 경우
        - 실시간 주가, 거래량, 시가/고가/저가 등 기본 시세 정보 제공
        - 주가 변동 추이와 등락률 분석
        ex) 오늘 애플 주가 얼마야?, 삼성전자 거래량 얼마야?
        
        2. 종합적인 시장 분석 (market_analysis)  -> 종합적인 시장 주가 분석 비교적 정밀한 경우
        - 기술적 지표(RSI, MACD, 볼린저밴드 등) 분석
        - 시장 동향과 섹터별 분석 리포트
        - 투자심리 지표와 수급 동향 분석
        -> 오늘 엔비디아를 매수 해도 되는지 궁금합니다 , 나스닥 주가흐름 방향이 어떤지 궁금합니다
        
        3. 투자 정보 검색 (youtube_search) -> 유투브 검색 도구
        - 종목 관련 전문가 분석 영상 검색
        - 시장 전망 및 투자 전략 관련 콘텐츠
        - 실시간 투자 뉴스와 리뷰 영상
        -> 테슬라 최신 영상 조회 해줘
                                      
        4. 멀티 모달 기반 차트 분석 (multimodal) -> 멀티모달 또는 차트기반 분석에 이용
        - AI 기반 차트 패턴 분석
        - 기술적/기본적 분석 결합한 종합 리포트
        - 투자 위험도와 매매 시점 제안
        -> 오늘 차트기반의 테슬라 주가 분석 해줘, 오늘 멀티모달 기반의 애플 주가 분석 해줘
                                      
        각 질문의 성격에 따라 적절한 도구를 선택하여 최적의 분석을 제공하세요.
        """)
        
        all_messages = [system_prompt]
        for history in state["chat_history"]:
            if history["role"] == "user":
                all_messages.append(HumanMessage(content=history["content"]))
            elif history["role"] == "assistant":
                all_messages.append(SystemMessage(content=history["content"]))
        all_messages.extend(state["messages"])
        
        response = model.invoke(all_messages, config)
        return {"messages": [response], "chat_history": state["chat_history"]}
    
    # 그래프 구성
    workflow = StateGraph(MarketAgentState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        lambda x: "end" if not hasattr(x["messages"][-1], "tool_calls") 
                 or not x["messages"][-1].tool_calls else "continue",
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    
    # 체크포인터 설정과 함께 컴파일
    return workflow.compile(checkpointer=memory)