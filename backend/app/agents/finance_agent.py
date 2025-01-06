from typing import Annotated, Sequence, TypedDict, Dict, Any, List
import json
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.vector_db_tool import VectorDBSearchTool

class FinanceAgentState(TypedDict):
    """금융 상담 에이전트의 상태"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    chat_history: List[Dict[str, str]]

def create_finance_agent(memory=None):
    """금융 정보 검색 에이전트 생성"""
    # 도구 초기화
    vector_db_tool = VectorDBSearchTool()
    tools = [vector_db_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    
    # LLM 초기화
    model = ChatOpenAI(model="gpt-4o-mini")
    model = model.bind_tools(tools)
    
    # 도구 노드
    def tool_node(state: FinanceAgentState) -> Dict:
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
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
        state: FinanceAgentState,
        config: RunnableConfig,
    ) -> Dict:
        system_prompt = SystemMessage(content="""
        금융 상담 AI 어시스턴트입니다. 
        vector_db_search 도구를 사용하여 사용자의 금융 관련 질문에 답변합니다.
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
    workflow = StateGraph(FinanceAgentState)
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
    
    return workflow.compile(checkpointer=memory)