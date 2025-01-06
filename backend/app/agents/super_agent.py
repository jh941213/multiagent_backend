from typing import Annotated, Sequence, TypedDict, Dict, List, Literal
from pydantic import BaseModel
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_models import ChatClovaX
import os
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row


from app.agents.finance_agent import create_finance_agent, FinanceAgentState
from app.agents.market_agent import create_market_agent, MarketAgentState
import uuid

from dotenv import load_dotenv
load_dotenv()
# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 파일 로깅 설정
file_handler = logging.FileHandler('supervisor.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 콘솔 로깅 설정
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 슈퍼바이저 상태 클래스 정의
class SupervisorState(TypedDict):
    """슈퍼바이저의 상태를 정의하는 클래스
    messages: 현재 대화 메시지들
    chat_history: 전체 대화 기록
    next: 다음 실행할 에이전트 지정
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    chat_history: List[Dict[str, str]]
    next: str

# 라우팅 결정을 위한 응답 모델
class RouteResponse(BaseModel):
    """라우팅 결정을 위한 응답 모델
    next: 다음에 실행할 에이전트 (FinanceAgent/MarketAgent/CHAT/FINISH)
    """
    next: Literal["FinanceAgent", "MarketAgent", "CHAT", "FINISH"]

class PostgresConnectionManager:
    def __init__(self):
        self.pool = None

    async def get_pool(self):
        if self.pool is None:
            self.pool = AsyncConnectionPool(
                conninfo=f"postgres://{os.getenv('PSQL_USERNAME')}:{os.getenv('PSQL_PASSWORD')}"
                f"@{os.getenv('PSQL_HOST')}:{os.getenv('PSQL_PORT')}/{os.getenv('PSQL_DATABASE')}"
                f"?sslmode={os.getenv('PSQL_SSLMODE')}",
                max_size=20,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                }
            )
            await self.pool.open()
        return self.pool

    async def close_pool(self):
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

# 전역 connection manager 인스턴스 생성
postgres_manager = PostgresConnectionManager()

def create_supervisor_agent(memory=None):
    """슈퍼바이저 에이전트를 생성하는 메인 함수"""
    logger.info("슈퍼바이저 에이전트 생성 시작")
    
    # 메모리 저장소 설정
    if memory is None:
        try:
            # 기본 메모리 저장소 사용
            memory = MemorySaver()
            logger.info("기본 메모리 저장소 초기화 완료")
        except Exception as e:
            logger.error(f"메모리 저장소 초기화 실패: {e}")
            # 폴백: 인메모리 저장소
            memory = MemorySaver()
    
    # 워크플로우 체크포인터 설정
    checkpointer = memory
    
    # 하위 에이전트들 초기화
    finance_agent = create_finance_agent(memory)
    market_agent = create_market_agent(memory)
    logger.info("하위 에이전트 초기화 완료")
    
    # GPT-4 모델 초기화 (온도값 0.7로 설정하여 적당한 창의성 부여)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    logger.info("LLM 모델 초기화 완료")
    
    # 슈퍼바이저의 메인 프롬프트 설정
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 친절하고 전문적인 AI 어시스턴트입니다.
        
        입력을 정확히 분석하여 다음과 같이 라우팅하세요:
        
        1. 주가/시장 관련 키워드:
           - 주가, 차트, 시세, 전망, 분석 → MarketAgent
           - 예: "삼성전자 주가", "코스피 전망"
        
        2. 금융 서비스 관련 키워드:
           - 계좌, 수수료, 매매, 인증, 위험 → FinanceAgent
           - 예: "계좌 개설 방법", "매매 수수료"
        
        3. 기타 일반 대화:
           - 인사, 감사, 일상 대화 → CHAT
           
        명확하지 않은 경우, 사용자에게 구체적인 의도를 물어보세요.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "다음 중 하나를 선택하세요: FinanceAgent, MarketAgent, CHAT")
    ])
    
    def supervisor_node(state: SupervisorState) -> Dict:
        """사용자 입력을 분석하여 적절한 에이전트로 라우팅하는 슈퍼바이저 노드
        
        Args:
            state: 현재 대화 상태 정보
            
        Returns:
            Dict: 업데이트된 상태 정보와 다음 실행할 에이전트
        """
        try:
            # 채팅 기록을 메시지로 변환
            chat_history_messages = []
            for history in state["chat_history"]:
                if history["role"] == "user":
                    chat_history_messages.append(HumanMessage(content=history["content"]))
                elif history["role"] == "assistant":
                    chat_history_messages.append(SystemMessage(content=history["content"]))

            # 라우팅 결정
            route_chain = supervisor_prompt | model.with_structured_output(RouteResponse)
            result = route_chain.invoke({
                "messages": state["messages"],
                "chat_history": chat_history_messages
            })
            logger.info(f"라우팅 결정: {result.next}")
            
            # 일반 대화(CHAT)인 경우 HyperCLOVA X로 응답 생성
            if result.next == "CHAT":
                # HyperCLOVA X 모델 초기화
                clova_model = ChatClovaX(
                    model="HCX-003",
                    temperature=0.1,
                    max_tokens=1000,
                    task_id="oi5ojnb8"
                )
                
                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", "당신은 경제유투버 슈카입니다.당신의 말투로 답변하세요.유저의 질문에 대하여 정확하게 답변하세요."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    MessagesPlaceholder(variable_name="messages")
                ])
                
                # HyperCLOVA X로 응답 생성
                response = clova_model.invoke(chat_prompt.format_messages(
                    messages=state["messages"],
                    chat_history=chat_history_messages
                ))
                
                new_messages = list(state["messages"])
                new_messages.append(response)
                
                # 대화 기록 업데이트
                chat_history = state["chat_history"]
                if state["messages"]:
                    chat_history.append({
                        "role": "user", 
                        "content": state["messages"][-1].content
                    })
                chat_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                return {
                    "messages": new_messages,
                    "chat_history": chat_history,
                    "next": "FINISH"
                }
            
            # 다른 에이전트로 라우팅
            return {
                "messages": state["messages"],
                "chat_history": state["chat_history"],
                "next": result.next
            }
        except Exception as e:
            logger.error(f"슈퍼바이저 노드 에러: {str(e)}")
            return {
                "messages": [SystemMessage(content="죄송합니다. 일시적인 오류가 발생했습니다.")],
                "chat_history": state["chat_history"],
                "next": "FINISH"
            }
    
    def finance_node(state: SupervisorState) -> Dict:
        """금융 상담 에이전트 노드"""
        logger.info("금융 에이전트 노드 실행")
        try:
            finance_state = FinanceAgentState(
                messages=state["messages"],
                chat_history=state["chat_history"]
            )
            result = finance_agent.invoke(finance_state)
            
            # 응답이 있는지 확인
            if result and "messages" in result and result["messages"]:
                # 채팅 기록 업데이트
                new_chat_history = result.get("chat_history", state["chat_history"])
                
                # 응답 메시지 확인 및 변환
                response_message = result["messages"][-1]
                if hasattr(response_message, 'content'):
                    return {
                        "messages": result["messages"],
                        "chat_history": new_chat_history,
                        "next": "FINISH"
                    }
            
            raise ValueError("금융 에이전트로부터 유효한 응답을 받지 못했습니다")
            
        except Exception as e:
            logger.error(f"금융 에이전트 처리 중 오류: {str(e)}")
            error_message = SystemMessage(content="죄송합니다. 금융 정보 처리 중 오류가 발생했습니다.")
            return {
                "messages": [error_message],
                "chat_history": state["chat_history"],
                "next": "FINISH"
            }
        
    def market_node(state: SupervisorState) -> Dict:
        """시장 분석 에이전트 노드
        주가, 시장 동향, 차트 분석 등을 처리
        
        Args:
            state: 현재 대화 상태
            
        Returns:
            Dict: 시장 분석 결과와 업데이트된 대화 기록
        """
        logger.debug("시장 분석 에이전트 노드 실행")
        market_state = MarketAgentState(
            messages=state["messages"],
            chat_history=state["chat_history"]
        )
        result = market_agent.invoke(market_state)
        logger.info("시장 분석 에이전트 응답 완료")
        return {
            "messages": result["messages"],
            "chat_history": result["chat_history"],
            "next": "FINISH"
        }
    
    # 워크플로우 그래프 구성
    logger.info("워크플로우 그래프 구성 시작")
    workflow = StateGraph(SupervisorState)
    
    # 각 에이전트 노드 추가
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("FinanceAgent", finance_node)
    workflow.add_node("MarketAgent", market_node)
    
    # 시작점을 슈퍼바이저로 설정
    workflow.set_entry_point("Supervisor")
    
    # 조건부 라우팅 규칙 설정
    conditional_map = {
        "FinanceAgent": "FinanceAgent",
        "MarketAgent": "MarketAgent",
        "CHAT": "Supervisor",
        "FINISH": END
    }
    
    def get_next(state):
        return state["next"]
    
    # 조건부 엣지 추가
    workflow.add_conditional_edges("Supervisor", get_next, conditional_map)
    
    logger.info("워크플로우 그래프 구성 완료")
    # 체크포인터 설정과 함께 컴파일
    try:
        return workflow.compile(checkpointer=checkpointer)
    except Exception as e:
        logger.error(f"워크플로우 컴파일 중 오류: {e}")
        # 폴백: 체크포인터 없이 컴파일
        return workflow.compile()
