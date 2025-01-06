from uuid import uuid4
from langchain_core.messages import HumanMessage
from ..agents.super_agent import create_supervisor_agent
import logging

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.agent = None
        self.thread_id = str(uuid4())
        
    def sync_initialize(self):
        """동기 방식으로 에이전트 초기화"""
        if self.agent is None:
            self.agent = create_supervisor_agent()
            
    def sync_process_chat(self, message: str, chat_history: list = None):
        """동기 방식으로 채팅 처리
        
        Args:
            message: 사용자 메시지
            chat_history: 이전 대화 기록 (optional)
            
        Returns:
            dict: 응답 메시지와 업데이트된 대화 기록을 포함하는 딕셔너리
        """
        if self.agent is None:
            self.sync_initialize()
            
        try:
            result = self.agent.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "chat_history": chat_history or []
                },
                {"configurable": {"thread_id": self.thread_id}}
            )
            
            return {
                "response": result["messages"][-1].content if result.get("messages") else "응답을 생성할 수 없습니다.",
                "chat_history": result.get("chat_history", [])
            }
            
        except Exception as e:
            logger.error(f"채팅 처리 중 오류 발생: {str(e)}")
            raise 