from agents.super_agent import create_supervisor_agent
from langchain_core.messages import HumanMessage
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chatbot():
    try:
        # 슈퍼바이저 에이전트 생성
        agent = create_supervisor_agent()
        
        # 초기 상태 설정
        initial_state = {
            "messages": [],
            "chat_history": [],
            "next": "START"
        }
        
        print("챗봇과 대화를 시작합니다. 종료하려면 'quit'를 입력하세요.")
        
        while True:
            # 사용자 입력 받기
            user_input = input("\n사용자: ")
            
            if user_input.lower() == 'quit':
                print("대화를 종료합니다.")
                break
                
            # 현재 메시지 추가
            current_state = initial_state.copy()
            current_state["messages"] = [HumanMessage(content=user_input)]
            
            try:
                # 에이전트 실행
                response = agent.invoke(current_state)
                
                # 응답 출력
                if response and "messages" in response:
                    print(f"챗봇: {response['messages'][-1].content}")
                else:
                    print("챗봇: 죄송합니다. 응답을 생성하는데 문제가 발생했습니다.")
                    
            except Exception as e:
                print(f"챗봇: 죄송합니다. 오류가 발생했습니다: {str(e)}")
                
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    test_chatbot()