# stock_report_pipeline.py

import ast
import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_teddynote.graphs import visualize_graph  # 시각화 함수(사용 안 할 경우 제거 가능)
from langgraph.graph import START, END, StateGraph
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver

# 필요 라이브러리
# from IPython.display import Markdown  # 주피터 노트북 전용
# import IPython.display  # 주피터 노트북 전용

# LLM 연결
from langchain_aws import ChatBedrock

llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs=dict(temperature=0, max_tokens=8192),
    credentials_profile_name="default",
    region_name="us-east-1"
)


##############################
# 1) KDBanalyst 클래스 정의
##############################
class KDBanalyst(BaseModel):
    affiliation: str = Field(
        description="리포터의 소속 회사",
        default=""
    )
    name: str = Field(
        description="리포터의 이름",
        default="김재현"
    )
    expertise: str = Field(
        description="주식 시장 전문 분야",
        default="나스닥 개인투자자"
    )
    analysis_focus: str = Field(
        description="주식 분석의 주요 관점",
        default="기술적 분석, 펀더멘털 분석, 시장 동향 분석"
    )
    investment_style: str = Field(
        description="투자 스타일과 전략",
        default="단기 투자 및 스켈핑 유저"
    )
    risk_management: str = Field(
        description="리스크 관리 방식",
        default="포트폴리오 분산, 손절매 규칙 준수"
    )

    @property
    def profile(self) -> str:
        return f"""
        📊 Stock Reporter Profile 📊
        소속: {self.affiliation}
        이름: {self.name}
        전문분야: {self.expertise}
        분석관점: {self.analysis_focus}
        투자스타일: {self.investment_style}
        리스크관리: {self.risk_management}
        """

    @property
    def description(self) -> str:
        return f"""
        전문분야: {self.expertise}
        분석관점: {self.analysis_focus}
        투자스타일: {self.investment_style}
        리스크관리: {self.risk_management}
        """


##############################
# 2) Perspectives 클래스 정의
##############################
class Perspectives(BaseModel):
    reporters: List[KDBanalyst] = Field(
        description="리포트 작성에 참여한 리포터 목록",
    )


##############################
# 3) GenerateAnalystsState 타입 정의
##############################
class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[KDBanalyst]


##############################
# 4) 분석가 생성 관련 함수 및 프롬프트
##############################
analyst_instructions = """주식 분석가 페르소나 생성을 담당하게 됩니다.

다음 지침을 주의 깊게 따라주세요:
1. 먼저 연구 주제를 검토하세요:

{topic}

2. 분석가 생성을 안내하기 위해 제공된 편집 피드백을 검토하세요:

{human_analyst_feedback}

3. 위의 문서 및/또는 피드백을 바탕으로 주식 분석에 필요한 주요 관점을 파악하세요.

4. 상위 {max_analysts}개의 관점을 선택하세요.

5. 각 관점별로 다음 속성을 가진 분석가를 생성하세요:
   - 소속: 4대은행(국민, 신한, 우리, 하나) 별 투자증권회사 및 금융기관
   - 이름: 한국식 이름
   - 전문분야: 주식시장 특정 분야
   - 분석관점: 기술적/펀더멘털/시장동향 분석 중 선택
   - 투자스타일: 단기/중기/장기 투자 전략
   - 리스크관리: 구체적인 리스크 관리 방식"""

def create_analysts(state: GenerateAnalystsState):
    """분석가 페르소나를 생성하는 함수"""
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    structured_llm = llm.with_structured_output(Perspectives)

    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )

    # LLM 호출
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="KDBanalyst 클래스 형식에 맞는 분석가 집합을 생성하세요.")]
    )
    return {"analysts": analysts.reporters}


def human_feedback(state: GenerateAnalystsState):
    """사용자 피드백을 받기 위한 중단점 노드"""
    pass


def should_continue(state: GenerateAnalystsState):
    """워크플로우의 다음 단계를 결정하는 함수"""
    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback:
        return "create_analysts"
    return END


##############################
# 5) 분석가 그래프 작성
##############################
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")

builder.add_conditional_edges(
    "human_feedback", should_continue, ["create_analysts", END]
)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

# 필요 시 그래프 시각화
# visualize_graph(graph)


###################################
# 6) 인터뷰 흐름(질문/검색/답변) 관련 코드
###################################
from langchain_core.messages import get_buffer_string

# 상태 정의
from langgraph.graph import MessagesState

class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: KDBanalyst
    interview: str
    sections: list


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


question_instructions = """당신은 Human-in-the-loop 방식으로 전문가와 협업하는 분석가입니다.

당신의 목표는 전문가와의 상호작용을 통해 더 나은 결과물을 도출하는 것입니다.

1. 협업: 전문가의 도메인 지식을 최대한 활용하여 함께 문제를 해결
2. 반복: 전문가의 피드백을 받아 지속적으로 개선
3. 구체화: 추상적인 아이디어를 구체적인 해결책으로 발전

다음은 당신이 집중해야 할 주제와 목표입니다: {goals}

먼저 당신의 페르소나에 맞는 이름으로 자신을 소개하고, 협업의 목적을 설명하세요.

전문가의 의견을 경청하고 이해한 내용을 바탕으로 구체적인 질문을 이어가세요.

충분한 협업이 이루어졌다고 판단되면 "협업해 주셔서 감사합니다!"라고 마무리하세요.

전체 과정에서 Human-in-the-loop 철학에 맞는 협업적이고 반복적인 접근을 유지하세요."""


def generate_question(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]

    system_msg = question_instructions.format(goals=analyst.profile)
    question = llm.invoke([SystemMessage(content=system_msg)] + messages)

    return {"messages": [question]}


# TavilySearch, YouTubeSearchTool 등
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_community.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi


def search_web(state: InterviewState):
    tavily_search = TavilySearch(max_results=3)
    structured_llm = llm.with_structured_output(SearchQuery)

    # 메시지를 검색 쿼리로 변환
    messages = state["messages"]
    search_query = structured_llm.invoke([
        SystemMessage(content="\n\nHuman: 검색 쿼리를 생성해주세요.\n\nAssistant:"),
        HumanMessage(content=f"\n\nHuman: {messages}\n\nAssistant:")
    ])

    # 실제 검색 수행
    search_docs = tavily_search.invoke(search_query.search_query)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"context": [formatted_search_docs]}


def search_youtube(state: InterviewState):
    structured_llm = llm.with_structured_output(SearchQuery)
    messages = state["messages"]

    # 마지막 AIMessage는 검색에 직접 반영하지 않도록 필터링(예: 불필요한 반복 방지)
    filtered_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg == messages[-1]:
            continue
        filtered_messages.append(msg)

    # 검색 쿼리 생성
    search_query = structured_llm.invoke([
        SystemMessage(content="\n\nHuman: 검색 쿼리를 생성해주세요.\n\nAssistant: 네, 검색 쿼리를 생성하겠습니다."),
        HumanMessage(content=f"\n\nHuman: {filtered_messages}\n\nAssistant:")
    ])

    def get_video_info(url):
        try:
            video_id = url.split('watch?v=')[1].split('&')[0]
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
            text = "\n".join([item['text'] for item in transcript])
            return {
                'video_id': video_id,
                'url': url,
                'transcript': text
            }
        except Exception as e:
            print(f"동영상 처리 중 오류 발생: {str(e)}")
            return None

    try:
        youtube_search_tool = YouTubeSearchTool()
        youtube_results = youtube_search_tool.run(search_query.search_query)
        if isinstance(youtube_results, str):
            youtube_results = ast.literal_eval(youtube_results)

        formatted_results = []
        for url in youtube_results:
            if isinstance(url, str) and 'youtube.com' in url:
                video_info = get_video_info(url)
                if video_info:
                    formatted_results.append(
                        f'<Document source="youtube" url="{video_info["url"]}">\n'
                        f'<Content>\n{video_info["transcript"][:1000]}...\n</Content>\n'
                        f'</Document>'
                    )

        formatted_search_docs = "\n\n---\n\n".join(formatted_results)
        return {"context": [formatted_search_docs]}

    except Exception as e:
        print(f"YouTube 검색 중 오류 발생: {str(e)}")
        return {
            "context": ["<Error>YouTube 검색 결과를 가져오는데 실패했습니다.</Error>"]
        }


answer_instructions = """당신은 분석가와 인터뷰를 하는 전문가입니다.

분석가의 관심 분야는 다음과 같습니다: {goals}

당신의 목표는 인터뷰어가 제기한 질문에 답변하는 것입니다.

질문에 답변하기 위해 다음 컨텍스트를 사용하세요:

{context}

답변 시 다음 지침을 따르세요:

1. 제공된 컨텍스트의 정보만 사용하세요.
2. 컨텍스트에 명시적으로 언급되지 않은 외부 정보나 가정을 도입하지 마세요.
3. 컨텍스트에는 각 개별 문서의 상단에 출처가 포함되어 있습니다.
4. 답변에서 관련 진술 옆에 이러한 출처를 포함하세요. 예: [1]
5. 답변 하단에 출처를 순서대로 나열하세요.
6. 불필요한 개인 정보 노출, 전문가의 실명 언급 등은 피해주세요."""


def generate_answer(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    if not context:
        context = ["검색 결과가 없습니다."]

    system_message = answer_instructions.format(
        goals=analyst.description,
        context="\n\n".join(context)
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content="\n컨텍스트:\n" + "\n".join(context) +
                         "\n\n질문:\n" + messages[-1].content)
        ])

        answer = AIMessage(content=response.content, name="expert")
        return {"messages": [answer]}

    except Exception as e:
        print(f"Error generating answer: {e}")
        return {"messages": [AIMessage(content="답변 생성 중 오류가 발생했습니다.", name="expert")]}


def save_interview(state: InterviewState):
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return {"interview": interview}


def route_messages(state: InterviewState, name: str = "expert"):
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # 전문가가 최대 턴 수 이상 답변한 경우 종료
    if num_responses >= max_num_turns:
        return "save_interview"

    # 마지막 질의에 "도움 주셔서 감사합니다" 등의 표현이 있으면 종료
    last_question = messages[-2] if len(messages) >= 2 else None
    if last_question and "도움 주셔서 감사합니다" in last_question.content:
        return "save_interview"

    return "ask_question"


section_writer_instructions = """당신은 전문 기술 작가입니다.

당신의 임무는 소스 문서 세트를 철저히 분석하여 상세하고 포괄적인 보고서 섹션을 작성하는 것입니다.
- 보고서에 대한 구조화된 마크다운 양식 활용
- 상세하고 논리적인 전개
- 가능한 한 풍부한 예시와 근거를 제시
- 전문적이고 객관적인 어조 유지
- 최소 800단어 이상

(필요 시 수정 가능)"""


def write_section(state: InterviewState):
    context = state["context"]
    analyst = state["analyst"]

    system_message = section_writer_instructions  # 필요하다면 format으로 analyst 정보 사용
    section = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"이 소스를 사용하여 섹션을 작성하세요: {context}")]
    )

    return {"sections": [section.content]}


##############################
# 7) 인터뷰용 그래프 정의
##############################
stock_interview_builder = StateGraph(InterviewState)
stock_interview_builder.add_node("ask_question", generate_question)
stock_interview_builder.add_node("search_web", search_web)
stock_interview_builder.add_node("search_youtube", search_youtube)
stock_interview_builder.add_node("answer_question", generate_answer)
stock_interview_builder.add_node("save_interview", save_interview)
stock_interview_builder.add_node("write_section", write_section)

stock_interview_builder.add_edge(START, "ask_question")
stock_interview_builder.add_edge("ask_question", "search_web")
stock_interview_builder.add_edge("ask_question", "search_youtube")
stock_interview_builder.add_edge("search_web", "answer_question")
stock_interview_builder.add_edge("search_youtube", "answer_question")
stock_interview_builder.add_conditional_edges(
    "answer_question", route_messages, ["ask_question", "save_interview"]
)
stock_interview_builder.add_edge("save_interview", "write_section")
stock_interview_builder.add_edge("write_section", END)

memory2 = MemorySaver()
stock_interview_graph = stock_interview_builder.compile(checkpointer=memory2).with_config(
    run_name="Conduct Interviews"
)

# 필요 시 시각화
# visualize_graph(stock_interview_graph)


##############################
# 8) 통합된 리서치/보고서 그래프
##############################
class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[KDBanalyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


def initiate_all_interviews(state: ResearchGraphState):
    human_analyst_feedback = state.get("human_analyst_feedback")
    if human_analyst_feedback:
        return "create_analysts"
    else:
        topic = state["topic"]
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "messages": [
                        HumanMessage(
                            content=f"So you said you were writing an article on {topic}?"
                        )
                    ],
                },
            )
            for analyst in state["analysts"]
        ]


report_writer_instructions = """당신은 다음 주제에 대한 투자 보고서를 작성하는 애널리스트입니다:

{topic}

여러 분석가들이 각각 다음과 같은 작업을 수행했습니다:

1. 해당 분야 전문가와의 인터뷰 진행
2. 분석 내용을 메모로 정리

당신의 임무:

1. 각 분석가들의 메모를 검토합니다.
2. 각 메모의 핵심 내용을 면밀히 분석합니다.
3. 모든 메모의 핵심 아이디어를 통합하여 종합적인 요약을 작성합니다.
4. 각 메모의 주요 포인트를 아래 섹션에 맞게 논리적으로 구성합니다.
5. 모든 필수 섹션을 `### 섹션명` 형식의 헤더로 포함시킵니다.
6. 각 섹션당 약 250자 내외로 심도있는 설명과 근거를 제시합니다.

**보고서 섹션 구성:**

- **시장 환경**
- **산업 분석**
- **기업 분석**
- **투자 포인트**
- **리스크 요인**
- **실적 전망**
- **투자의견**

보고서 형식:
1. 마크다운 형식 사용
2. 서두 없이 바로 본문 시작
3. 소제목 사용하지 않음
4. 보고서는 ## 투자 분석 헤더로 시작
5. 분석가 이름 언급하지 않음
6. 메모의 인용 출처는 [1], [2] 등으로 표시
7. 마지막에 ## 참고자료 섹션에 출처 목록 정리
8. 출처는 순서대로 나열하고 중복 제거
"""


def write_report(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = report_writer_instructions.format(
        topic=topic, context=formatted_str_sections
    )
    report = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"이 메모들을 바탕으로 보고서를 작성해주세요.")]
    )
    return {"content": report.content}


intro_conclusion_instructions = """당신은 {topic}에 대한 투자 보고서를 마무리하는 애널리스트입니다.

보고서의 모든 섹션이 주어질 것입니다.

당신의 임무는 간결하고 설득력 있는 서론 또는 결론을 작성하는 것입니다.

불필요한 서두는 생략합니다.

약 200자 내외로, 서론의 경우 보고서의 모든 섹션을 미리보기하고, 결론의 경우 핵심 내용을 요약합니다.

마크다운 형식을 사용합니다.

서론의 경우 매력적인 제목을 만들고 # 헤더를 사용합니다.

서론은 ## 개요 헤더를 사용합니다.

결론은 ## 결론 헤더를 사용합니다.

참고할 섹션들은 다음과 같습니다: {formatted_str_sections}"""


def write_introduction(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    intro = llm.invoke(
        [SystemMessage(content=instructions)]
        + [HumanMessage(content=f"보고서의 서론을 작성해주세요")]
    )
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke(
        [SystemMessage(content=instructions)]
        + [HumanMessage(content=f"보고서의 결론을 작성해주세요")]
    )
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    content = state["content"]
    # "## 투자 분석" 제거(필요 시)
    if content.startswith("## 투자 분석"):
        content = content.strip("## 투자 분석")

    if "## 참고자료" in content:
        try:
            content, sources = content.split("\n## 참고자료\n", maxsplit=1)
        except:
            sources = None
    else:
        sources = None

    final_report = (
        state["introduction"]
        + "\n\n---\n\n## 핵심 투자포인트\n\n"
        + content
        + "\n\n---\n\n"
        + state["conclusion"]
    )
    if sources is not None:
        final_report += "\n\n## 참고자료\n" + sources

    return {"final_report": final_report}


##############################
# 9) 최종 리서치 그래프 구성
##############################
research_builder = StateGraph(ResearchGraphState)

research_builder.add_node("create_analysts", create_analysts)
research_builder.add_node("human_feedback", human_feedback)
research_builder.add_node("conduct_interview", stock_interview_graph)
research_builder.add_node("write_report", write_report)
research_builder.add_node("write_introduction", write_introduction)
research_builder.add_node("write_conclusion", write_conclusion)
research_builder.add_node("finalize_report", finalize_report)

research_builder.add_edge(START, "create_analysts")
research_builder.add_edge("create_analysts", "human_feedback")
research_builder.add_conditional_edges(
    "human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"]
)
research_builder.add_edge("conduct_interview", "write_report")
research_builder.add_edge("conduct_interview", "write_introduction")
research_builder.add_edge("conduct_interview", "write_conclusion")
research_builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
research_builder.add_edge("finalize_report", END)

memory3 = MemorySaver()
research_graph = research_builder.compile(interrupt_before=["human_feedback"], checkpointer=memory3)