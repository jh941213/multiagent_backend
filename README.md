# multiagent_backend

가짜연 9th 깃허브 잔디심기 Stockelper Multi Agent Backend Fastapi
<img width="1077" alt="스크린샷 2025-01-06 오후 9 49 32" src="https://github.com/user-attachments/assets/449a2d67-8d14-4dff-aa42-b8b78be5cebf" />
<img width="723" alt="스크린샷 2025-01-06 오후 9 52 28" src="https://github.com/user-attachments/assets/639134a3-8368-49e3-b820-367ea86fc37c" />


## 🤖 AI 에이전트 구조

### 1. 슈퍼바이저 에이전트 (super_agent.py)
- 사용자 입력을 분석하여 적절한 하위 에이전트로 라우팅
- Human-in-the-loop 방식의 협업 지원
- 대화 기록 관리 및 컨텍스트 유지

### 2. 금융 정보 에이전트 (finance_agent.py)
- 계좌, 수수료, 금융 서비스 관련 정보 제공
- Vector DB 기반 정보 검색 지원
- 금융 상담 서비스 제공

### 3. 시장 분석 에이전트 (market_agent.py)
- 실시간 주가 정보 분석
- 기술적/기본적 분석 제공
- 차트 분석 및 시장 동향 파악

### 4. HIL 에이전트 (hil_agent.py)
- Human-in-the-loop 기반 리서치 리포트 생성
- 전문가 페르소나 생성 및 관리
- 멀티턴 대화 기반 분석 수행

## 🛠 주요 기능

1. **실시간 시장 분석**
   - 주가 데이터 실시간 조회
   - 기술적 지표 분석
   - 차트 패턴 분석

2. **금융 정보 검색**
   - Vector DB 기반 정보 검색
   - 맥락 기반 응답 생성
   - 금융 상담 서비스

3. **리서치 리포트 생성**
   - Human-in-the-loop 방식 협업
   - 전문가 분석 통합
   - 구조화된 보고서 생성

4. **멀티모달 분석**
   - 차트 이미지 분석
   - 유튜브 콘텐츠 검색
   - 뉴스 데이터 통합
