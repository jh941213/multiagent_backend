# multiagent_backend

가짜연 9th 깃허브 잔디심기 Stockelper Multi Agent Backend Fastapi
<img width="1077" alt="스크린샷 2025-01-06 오후 9 49 32" src="https://github.com/user-attachments/assets/449a2d67-8d14-4dff-aa42-b8b78be5cebf" />
<img width="723" alt="스크린샷 2025-01-06 오후 9 52 28" src="https://github.com/user-attachments/assets/639134a3-8368-49e3-b820-367ea86fc37c" />


## 🤖 AI Agent Structure

### 1. Supervisor Agent (super_agent.py)
- Analyzes user input and routes it to the appropriate subagents
- Supports human-in-the-loop collaboration
- Manages conversation history and maintains context

### 2. Financial Information Agent (finance_agent.py)
- Provides information about accounts, fees, and financial services
- Supports information search based on Vector DB
- Provides financial counseling services

### 3. Market Analysis Agent (market_agent.py)
- Analyzes real-time stock price information
- Provides technical/fundamental analysis
- Analyzes charts and identifies market trends

### 4. HIL Agent (hil_agent.py)
- Generate human-in-the-loop-based research reports
- Create and manage expert personas
- Perform multi-turn conversation-based analysis

## 🛠 Key Features

1. **Real-time market analysis**
   - Get real-time stock price data
   - Technical indicator analysis
   - Analyze chart patterns

2. **Financial Information Search**
   - Vector DB-based information search
   - Context-based response generation
   - Financial counseling service

3. **Research report generation**
   - Human-in-the-loop collaboration
   - Integrate expert analysis
   - Generate structured reports

4. **Multimodal Analytics**
   - Chart image analysis
   - YouTube content search
   - News data integration
