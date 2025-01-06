import os
import sys
import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
import yfinance as yf
from datetime import datetime, timedelta
import psycopg2
from dotenv import load_dotenv
import json
from constants import INDICES, STOCKS  # 로컬 constants.py에서 가져오기

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL 연결 설정
DB_CONFIG = {
    'dbname': os.getenv('PSQL_DATABASE'),
    'user': os.getenv('PSQL_USERNAME'),
    'password': os.getenv('PSQL_PASSWORD'),
    'host': os.getenv('PSQL_HOST'),
    'port': os.getenv('PSQL_PORT')
}

async def update_market_data():
    logger.info("시장 데이터 업데이트 시작...")
    try:
        # PostgreSQL 연결
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # 모든 시장 데이터를 저장할 리스트
        market_data = []
        
        # 지수 데이터 업데이트 (KOSPI, KOSDAQ 포함)
        indices = {
            **INDICES,
            "KOSPI": "KS11",
            "KOSDAQ": "KQ11"
        }
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(f"^{symbol}")
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    price_change = hist['Close'].iloc[-1] - hist['Open'].iloc[0]
                    change_percent = (price_change / hist['Open'].iloc[0]) * 100
                    
                    # 차트 데이터 생성
                    chart_data = [
                        {
                            'time': str(idx),
                            'value': row['Close']
                        } for idx, row in hist.iterrows()
                    ]
                    
                    market_data.append({
                        'name': name,
                        'symbol': symbol,
                        'current': current_price,
                        'change': price_change,
                        'changePercent': change_percent,
                        'flag': 'down' if price_change < 0 else 'up',
                        'color': '#ef4444' if price_change < 0 else '#22c55e',
                        'data': chart_data,
                        'is_index': True
                    })
                    logger.info(f"지수 업데이트 완료: {name} ({symbol})")
                    
            except Exception as e:
                logger.error(f"지수 업데이트 오류 {symbol}: {str(e)}")
                continue

        # 빅테크 주식 데이터 업데이트
        for name, symbol in STOCKS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    price_change = hist['Close'].iloc[-1] - hist['Open'].iloc[0]
                    change_percent = (price_change / hist['Open'].iloc[0]) * 100
                    
                    # 차트 데이터 생성
                    chart_data = [
                        {
                            'time': str(idx),
                            'value': row['Close']
                        } for idx, row in hist.iterrows()
                    ]
                    
                    market_data.append({
                        'name': name,
                        'symbol': symbol,
                        'current': current_price,
                        'change': price_change,
                        'changePercent': change_percent,
                        'flag': 'down' if price_change < 0 else 'up',
                        'color': '#ef4444' if price_change < 0 else '#22c55e',
                        'data': chart_data,
                        'is_index': False
                    })
                    logger.info(f"주식 업데이트 완료: {name} ({symbol})")
                    
            except Exception as e:
                logger.error(f"주식 업데이트 오류 {symbol}: {str(e)}")
                continue

        # 데이터베이스에 저장
        try:
            # market_data 테이블 업데이트
            cur.execute("TRUNCATE TABLE market_data")  # 기존 데이터 삭제
            for data in market_data:
                cur.execute("""
                    INSERT INTO market_data (
                        name, symbol, current_price, price_change, 
                        change_percent, flag, color, chart_data, is_index
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    data['name'], data['symbol'], data['current'], 
                    data['change'], data['changePercent'], data['flag'],
                    data['color'], json.dumps(data['data']), data['is_index']
                ))
            
            conn.commit()
            logger.info("시장 데이터 업데이트가 성공적으로 완료되었습니다")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"데이터베이스 오류: {str(e)}")
            
        finally:
            cur.close()
            conn.close()

    except Exception as e:
        logger.error(f"스케줄러 오류: {str(e)}")

async def main():
    # 스케줄러 초기화
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        func=update_market_data,
        trigger=IntervalTrigger(minutes=5),
        id='market_data_update',
        name='5분마다 시장 데이터 업데이트',
        replace_existing=True,
        max_instances=1
    )

    # 에러 핸들러
    def handle_scheduler_error(event):
        if hasattr(event, 'code'):
            logger.error(f"스케줄러 오류: {event.code}")
        if hasattr(event, 'job_id'):
            logger.error(f"작업 ID: {event.job_id}")
        if hasattr(event, 'scheduled_run_time'):
            logger.error(f"예약된 실행 시간: {event.scheduled_run_time}")

    scheduler.add_listener(handle_scheduler_error, EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    
    try:
        scheduler.start()
        logger.info("스케줄러가 성공적으로 시작되었습니다")
        # 무한 실행
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("스케줄러가 성공적으로 종료되었습니다")

if __name__ == "__main__":
    asyncio.run(main())