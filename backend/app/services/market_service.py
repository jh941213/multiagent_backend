from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# PostgreSQL 연결 설정
DB_CONFIG = {
    "dbname": "stockhelper",
    "user": "kdb",
    "password": "1234",
    "host": "localhost"
}

class MarketService:
    @staticmethod
    async def save_market_data(data: List[Dict[str, Any]], is_index: bool) -> None:
        conn = None
        cur = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            
            for item in data:
                query = """
                INSERT INTO market_data 
                (name, symbol, current_price, price_change, change_percent, flag, color, chart_data, is_index)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) 
                DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    price_change = EXCLUDED.price_change,
                    change_percent = EXCLUDED.change_percent,
                    color = EXCLUDED.color,
                    chart_data = EXCLUDED.chart_data,
                    created_at = CURRENT_TIMESTAMP;
                """
                
                cur.execute(
                    query,
                    (
                        item['name'],
                        item['symbol'],
                        item['current'],
                        item['change'],
                        item['changePercent'],
                        item['flag'],
                        item['color'],
                        json.dumps(item['data']),
                        is_index
                    )
                )
            
            conn.commit()
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"데이터 저장 중 오류 발생: {str(e)}")
            raise
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    @staticmethod
    async def get_market_data() -> List[Dict[str, Any]]:
        conn = None
        cur = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT * FROM market_data 
            ORDER BY is_index DESC, name;
            """
            
            cur.execute(query)
            result = cur.fetchall()
            
            return [
                {
                    "name": row['name'],
                    "symbol": row['symbol'],
                    "current": float(row['current_price']),
                    "change": float(row['price_change']),
                    "changePercent": float(row['change_percent']),
                    "flag": row['flag'],
                    "color": row['color'],
                    "data": row['chart_data']
                }
                for row in result
            ]
                
        except Exception as e:
            print(f"시장 데이터 처리 중 오류 발생: {str(e)}")
            raise
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close() 