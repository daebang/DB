import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

class DatabaseManager:
    """데이터베이스 관리자"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 주식 데이터 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stock_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        adj_close REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                """)
                
                # 뉴스 데이터 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS news_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        title TEXT NOT NULL,
                        content TEXT,
                        source TEXT,
                        url TEXT,
                        published_at TIMESTAMP,
                        sentiment_score REAL,
                        sentiment_label TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 거래 신호 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        price REAL NOT NULL,
                        confidence REAL NOT NULL,
                        strategy_name TEXT NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 주문 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        order_type TEXT NOT NULL,
                        price REAL,
                        status TEXT NOT NULL,
                        filled_quantity INTEGER DEFAULT 0,
                        filled_price REAL,
                        strategy_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        filled_at TIMESTAMP
                    )
                """)
                
                # 포트폴리오 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        avg_price REAL NOT NULL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol)
                    )
                """)
                
                # 인덱스 생성
                conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_symbol_date ON stock_data(symbol, date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol_date ON news_data(symbol, published_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_date ON trading_signals(symbol, created_at)")
                
                conn.commit()
                self.logger.info("데이터베이스 초기화 완료")
                
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 오류: {e}")
    
    def save_stock_data(self, symbol: str, data: pd.DataFrame):
        """주식 데이터 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 데이터 준비
                data_to_save = data.copy()
                data_to_save['symbol'] = symbol
                data_to_save = data_to_save.reset_index()  # date 컬럼을 인덱스에서 일반 컬럼으로
                
                # 데이터베이스에 저장 (중복 무시)
                data_to_save.to_sql('stock_data', conn, if_exists='append', index=False, method='ignore')
                
                self.logger.info(f"{symbol} 주식 데이터 {len(data_to_save)}개 저장")
                
        except Exception as e:
            self.logger.error(f"주식 데이터 저장 오류: {e}")
    
    def load_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """주식 데이터 로드"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM stock_data WHERE symbol = ?"
                params = [symbol]
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY date"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                return df
                
        except Exception as e:
            self.logger.error(f"주식 데이터 로드 오류: {e}")
            return pd.DataFrame()
    
    def save_news_data(self, news_list: List[Dict]):
        """뉴스 데이터 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for news in news_list:
                    conn.execute("""
                        INSERT OR IGNORE INTO news_data 
                        (symbol, title, content, source, url, published_at, sentiment_score, sentiment_label)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        news.get('symbol'),
                        news.get('title'),
                        news.get('content'),
                        news.get('source'),
                        news.get('url'),
                        news.get('published_at'),
                        news.get('sentiment_score'),
                        news.get('sentiment_label')
                    ))
                
                conn.commit()
                self.logger.info(f"뉴스 데이터 {len(news_list)}개 저장")
                
        except Exception as e:
            self.logger.error(f"뉴스 데이터 저장 오류: {e}")
    
    def save_trading_signal(self, signal):
        """거래 신호 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trading_signals 
                    (symbol, action, price, confidence, strategy_name, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    signal.symbol,
                    signal.action,
                    signal.price,
                    signal.confidence,
                    signal.strategy_name,
                    str(signal.metadata) if signal.metadata else None
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"거래 신호 저장 오류: {e}")
    
    def save_order(self, order):
        """주문 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO orders 
                    (id, symbol, action, quantity, order_type, price, status, 
                     filled_quantity, filled_price, strategy_name, filled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order.id,
                    order.symbol,
                    order.action,
                    order.quantity,
                    order.order_type.value,
                    order.price,
                    order.status.value,
                    order.filled_quantity,
                    order.filled_price,
                    order.strategy_name,
                    order.filled_time
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"주문 저장 오류: {e}")
    
    def update_portfolio(self, symbol: str, quantity: int, avg_price: float):
        """포트폴리오 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO portfolio (symbol, quantity, avg_price)
                    VALUES (?, ?, ?)
                """, (symbol, quantity, avg_price))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"포트폴리오 업데이트 오류: {e}")
    
    def get_portfolio(self) -> pd.DataFrame:
        """포트폴리오 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM portfolio WHERE quantity > 0", conn)
                return df
                
        except Exception as e:
            self.logger.error(f"포트폴리오 조회 오류: {e}")
            return pd.DataFrame()