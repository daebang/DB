# data/database_manager.py
import os
import pandas as pd
from sqlalchemy import create_engine, and_, desc, asc, inspect, or_, func, Integer
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from contextlib import contextmanager
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import json

# .models에서 Base와 각 모델 클래스를 import
# DatabaseManager가 data 폴더에 있고, models.py도 data 폴더에 있다면 from .models import ...
# 만약 models.py가 프로젝트 루트에 있다면 from models import ...
# 현재 구조상 data 폴더 내에 models.py가 있다고 가정
from .models import Base, StockPrice, EconomicEvent, TechnicalIndicator, NewsArticle 

logger = logging.getLogger(__name__)

class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, database_config=None, db_path: str = None, ensure_created: bool = True):
        self.database_config = database_config
        self.engine = None
        self.SessionLocal = None
        self.db_url = None # 사용할 최종 DB URL 저장용

        # 1. PostgreSQL 사용 조건 확인
        use_external_db = False
        if self.database_config and \
           hasattr(self.database_config, 'host') and \
           self.database_config.host and \
           isinstance(self.database_config.host, str) and \
           self.database_config.host.strip() != "":
            use_external_db = True
            self.db_url = f"postgresql+psycopg2://{self.database_config.username}:{self.database_config.password}@" \
                          f"{self.database_config.host}:{self.database_config.port}/{self.database_config.database}"
            logger.info(f"외부 데이터베이스 설정 감지: {self.db_url.split('@')[0]}...") # 민감 정보 제외하고 로그
        
        # 2. PostgreSQL 조건이 아니면 SQLite 사용
        if not use_external_db:
            if db_path: # 명시적 SQLite 경로가 있다면 사용
                self.db_path = db_path
            else: # 명시적 SQLite 경로가 없다면 기본 경로 사용
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_data_dir = os.path.join(os.path.dirname(current_dir), 'data_storage')
                if not os.path.exists(project_data_dir):
                    try:
                        os.makedirs(project_data_dir, exist_ok=True)
                    except OSError as e:
                        logger.error(f"DB 저장 디렉토리 생성 실패 '{project_data_dir}': {e}")
                        # 심각한 오류이므로, 여기서 프로그램을 중단하거나 대체 경로를 설정해야 함
                        raise # 또는 self.db_path = None 처리 후 _initialize_database에서 방어
                self.db_path = os.path.join(project_data_dir, 'trading_system.db')
            
            self.db_url = f"sqlite:///{self.db_path}"
            logger.info(f"SQLite 사용 설정됨. 경로: {self.db_path}")
        
        if ensure_created and self.db_url: # db_url이 설정되었을 때만 초기화 진행
            self._initialize_database_engine_and_tables()
        elif not self.db_url:
            logger.critical("DB URL이 설정되지 않아 DatabaseManager 초기화 불가.")
            # self.db_manager = None 처리와 유사하게, Controller에서 이 상태를 확인해야 함.

    def _initialize_database_engine_and_tables(self): # 메서드명 변경하여 역할 명확화
        """실제 데이터베이스 엔진 및 세션 초기화, 테이블 생성"""
        if not self.db_url:
            logger.error("DB URL이 없어 엔진을 생성할 수 없습니다.")
            raise ValueError("DB URL 미설정 오류")

        try:
            if "postgresql" in self.db_url:
                logger.info(f"PostgreSQL 엔진 생성 시도: {self.db_url.split('@')[0]}...")
                self.engine = create_engine(self.db_url, echo=False, pool_size=10, max_overflow=20, pool_timeout=30, pool_recycle=1800)
            elif "sqlite" in self.db_url:
                logger.info(f"SQLite 엔진 생성 시도: {self.db_url}")
                self.engine = create_engine(self.db_url, echo=False, connect_args={'check_same_thread': False})
            else:
                logger.error(f"지원되지 않는 DB URL 형식: {self.db_url}")
                raise ValueError("지원되지 않는 DB URL")
                
            self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
            self._create_tables()
            logger.info(f"데이터베이스 엔진 및 테이블 초기화 완료 ({'SQLite' if 'sqlite' in self.db_url else 'PostgreSQL'}).")
            
        except Exception as e:
            logger.error(f"데이터베이스 엔진/테이블 초기화 실패: {e}", exc_info=True)
            self.engine = None # 실패 시 엔진 None으로
            self.SessionLocal = None
            raise # 오류를 다시 발생시켜 호출부에서 인지하도록 함

    def _create_tables(self):
        """Base에 정의된 모든 테이블을 생성합니다."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("데이터베이스 테이블 생성/확인 완료.")
        except Exception as e:
            logger.error(f"테이블 생성 실패: {e}", exc_info=True)
            raise

    @contextmanager
    def get_session(self):
        """세션 컨텍스트 매니저. 사용 후 세션을 자동으로 닫습니다."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e: # 좀 더 구체적인 SQLAlchemy 예외 처리
            session.rollback()
            logger.error(f"데이터베이스 세션 오류 (SQLAlchemyError): {e}", exc_info=True)
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"데이터베이스 세션 중 일반 오류: {e}", exc_info=True)
            raise
        finally:
            session.close() # scoped_session 사용 시 remove()가 더 적절할 수 있으나, close()도 일반적으로 문제 없음
                           # self.SessionLocal.remove()는 스레드 종료 시 호출하는 것이 일반적

    def close(self):
        """데이터베이스 엔진 연결을 명시적으로 종료합니다 (애플리케이션 종료 시)."""
        if self.SessionLocal:
            self.SessionLocal.remove() # scoped_session의 경우 remove() 권장
            logger.info("DB 세션 풀 제거됨.")
        if self.engine:
            self.engine.dispose()
            logger.info("DB 엔진 연결 풀 해제됨.")

    def _dataframe_to_models(self, session, df: pd.DataFrame, model_class, 
                             unique_check_cols: List[str], 
                             column_mapping: Dict[str, str], 
                             update_existing: bool = True) -> Tuple[int, int]:
        """DataFrame을 SQLAlchemy 모델 객체 리스트로 변환하고, DB에 저장/업데이트합니다."""
        added_count = 0
        updated_count = 0
        objects_to_add = []
        
        for _, row in df.iterrows():
            # 고유성 체크를 위한 필터 조건 생성
            filters = []
            all_unique_cols_present = True
            for u_col_model, u_col_df in zip(unique_check_cols, [column_mapping.get(u, u) for u in unique_check_cols]):
                # column_mapping에 따라 DataFrame의 실제 컬럼명 사용
                df_col_name = column_mapping.get(u_col_model, u_col_model) 
                if df_col_name not in row or pd.isna(row[df_col_name]):
                    all_unique_cols_present = False
                    break
                # timestamp나 datetime은 Python datetime 객체로 변환
                value = row[df_col_name]
                if isinstance(getattr(model_class, u_col_model).type, datetime):
                    if isinstance(value, (int, float)): # Unix timestamp (초 단위 가정)
                        value = datetime.fromtimestamp(value)
                    elif not isinstance(value, datetime):
                        value = pd.to_datetime(value).to_pydatetime()
                filters.append(getattr(model_class, u_col_model) == value)
            
            if not all_unique_cols_present:
                logger.debug(f"고유키 컬럼 누락으로 건너뜀: {row.to_dict()}")
                continue

            existing_record = session.query(model_class).filter(and_(*filters)).first()

            model_data = {}
            valid_row = True
            for model_attr, df_col in column_mapping.items():
                if df_col in row and pd.notna(row[df_col]):
                    value = row[df_col]
                    # SQLAlchemy 모델의 컬럼 타입에 맞게 데이터 변환
                    col_type = getattr(model_class, model_attr).type
                    if isinstance(col_type, datetime):
                        if isinstance(value, (int, float)): # Unix timestamp
                             value = datetime.fromtimestamp(value)
                        elif not isinstance(value, datetime): # 문자열 등
                            value = pd.to_datetime(value, errors='coerce').to_pydatetime()
                        if pd.isna(value): # 변환 실패
                            logger.warning(f"'{df_col}'의 datetime 변환 실패: {row[df_col]}")
                            valid_row = False; break
                    elif isinstance(col_type, Integer):
                        value = int(value) if pd.notna(value) else None
                    elif isinstance(col_type, Float):
                        value = float(value) if pd.notna(value) else None
                    # 다른 타입에 대한 처리 추가 가능
                    model_data[model_attr] = value
                elif model_attr in unique_check_cols : #고유키 컬럼이면서 값이 없는 경우 (이미 위에서 체크했지만 방어적으로)
                    valid_row = False; break


            if not valid_row:
                logger.warning(f"필수 컬럼 또는 유효한 값 누락으로 건너뜀: {row.to_dict()}")
                continue


            if existing_record:
                if update_existing:
                    changed = False
                    for key, value in model_data.items():
                        if getattr(existing_record, key) != value:
                            setattr(existing_record, key, value)
                            changed = True
                    if changed:
                        if hasattr(existing_record, 'updated_at'):
                            existing_record.updated_at = datetime.now()
                        updated_count += 1
                # else: 건너뜀 (위에서 continue로 처리됨)
            else:
                objects_to_add.append(model_class(**model_data))
                added_count += 1
        
        if objects_to_add:
            session.bulk_save_objects(objects_to_add)
            
        return added_count, updated_count

    def save_stock_prices(self, symbol: str, timeframe: str, data_df: pd.DataFrame, 
                          update_existing: bool = True) -> Tuple[int, int]:
        if data_df is None or data_df.empty:
            logger.warning(f"저장할 주식 데이터가 없습니다: {symbol} ({timeframe})")
            return 0, 0

        # DataFrame에 symbol, timeframe 컬럼 추가 (DB 모델에 맞게)
        df_to_save = data_df.copy()
        df_to_save['symbol'] = symbol
        df_to_save['timeframe'] = timeframe
        
        # yfinance 컬럼명을 StockPrice 모델 컬럼명으로 매핑
        # timestamp는 인덱스 또는 'timestamp' 컬럼에서 가져옴
        column_mapping = {
            'symbol': 'symbol', 'timeframe': 'timeframe',
            'timestamp': df_to_save.index.name if df_to_save.index.name == 'timestamp' else 'timestamp', # 인덱스가 timestamp이거나, 컬럼에 timestamp
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'adj_close': 'Adj Close', 'volume': 'Volume'
        }
        # 만약 df_to_save에 이미 소문자 컬럼명('open', 'high' 등)이 있다면, 그에 맞게 수정
        # 예: column_mapping = {'open': 'open', ...}
        # Controller에서 이미 소문자로 변경했다면:
        column_mapping_lower = {
            'symbol': 'symbol', 'timeframe': 'timeframe',
            'timestamp': df_to_save.index.name if df_to_save.index.name == 'timestamp' else 'timestamp',
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
            'adj_close': 'adj_close', 'volume': 'volume'
        }
        # 실제 data_df의 컬럼명 확인 후 적절한 매핑 사용
        # 여기서는 Controller에서 rename했다고 가정하고 소문자 매핑 사용
        if 'Open' in df_to_save.columns: # 원본 yfinance 컬럼명 존재시
             active_column_mapping = column_mapping
        else: # 이미 소문자로 rename된 경우
             active_column_mapping = column_mapping_lower

        # StockPrice 모델의 복합 고유키
        unique_cols_stock_price = ['symbol', 'timeframe', 'timestamp']
        
        try:
            with self.get_session() as session:
                added, updated = self._dataframe_to_models(
                    session, df_to_save.reset_index(), StockPrice, 
                    unique_cols_stock_price, active_column_mapping, update_existing
                )
                logger.info(f"주식 데이터 저장/업데이트: {symbol} ({timeframe}) - 추가 {added}개, 업데이트 {updated}개")
                return added, updated
        except Exception as e:
            logger.error(f"주식 데이터 저장 실패: {symbol} ({timeframe}) - {e}", exc_info=True)
            return 0, 0

    def save_economic_events(self, events_df: pd.DataFrame, update_existing: bool = True) -> Tuple[int, int]:
        if events_df is None or events_df.empty:
            logger.warning("저장할 경제 이벤트 데이터가 없습니다")
            return 0, 0

        # EconomicEvent 모델의 컬럼명과 DataFrame 컬럼명 매핑
        column_mapping = {
            'datetime': 'datetime', 'country_code': 'country_code', 'importance': 'importance',
            'event_name': 'event', # DataFrame 컬럼명 'event' -> 모델 'event_name'
            'actual': 'actual', 'forecast': 'forecast', 'previous': 'previous',
            'actual_raw': 'actual_raw', 'forecast_raw': 'forecast_raw', 'previous_raw': 'previous_raw',
            'unit': 'unit', 'currency': 'currency', 'event_category': 'event_category',
            'event_link': 'event_link', 'source': 'source'
        }
        # EconomicEvent 모델의 복합 고유키
        unique_cols_economic_event = ['datetime', 'event_name', 'country_code']

        try:
            with self.get_session() as session:
                added, updated = self._dataframe_to_models(
                    session, events_df, EconomicEvent, 
                    unique_cols_economic_event, column_mapping, update_existing
                )
                logger.info(f"경제 이벤트 데이터 저장/업데이트 완료: 추가 {added}개, 업데이트 {updated}개")
                return added, updated
        except Exception as e:
            logger.error(f"경제 이벤트 데이터 저장 실패: {e}", exc_info=True)
            return 0, 0
            
    def save_technical_indicators(self, symbol: str, timeframe: str, timestamp: datetime, 
                                  indicators_data: Dict[str, float], update_existing: bool = True) -> int:
        """단일 타임스탬프에 대한 여러 기술적 지표들을 저장합니다."""
        saved_count = 0
        updated_count = 0
        try:
            with self.get_session() as session:
                for name, value in indicators_data.items():
                    if pd.isna(value): # NaN 값은 저장하지 않음 (선택적)
                        continue

                    existing = session.query(TechnicalIndicator).filter(
                        and_(
                            TechnicalIndicator.symbol == symbol,
                            TechnicalIndicator.timeframe == timeframe,
                            TechnicalIndicator.timestamp == timestamp,
                            TechnicalIndicator.indicator_name == name
                        )
                    ).first()

                    if existing:
                        if update_existing and existing.value != value:
                            existing.value = float(value)
                            updated_count +=1
                    else:
                        new_indicator = TechnicalIndicator(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=timestamp,
                            indicator_name=name,
                            value=float(value)
                        )
                        session.add(new_indicator)
                        saved_count += 1
                logger.info(f"기술적 지표 저장: {symbol} ({timeframe}, {timestamp}) - 추가 {saved_count}개, 업데이트 {updated_count}개")
                return saved_count + updated_count # 총 변경된 레코드 수
        except Exception as e:
            logger.error(f"기술적 지표 저장 실패: {symbol} ({timeframe}, {timestamp}) - {e}", exc_info=True)
            return 0

    def save_bulk_technical_indicators(self, df_with_ta: pd.DataFrame, symbol: str, timeframe: str, 
                                       indicator_cols: List[str], update_existing: bool = True) -> Tuple[int, int]:
        """DataFrame으로부터 여러 타임스탬프에 대한 기술적 지표들을 일괄 저장합니다."""
        if df_with_ta is None or df_with_ta.empty:
            logger.warning("저장할 기술적 지표 데이터가 없습니다.")
            return 0, 0
        
        added_count_total = 0
        updated_count_total = 0
        
        records_to_process = []
        for timestamp_idx, row in df_with_ta.iterrows():
            ts = timestamp_idx.to_pydatetime() if isinstance(timestamp_idx, pd.Timestamp) else pd.to_datetime(timestamp_idx).to_pydatetime()
            for col_name in indicator_cols:
                if col_name in row and pd.notna(row[col_name]):
                    records_to_process.append({
                        'symbol': symbol, 'timeframe': timeframe, 'timestamp': ts,
                        'indicator_name': col_name, 'value': float(row[col_name])
                    })
        
        try:
            with self.get_session() as session:
                for record_data in records_to_process:
                    existing = session.query(TechnicalIndicator).filter(
                        and_(
                            TechnicalIndicator.symbol == record_data['symbol'],
                            TechnicalIndicator.timeframe == record_data['timeframe'],
                            TechnicalIndicator.timestamp == record_data['timestamp'],
                            TechnicalIndicator.indicator_name == record_data['indicator_name']
                        )
                    ).first()

                    if existing:
                        if update_existing and existing.value != record_data['value']:
                            existing.value = record_data['value']
                            updated_count_total += 1
                    else:
                        new_indicator = TechnicalIndicator(**record_data)
                        session.add(new_indicator)
                        added_count_total += 1
                
                logger.info(f"기술적 지표 일괄 저장: {symbol} ({timeframe}) - 추가 {added_count_total}개, 업데이트 {updated_count_total}개")
                return added_count_total, updated_count_total
        except Exception as e:
            logger.error(f"기술적 지표 일괄 저장 실패: {symbol} ({timeframe}) - {e}", exc_info=True)
            return 0, 0

    def save_news_articles(self, news_list: List[Dict], update_existing: bool = True) -> Tuple[int, int]:
        """뉴스 기사 리스트를 DB에 저장합니다."""
        if not news_list:
            logger.info("저장할 뉴스 기사가 없습니다.")
            return 0, 0

        added_count = 0
        updated_count = 0
        
        try:
            with self.get_session() as session:
                for news_item in news_list:
                    url = news_item.get('url')
                    if not url:
                        logger.warning(f"URL이 없는 뉴스 항목 건너뜀: {news_item.get('title')}")
                        continue

                    existing_article = session.query(NewsArticle).filter(NewsArticle.url == url).first()
                    
                    published_at_str = news_item.get('publishedAt', news_item.get('published_at'))
                    if published_at_str:
                        try:
                            # 다양한 날짜 형식 시도 (ISO 8601, 일반적인 형식 등)
                            published_at_dt = pd.to_datetime(published_at_str, errors='coerce').to_pydatetime()
                            if pd.isna(published_at_dt): # 파싱 실패 시 현재 시간으로 대체 (또는 오류 처리)
                                logger.warning(f"뉴스 발행일 파싱 실패 '{published_at_str}', 현재 시간 사용.")
                                published_at_dt = datetime.now()
                        except Exception:
                            logger.warning(f"뉴스 발행일 파싱 중 예외 '{published_at_str}', 현재 시간 사용.")
                            published_at_dt = datetime.now()
                    else:
                        published_at_dt = datetime.now()


                    article_data = {
                        'source_name': news_item.get('source', {}).get('name') if isinstance(news_item.get('source'), dict) else news_item.get('source_name'),
                        'title': news_item.get('title'),
                        'url': url,
                        'published_at': published_at_dt,
                        'content_snippet': news_item.get('description', news_item.get('content_snippet')),
                        'sentiment_score': self._parse_numeric_value(news_item.get('sentiment_score')),
                        'sentiment_label': news_item.get('sentiment_label'),
                        'related_symbols': json.dumps(news_item.get('related_symbols')) if news_item.get('related_symbols') else None
                    }
                    # 필수 값 누락 시 건너뜀
                    if not article_data['title']: 
                        logger.warning(f"제목이 없는 뉴스 항목 건너뜀: {url}")
                        continue


                    if existing_article:
                        if update_existing:
                            changed = False
                            for key, value in article_data.items():
                                if getattr(existing_article, key) != value:
                                    setattr(existing_article, key, value)
                                    changed = True
                            if changed:
                                existing_article.updated_at = datetime.now()
                                updated_count += 1
                    else:
                        new_article = NewsArticle(**{k:v for k,v in article_data.items() if v is not None})
                        session.add(new_article)
                        added_count += 1
                
                logger.info(f"뉴스 기사 저장/업데이트 완료: 추가 {added_count}개, 업데이트 {updated_count}개")
                return added_count, updated_count
        except Exception as e:
            logger.error(f"뉴스 기사 저장 실패: {e}", exc_info=True)
            return 0, 0

    # --- 데이터 조회 메서드 (기존 StockPrice, EconomicEvent + 추가) ---
    def get_stock_prices(self, symbol: str, timeframe: str, 
                         start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None,
                         limit: Optional[int] = None, 
                         order_desc: bool = True) -> pd.DataFrame:
        try:
            with self.get_session() as session:
                query = session.query(StockPrice).filter(
                    and_(StockPrice.symbol == symbol, StockPrice.timeframe == timeframe)
                )
                if start_date: query = query.filter(StockPrice.timestamp >= start_date)
                if end_date: query = query.filter(StockPrice.timestamp <= end_date)
                
                query = query.order_by(desc(StockPrice.timestamp) if order_desc else asc(StockPrice.timestamp))
                
                if limit: query = query.limit(limit)
                
                results_orm = query.all()
                if not results_orm: return pd.DataFrame()
                
                # ORM 객체 리스트를 DataFrame으로 변환
                # to_dict 메서드가 모델에 있다면 더 효율적
                results_dict = [
                    {"timestamp": r.timestamp, "open": r.open, "high": r.high, "low": r.low, 
                     "close": r.close, "adj_close": r.adj_close, "volume": r.volume,
                     "symbol": r.symbol, "timeframe": r.timeframe} for r in results_orm
                ]
                df = pd.DataFrame(results_dict)
                if not df.empty:
                     df = df.sort_values('timestamp', ascending=not order_desc).set_index('timestamp')
                logger.info(f"주식 데이터 조회: {symbol} ({timeframe}) - {len(df)}개 레코드")
                return df
        except Exception as e:
            logger.error(f"주식 데이터 조회 실패: {symbol} ({timeframe}) - {e}", exc_info=True)
            return pd.DataFrame()

    def get_technical_indicators(self, symbol: str, timeframe: str, indicator_name: Optional[str] = None,
                                 start_date: Optional[datetime] = None, 
                                 end_date: Optional[datetime] = None,
                                 limit: Optional[int] = None,
                                 pivot: bool = True) -> pd.DataFrame:
        """기술적 지표 조회. pivot=True이면 지표명을 컬럼으로 하는 DataFrame 반환."""
        try:
            with self.get_session() as session:
                query = session.query(TechnicalIndicator).filter(
                    and_(TechnicalIndicator.symbol == symbol, TechnicalIndicator.timeframe == timeframe)
                )
                if indicator_name: query = query.filter(TechnicalIndicator.indicator_name == indicator_name)
                if start_date: query = query.filter(TechnicalIndicator.timestamp >= start_date)
                if end_date: query = query.filter(TechnicalIndicator.timestamp <= end_date)
                
                query = query.order_by(desc(TechnicalIndicator.timestamp)) # 최신순
                if limit and not pivot : query = query.limit(limit) # 피봇 시에는 모든 데이터를 가져와서 처리

                results_orm = query.all()
                if not results_orm: return pd.DataFrame()
                
                results_dict = [
                    {"timestamp": r.timestamp, "indicator_name": r.indicator_name, "value": r.value,
                     "symbol": r.symbol, "timeframe": r.timeframe} for r in results_orm
                ]
                df = pd.DataFrame(results_dict)
                
                if pivot and not df.empty:
                    df = df.pivot_table(index='timestamp', columns='indicator_name', values='value')
                    df = df.sort_index(ascending=True) # 시간순 정렬
                    if limit: df = df.tail(limit) # 피봇 후 limit 적용
                elif not df.empty:
                     df = df.sort_values('timestamp', ascending=False) # 최신순 정렬 유지 (피봇 안 할 시)
                
                logger.info(f"기술적 지표 조회: {symbol} ({timeframe}) - {len(df)}개 레코드 (피봇: {pivot})")
                return df
        except Exception as e:
            logger.error(f"기술적 지표 조회 실패: {symbol} ({timeframe}) - {e}", exc_info=True)
            return pd.DataFrame()
            
    def get_economic_events(self, start_date: Optional[datetime] = None, #
                            end_date: Optional[datetime] = None,
                            country_codes: Optional[List[str]] = None,
                            importance_min: Optional[int] = None, # Optional로 변경
                            event_names: Optional[List[str]] = None, # 이벤트명 리스트로 검색
                            limit: Optional[int] = None) -> pd.DataFrame:
        try:
            with self.get_session() as session:
                query = session.query(EconomicEvent)
                if start_date: query = query.filter(EconomicEvent.datetime >= start_date)
                if end_date: query = query.filter(EconomicEvent.datetime <= end_date)
                if country_codes: query = query.filter(EconomicEvent.country_code.in_(country_codes))
                if importance_min is not None: query = query.filter(EconomicEvent.importance >= importance_min)
                if event_names: query = query.filter(EconomicEvent.event_name.in_(event_names))
                
                query = query.order_by(asc(EconomicEvent.datetime)) # 시간순 정렬
                if limit: query = query.limit(limit)
                
                results_orm = query.all()
                if not results_orm: return pd.DataFrame()
                
                results_dict = [
                    {'datetime': r.datetime, 'country_code': r.country_code, 'event': r.event_name, # 컬럼명 일관성 위해 'event' 사용
                     'category': r.event_category, 'importance': r.importance, 
                     'actual': r.actual, 'forecast': r.forecast, 'previous': r.previous,
                     'actual_raw': r.actual_raw, 'forecast_raw': r.forecast_raw, 'previous_raw': r.previous_raw,
                     'event_link': r.event_link, 'currency': r.currency, 'unit': r.unit, 'source': r.source
                    } for r in results_orm
                ]
                df = pd.DataFrame(results_dict)
                logger.info(f"경제 이벤트 조회 완료: {len(df)}개 레코드")
                return df
        except Exception as e:
            logger.error(f"경제 이벤트 조회 실패: {e}", exc_info=True)
            return pd.DataFrame()

    def get_news_articles(self, symbol: Optional[str] = None, keywords: Optional[List[str]] = None,
                          start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                          limit: Optional[int] = None, source_name: Optional[str] = None) -> pd.DataFrame:
        try:
            with self.get_session() as session:
                query = session.query(NewsArticle)
                if symbol: # 특정 심볼 관련 뉴스 (related_symbols 필드 활용 필요)
                    query = query.filter(NewsArticle.related_symbols.like(f'%"{symbol}"%')) # JSON 문자열 내 검색
                if keywords: # 키워드 검색 (title 또는 content_snippet)
                    keyword_filters = [or_(NewsArticle.title.ilike(f"%{kw}%"), NewsArticle.content_snippet.ilike(f"%{kw}%")) for kw in keywords]
                    query = query.filter(or_(*keyword_filters))
                if start_date: query = query.filter(NewsArticle.published_at >= start_date)
                if end_date: query = query.filter(NewsArticle.published_at <= end_date)
                if source_name: query = query.filter(NewsArticle.source_name == source_name)
                
                query = query.order_by(desc(NewsArticle.published_at)) # 최신순
                if limit: query = query.limit(limit)
                
                results_orm = query.all()
                if not results_orm: return pd.DataFrame()

                results_dict = [
                    {'id':r.id, 'source_name': r.source_name, 'title': r.title, 'url': r.url, 
                     'published_at': r.published_at, 'content_snippet': r.content_snippet,
                     'sentiment_score': r.sentiment_score, 'sentiment_label': r.sentiment_label,
                     'related_symbols': json.loads(r.related_symbols) if r.related_symbols else None
                    } for r in results_orm
                ]
                df = pd.DataFrame(results_dict)
                logger.info(f"뉴스 기사 조회: {len(df)}개 레코드")
                return df
        except Exception as e:
            logger.error(f"뉴스 기사 조회 실패: {e}", exc_info=True)
            return pd.DataFrame()


    def get_last_timestamp(self, model_class, **filters) -> Optional[datetime]:
        """특정 모델과 필터 조건에 맞는 가장 최근의 timestamp를 반환합니다."""
        try:
            with self.get_session() as session:
                query = session.query(func.max(model_class.timestamp))
                for col_name, value in filters.items():
                    if hasattr(model_class, col_name):
                        query = query.filter(getattr(model_class, col_name) == value)
                last_ts = query.scalar()
                return last_ts
        except Exception as e:
            logger.error(f"{model_class.__name__}의 마지막 타임스탬프 조회 실패: {filters} - {e}", exc_info=True)
            return None
            
    def _parse_numeric_value(self, value) -> Optional[float]:
        """텍스트를 숫자로 변환 (예: '1.5M' -> 1500000.0)"""
        if pd.isna(value) or value is None:
            return None
        
        try:
            # 이미 숫자인 경우
            if isinstance(value, (int, float)):
                return float(value)
            
            # 문자열 처리
            value_str = str(value).strip().replace(',', '')
            
            # 단위 처리
            multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000, 'T': 1000000000000}
            
            for unit, multiplier in multipliers.items():
                if value_str.upper().endswith(unit):
                    number_part = value_str[:-1]
                    return float(number_part) * multiplier
            
            # 일반 숫자 변환
            return float(value_str)
            
        except (ValueError, TypeError):
            return None
    
    # === 유틸리티 메서드 ===
    
    def get_database_stats(self) -> Dict[str, int]:
        """데이터베이스 통계 정보 조회"""
        try:
            with self.get_session() as session:
                stats = {}
                
                # 각 테이블별 레코드 수
                stats['stock_prices'] = session.query(StockPrice).count()
                stats['economic_events'] = session.query(EconomicEvent).count()
                stats['technical_indicators'] = session.query(TechnicalIndicator).count()
                stats['news_articles'] = session.query(NewsArticle).count()
                
                # 주식 데이터 심볼별 통계
                stock_symbols = session.query(StockPrice.symbol).distinct().all()
                stats['unique_stock_symbols'] = len(stock_symbols)
                
                # 경제 이벤트 국가별 통계
                countries = session.query(EconomicEvent.country_code).distinct().all()
                stats['unique_countries'] = len(countries)
                
                logger.info(f"데이터베이스 통계: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"데이터베이스 통계 조회 실패: {e}", exc_info=True)
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """오래된 데이터 정리"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            with self.get_session() as session:
                # 오래된 주식 데이터 삭제 (분봉 데이터는 더 짧게)
                deleted_stocks = session.query(StockPrice).filter(
                    and_(
                        StockPrice.timestamp < cutoff_date,
                        StockPrice.timeframe.in_(['1분', '5분', '15분', '30분'])
                    )
                ).delete()
                
                # 오래된 뉴스 데이터 삭제
                news_cutoff = datetime.now() - timedelta(days=90)  # 뉴스는 3개월만 보관
                deleted_news = session.query(NewsArticle).filter(
                    NewsArticle.published_at < news_cutoff
                ).delete()
                
                logger.info(f"데이터 정리 완료: 주식 {deleted_stocks}개, 뉴스 {deleted_news}개 삭제")
                
        except Exception as e:
            logger.error(f"데이터 정리 실패: {e}", exc_info=True)
    
    def vacuum_database(self):
        """데이터베이스 최적화 (SQLite 전용)"""
        if 'sqlite' in str(self.engine.url):
            try:
                with self.engine.connect() as conn:
                    conn.execute("VACUUM")
                logger.info("SQLite 데이터베이스 최적화 완료")
            except Exception as e:
                logger.error(f"데이터베이스 최적화 실패: {e}", exc_info=True)


if __name__ == '__main__':
    # 테스트 코드
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 테스트용 DatabaseManager 생성
    db_manager = DatabaseManager(db_path='test_trading_system.db')
    
    try:
        # 테스트 주식 데이터 생성
        import numpy as np
        
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        test_stock_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, len(dates)),
            'high': np.random.uniform(110, 120, len(dates)),
            'low': np.random.uniform(90, 100, len(dates)),
            'close': np.random.uniform(100, 110, len(dates)),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        })
        
        # 주식 데이터 저장 테스트
        saved_count = db_manager.save_stock_prices('AAPL', '1일', test_stock_data)
        print(f"저장된 주식 데이터: {saved_count}개")
        
        # 주식 데이터 조회 테스트
        retrieved_data = db_manager.get_stock_prices('AAPL', '1일', limit=5)
        print(f"조회된 주식 데이터: {len(retrieved_data)}개")
        print(retrieved_data.head())
        
        # 테스트 경제 이벤트 데이터 생성
        test_economic_data = pd.DataFrame({
            'datetime': [datetime(2023, 1, 15, 14, 30), datetime(2023, 1, 20, 10, 0)],
            'country_code': ['US', 'US'],
            'event': ['Non-Farm Payrolls', 'GDP Growth Rate'],
            'importance': [3, 3],
            'actual': [250000, 2.1],
            'forecast': [200000, 2.0],
            'previous': [180000, 1.8],
            'actual_raw': ['250K', '2.1%'],
            'forecast_raw': ['200K', '2.0%'],
            'previous_raw': ['180K', '1.8%']
        })
        
        # 경제 이벤트 데이터 저장 테스트
        saved_events = db_manager.save_economic_events(test_economic_data)
        print(f"저장된 경제 이벤트: {saved_events}개")
        
        # 경제 이벤트 조회 테스트
        retrieved_events = db_manager.get_economic_events(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        print(f"조회된 경제 이벤트: {len(retrieved_events)}개")
        print(retrieved_events.head())
        
        # 데이터베이스 통계
        stats = db_manager.get_database_stats()
        print(f"데이터베이스 통계: {stats}")
        
    except Exception as e:
        print(f"테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db_manager.close()
        print("테스트 완료")
