# data/models.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, UniqueConstraint, Index
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class StockPrice(Base):
    """주식 가격 데이터 모델 (OHLCV)"""
    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    timeframe = Column(String, nullable=False, index=True) # 예: '1일', '1시간', '5분'
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    adj_close = Column(Float, nullable=True) # 수정 종가
    volume = Column(Integer, nullable=True) # API에 따라 Float일 수도 있음, 여기서는 Integer로 가정

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # 복합 고유 제약 조건: 동일 심볼, 동일 시간대, 동일 타임스탬프 데이터는 유일해야 함
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_stock_price_symbol_timeframe_timestamp'),
        Index('ix_stock_prices_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )

    def __repr__(self):
        return f"<StockPrice(symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}', close='{self.close}')>"

class TechnicalIndicator(Base):
    """계산된 기술적 지표 데이터 모델"""
    __tablename__ = "technical_indicators"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    stock_price_id = Column(Integer, ForeignKey('stock_prices.id'), nullable=True) # 특정 StockPrice 레코드와 연결 (선택적)
    symbol = Column(String, nullable=False, index=True)
    timeframe = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True) # 해당 지표가 계산된 시점의 가격 데이터 timestamp

    indicator_name = Column(String, nullable=False, index=True) # 예: 'SMA_20', 'RSI_14'
    value = Column(Float, nullable=True) # 지표 값
    # 추가적인 지표 관련 값 (예: MACD의 경우 signal, histogram 등)은 json이나 개별 컬럼으로 확장 가능
    # 여기서는 단순화를 위해 단일 value로 저장하고, 필요시 indicator_name에 세부사항 포함 (예: 'MACD_Signal')

    created_at = Column(DateTime, default=func.now())

    # 복합 고유 제약 조건: 동일 심볼, 시간대, 타임스탬프, 지표명 데이터는 유일해야 함
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp', 'indicator_name', name='uq_indicator_symbol_timeframe_timestamp_name'),
        Index('ix_technical_indicators_symbol_timeframe_timestamp_name', 'symbol', 'timeframe', 'timestamp', 'indicator_name'),
    )

    def __repr__(self):
        return f"<TechnicalIndicator(symbol='{self.symbol}', timestamp='{self.timestamp}', name='{self.indicator_name}', value='{self.value}')>"


class EconomicEvent(Base):
    """경제 캘린더 이벤트 데이터 모델"""
    __tablename__ = "economic_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    datetime = Column(DateTime, nullable=False, index=True) # 이벤트 발표 시간 (UTC 권장)
    country_code = Column(String(10), index=True, nullable=True) # 국가 코드 (예: 'US', 'KR')
    importance = Column(Integer, index=True, nullable=True) # 중요도 (예: 1, 2, 3)
    event_name = Column(String, nullable=False, index=True)
    
    actual = Column(Float, nullable=True)
    forecast = Column(Float, nullable=True)
    previous = Column(Float, nullable=True)
    
    actual_raw = Column(String, nullable=True) # 원본 문자열 (예: "2.5M", "0.5%")
    forecast_raw = Column(String, nullable=True)
    previous_raw = Column(String, nullable=True)

    unit = Column(String, nullable=True) # 값의 단위 (예: '%', 'K', 'M', 'B', 통화 코드)
    currency = Column(String, nullable=True) # 관련 통화 (해당되는 경우)
    event_category = Column(String, nullable=True, index=True) # 이벤트 카테고리 (예: '고용', '인플레이션')
    event_link = Column(String, nullable=True) # 상세 정보 링크
    source = Column(String, default="investing.com") # 데이터 출처

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # 복합 고유 제약 조건: 동일 날짜, 이벤트명, 국가 코드는 유일해야 함 (동일 이벤트가 여러번 발표되진 않으므로)
    __table_args__ = (
        UniqueConstraint('datetime', 'event_name', 'country_code', name='uq_economic_event_datetime_name_country'),
        Index('ix_economic_events_datetime_country_event', 'datetime', 'country_code', 'event_name'),
    )

    def __repr__(self):
        return f"<EconomicEvent(datetime='{self.datetime}', country='{self.country_code}', event='{self.event_name}', actual='{self.actual_raw}')>"


class NewsArticle(Base):
    """뉴스 기사 데이터 모델"""
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    source_name = Column(String, nullable=True, index=True) # 뉴스 출처 (예: 'Investing.com', 'Yonhap')
    title = Column(String, nullable=False)
    url = Column(String, nullable=False, unique=True, index=True) # URL을 고유키로 사용
    published_at = Column(DateTime, nullable=False, index=True) # 발행일시 (UTC 권장)
    content_snippet = Column(Text, nullable=True) # 기사 요약 또는 일부 내용
    
    # 감성 분석 결과
    sentiment_score = Column(Float, nullable=True, index=True) # 예: -1.0 ~ 1.0
    sentiment_label = Column(String, nullable=True, index=True) # 예: 'positive', 'negative', 'neutral'
    
    # 관련 심볼/키워드 (JSON 문자열 또는 별도 테이블로 관리 가능)
    related_symbols = Column(Text, nullable=True) # 예: 'AAPL,MSFT' 또는 JSON 문자열 '["AAPL", "MSFT"]'

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<NewsArticle(title='{self.title[:50]}...', published_at='{self.published_at}')>"

# 필요한 경우 추가 모델 (예: MLPrediction, TradingSignal 등) 정의

# 모든 모델 클래스 리스트 (테이블 생성 시 사용될 수 있으나, Base.metadata.create_all로 충분)
ALL_MODELS = [StockPrice, EconomicEvent, TechnicalIndicator, NewsArticle]