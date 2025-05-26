# core/controller.py
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta # timedelta 추가
from threading import Thread, Lock
import queue
import pandas as pd
import numpy as np # 추가
from PyQt5.QtCore import QObject, pyqtSignal

from utils.scheduler import Scheduler
from core.workflow_engine import WorkflowEngine, Task
from core.event_manager import EventManager
from data.collectors.stock_data_collector import StockDataCollector
from data.collectors.news_collector import NewsCollector
from data.collectors.economic_calendar_collector import EconomicCalendarCollector
from data.database_manager import DatabaseManager
# from data.models import Base # Base는 DatabaseManager 내부에서만 사용될 수 있음
from config.settings import Settings
from analysis.technical.indicators import add_all_selected_indicators
# 임시 ModelManager (이전과 동일)
class TempModelManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TempModelManager")

    def predict(self, symbol: str, data: Optional[pd.DataFrame]) -> Optional[Dict]:
        self.logger.debug(f"임시 모델 매니저: {symbol} 예측 요청 (데이터 {len(data) if data is not None else 0}개)")
        if data is not None and not data.empty:
            price_col_to_check = 'Close' if 'Close' in data.columns else 'close' # 대소문자 컬럼명 고려
            if price_col_to_check in data.columns and len(data[price_col_to_check]) >= 5:
                if data[price_col_to_check].iloc[-1] > data[price_col_to_check].iloc[-5:].mean():
                    return {"action": "BUY", "confidence": 0.6, "reason": "최근 상승세 (임시)", "target_price": data[price_col_to_check].iloc[-1] * 1.05}
                else:
                    return {"action": "SELL", "confidence": 0.6, "reason": "최근 하락세 또는 횡보 (임시)", "target_price": data[price_col_to_check].iloc[-1] * 0.95}
        self.logger.warning(f"임시 모델 매니저: {symbol} 예측 위한 데이터 부족")
        return None

    def train_model_async(self, symbol: str, data: Optional[pd.DataFrame]):
        self.logger.info(f"임시 모델 매니저: {symbol} 모델 학습 비동기 요청 (데이터 {len(data) if data is not None else 0}개)")
        pass

class MainController(QObject):
    # --- Signals ---
    historical_data_updated_signal = pyqtSignal(str, object) # 차트용 원본 데이터 (symbol, df)
    technical_indicators_updated_signal = pyqtSignal(str, str, object) # 기술적 지표 포함 데이터 (symbol, timeframe, df_with_ta)
    realtime_quote_updated_signal = pyqtSignal(str, dict)
    news_data_updated_signal = pyqtSignal(str, list)
    analysis_result_updated_signal = pyqtSignal(str, dict)
    economic_data_updated_signal = pyqtSignal(object) # 경제 캘린더 데이터용 시그널 추가 (object는 pd.DataFrame)
    connection_status_changed_signal = pyqtSignal(str, bool)
    status_message_signal = pyqtSignal(str, int) # UI 상태바 메시지용 (message, timeout)
    task_feedback_signal = pyqtSignal(str, str) # UI 작업 피드백용 (task_name, status)

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        self.event_manager = EventManager()
        self.workflow_engine = WorkflowEngine(self.event_manager, num_workers=5)
        self.scheduler = Scheduler()

        try:
            self.db_manager = DatabaseManager(database_config=settings.db_config)
            self.logger.info("데이터베이스 매니저 초기화 완료")
        except Exception as e:
            self.logger.critical(f"데이터베이스 매니저 초기화 실패: {e}", exc_info=True)
            self.db_manager = None
        self.stock_collector = StockDataCollector(settings.api_config) # yfinance 기반 Collector
        self.news_collector = NewsCollector(settings.api_config)
        self.economic_calendar_collector = EconomicCalendarCollector() # api_config는 현재 불필요
        self.model_manager = TempModelManager()

        self._data_cache: Dict[str, Any] = {}
        self._cache_lock = Lock()
        self._active_data_requests: set[str] = set() # 모든 데이터 요청 키 관리 (중복 방지용)

        self._register_event_handlers()
        self._setup_scheduled_tasks()
        self.logger.info("메인 컨트롤러 초기화 완료. DB 연동 및 캐싱 로직 포함.")

    def _register_event_handlers(self):
        self.event_manager.subscribe('task_started', self._on_task_started)
        self.event_manager.subscribe('task_completed', self._on_task_completed)
        self.event_manager.subscribe('task_failed', self._on_task_failed)
        self.logger.debug("컨트롤러 이벤트 핸들러 등록 완료.")

    def _setup_scheduled_tasks(self):
        if self.scheduler: # 스케줄러 객체가 정상적으로 생성되었는지 확인
            try:
                self.scheduler.add_job(
                    # request_economic_calendar_update에 필요한 인자를 lambda로 전달
                    lambda: self.request_economic_calendar_update(days_future=7, importance_min=1, force_fetch_from_api=True),
                    'cron',
                    hour=7, minute=0, # 매일 아침 7시
                    id="daily_economic_calendar_update",
                    replace_existing=True 
                )
                self.logger.info("경제 캘린더 일일 업데이트 작업 스케줄됨 (매일 07:00).")
            except Exception as e:
                self.logger.error(f"스케줄된 작업 설정 중 오류: {e}", exc_info=True)
        else:
            self.logger.warning("스케줄러 객체가 초기화되지 않아 스케줄 작업을 설정할 수 없습니다.")

    def start(self):
        self.logger.info("컨트롤러 시작: 워크플로우 엔진 및 스케줄러 가동 시도.")
        if self.workflow_engine: self.workflow_engine.start()
        if self.scheduler and not self.scheduler.scheduler.running:
            try:
                self.scheduler.start()
                self.logger.info("스케줄러 시작됨.")
            except Exception as e:
                self.logger.warning(f"스케줄러 시작 중 문제 발생: {e}")
        self.connection_status_changed_signal.emit("시스템 준비됨 (컨트롤러 시작)", True)

    def stop(self):
        self.logger.info("컨트롤러 중지 요청: 워크플로우 엔진, 스케줄러, DB 매니저 종료 시도.")
        if self.workflow_engine: self.workflow_engine.stop()
        if self.scheduler and self.scheduler.scheduler.running:
            self.scheduler.shutdown()
            self.logger.info("스케줄러 종료됨.")
        if self.db_manager:
            self.db_manager.close()
            self.logger.info("데이터베이스 연결 종료됨.")
        self.logger.info("워크플로우 엔진 종료됨.")
        self.connection_status_changed_signal.emit("시스템 중지됨", False)

    def _is_task_active(self, request_key: str) -> bool:
        """지정된 요청 키가 이미 활성 상태인지 확인합니다."""
        if request_key in self._active_data_requests:
            task_name_for_message = request_key.replace("_", " ").title() # 보기 좋은 이름으로 변환
            self.status_message_signal.emit(f"{task_name_for_message}는 이미 요청 중입니다.", 3000)
            return True
        return False

    def _add_active_task(self, request_key: str):
        self._active_data_requests.add(request_key)

    def _remove_active_task(self, request_key: str):
        if request_key in self._active_data_requests:
            self._active_data_requests.remove(request_key)

    @staticmethod
    def _get_ohlcv_mapping(to_lower: bool = True):
        """yfinance 컬럼명과 내부 소문자 컬럼명 간의 매핑 반환"""
        mapping = {'Open': 'open', 'High': 'high', 'Low': 'low', 
                   'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}
        if to_lower:
            return mapping
        return {v: k for k, v in mapping.items()} # 소문자 -> 대문자

    def _standardize_df_columns_and_index(self, df: pd.DataFrame, symbol:str, timeframe:str) -> pd.DataFrame:
        """DataFrame의 인덱스를 'timestamp' 컬럼으로, 컬럼명을 소문자로 표준화합니다."""
        df_std = df.copy()
        if df_std.index.name in ['Date', 'Datetime']: # yfinance 인덱스명 처리
            df_std.index.name = 'timestamp'
        if 'timestamp' not in df_std.columns and df_std.index.name == 'timestamp':
            df_std = df_std.reset_index()
        
        df_std.rename(columns=self._get_ohlcv_mapping(to_lower=True), inplace=True)
        
        # 'timestamp' 컬럼이 datetime 객체인지 확인 및 변환
        if 'timestamp' in df_std.columns and not pd.api.types.is_datetime64_any_dtype(df_std['timestamp']):
            df_std['timestamp'] = pd.to_datetime(df_std['timestamp'], errors='coerce')
            if df_std['timestamp'].isnull().any():
                self.logger.warning(f"{symbol} ({timeframe}): 일부 timestamp 변환 실패.")
                df_std.dropna(subset=['timestamp'], inplace=True)
        return df_std

    def _process_and_emit_historical_data(self, symbol: str, timeframe: str, df_original_yf: pd.DataFrame, task_display_name: str):
        """주어진 원본 주가 데이터로 기술적 지표 계산, 캐시 및 UI 업데이트, 분석 요청을 수행합니다."""
        if df_original_yf is None or df_original_yf.empty:
            self.historical_data_updated_signal.emit(symbol, pd.DataFrame())
            self.technical_indicators_updated_signal.emit(symbol, timeframe, pd.DataFrame())
            return

        self.task_feedback_signal.emit(task_display_name, "기술적 지표 계산 중...")
        df_for_ta = self._standardize_df_columns_and_index(df_original_yf, symbol, timeframe) # 표준화된 df (timestamp 컬럼, 소문자 ohlcv)
        price_col_for_ta = 'adj_close' if 'adj_close' in df_for_ta.columns else 'close'
        
        required_cols_for_ta = ['open', 'high', 'low', price_col_for_ta]
        if not all(col in df_for_ta.columns for col in required_cols_for_ta):
            self.logger.error(f"{task_display_name}: TA용 컬럼 누락: {df_for_ta.columns.tolist()}")
            data_with_indicators = df_for_ta.copy() 
        else:
            data_with_indicators = add_all_selected_indicators(df_for_ta.copy(), price_col=price_col_for_ta)
        
        self.logger.info(f"{task_display_name}: 기술적 지표 계산 완료.")
        
        if self.db_manager: # 기술적 지표도 DB에 저장
            indicator_cols = [col for col in data_with_indicators.columns if any(kw in col for kw in ['SMA_', 'EMA_', 'RSI_', 'MACD', 'BB_'])]
            if indicator_cols and 'timestamp' in data_with_indicators.columns:
                self.db_manager.save_bulk_technical_indicators(
                    data_with_indicators, symbol, timeframe, indicator_cols, update_existing=True
                ) # data_with_indicators는 timestamp 컬럼을 가지고 있어야 함

        with self._cache_lock: # 캐시에는 yfinance 원본과 TA 포함본 둘 다 저장
            self._data_cache[f"{symbol}_{timeframe}_historical"] = df_original_yf.copy()
            self._data_cache[f"{symbol}_{timeframe}_historical_ta"] = data_with_indicators.set_index('timestamp').copy() # 분석/UI용은 timestamp 인덱스

        self.historical_data_updated_signal.emit(symbol, df_original_yf.copy()) # ChartWidget용 (yf 원본)
        self.technical_indicators_updated_signal.emit(symbol, timeframe, data_with_indicators.set_index('timestamp').copy()) # DataWidget용 (timestamp 인덱스)
        
        # ML 분석 요청
        analysis_df = data_with_indicators.set_index('timestamp').copy()
        if not hasattr(analysis_df, 'attrs'): analysis_df.attrs = {}
        analysis_df.attrs['timeframe'] = timeframe
        analysis_df.attrs['symbol'] = symbol
        self.request_symbol_analysis(symbol, timeframe, analysis_df)

    def request_historical_data(self, symbol: str, timeframe: str, force_fetch_from_api: bool = False):
        request_key = f"historical_data_{symbol}_{timeframe}"
        task_display_name = f"{symbol}({timeframe}) 과거 데이터"

        if not force_fetch_from_api:
            # 1. 메모리 캐시 확인
            cached_original = self.get_cached_historical_data(symbol, timeframe, with_ta=False)
            cached_ta = self.get_cached_historical_data(symbol, timeframe, with_ta=True) # timestamp 인덱스
            if cached_original is not None and not cached_original.empty and cached_ta is not None and not cached_ta.empty:
                self.logger.info(f"{task_display_name}: 메모리 캐시에서 로드.")
                self.historical_data_updated_signal.emit(symbol, cached_original.copy())
                self.technical_indicators_updated_signal.emit(symbol, timeframe, cached_ta.copy())
                self.request_symbol_analysis(symbol, timeframe, cached_ta.copy()) # cached_ta는 이미 timestamp 인덱스
                self.task_feedback_signal.emit(task_display_name, "캐시 로드 완료")
                return

            # 2. DB 확인
            if self.db_manager:
                self.task_feedback_signal.emit(task_display_name, "DB 조회 중...")
                # DB에서 최근 2년치 데이터 조회 (예시), 시간 오름차순, timestamp 인덱스 포함
                db_original = self.db_manager.get_stock_prices(symbol, timeframe, 
                                                                start_date=datetime.now() - timedelta(days=730), 
                                                                order_desc=False) 
                if db_original is not None and not db_original.empty:
                    self.logger.info(f"{task_display_name}: DB에서 원본 데이터 로드 ({len(db_original)} 행).")
                    # db_original은 timestamp 인덱스, 소문자 컬럼명을 가진다고 가정 (DatabaseManager에서 처리)
                    # ChartWidget이 yfinance 원본(대문자 컬럼, DatetimeIndex)을 기대하면 변환 필요
                    db_original_for_chart = db_original.rename(columns=self._get_ohlcv_mapping(to_lower=False))
                    
                    self._process_and_emit_historical_data(symbol, timeframe, db_original_for_chart, task_display_name)
                    self.task_feedback_signal.emit(task_display_name, "DB 로드 완료")
                    return
                else:
                    self.logger.info(f"{task_display_name}: DB에 데이터 없음. API 호출 진행.")
        
        self._dispatch_api_task(request_key, task_display_name, symbol, timeframe)
    def _dispatch_api_task(self, request_key:str, task_display_name:str, symbol:str, timeframe:str):
        """API 호출 작업을 WorkflowEngine에 제출합니다."""
        if self._is_task_active(request_key):
            return
        
        self.task_feedback_signal.emit(task_display_name, "API 요청 시작")
        self._add_active_task(request_key)
        task = Task(
            id=request_key, name=task_display_name,
            function=self._task_fetch_historical_data_from_api, # API 호출 전용 태스크 함수
            args=(symbol, timeframe),
            kwargs={'request_key': request_key, 'task_display_name': task_display_name},
            priority=1
        )
        self.workflow_engine.submit_task(task)

    def _task_fetch_historical_data_from_api(self, symbol: str, timeframe: str, request_key: str, task_display_name: str) -> Optional[pd.DataFrame]:
        self.logger.debug(f"태스크 시작 (API 호출): {task_display_name}")
        data_df_original_yf = None # yfinance에서 직접 받은 원본 (DatetimeIndex, 대문자 컬럼)
        try:
            self.task_feedback_signal.emit(task_display_name, "API 주가 데이터 요청 중...")
            
            yf_period = "1y"
            if timeframe in ['1분', '5분', '15분', '30분', '1시간', '90분']: yf_period = "60d" 
            elif timeframe == '1일': yf_period = "2y" 
            elif timeframe == '1주': yf_period = "5y"
            elif timeframe == '1개월': yf_period = "max"

            data_df_original_yf = self.stock_collector.get_historical_data(symbol, timeframe, period=yf_period)

            if data_df_original_yf is None or data_df_original_yf.empty:
                self.logger.warning(f"{task_display_name}: API 주가 데이터 수집 실패 또는 데이터 없음.")
                self.task_feedback_signal.emit(task_display_name, "API 주가 데이터 없음")
                self.historical_data_updated_signal.emit(symbol, pd.DataFrame())
                self.technical_indicators_updated_signal.emit(symbol, timeframe, pd.DataFrame())
                return None

            self.logger.info(f"{task_display_name}: API 주가 데이터 수집 성공 ({len(data_df_original_yf)} 행).")
            
            df_for_db_save = self._standardize_df_columns_and_index(data_df_original_yf, symbol, timeframe)
            if self.db_manager and not df_for_db_save.empty:
                self.db_manager.save_stock_prices(symbol, timeframe, df_for_db_save.copy(), update_existing=True)

            self._process_and_emit_historical_data(symbol, timeframe, data_df_original_yf, task_display_name)
            self.task_feedback_signal.emit(task_display_name, "API 데이터 처리 완료")
            return data_df_original_yf

        except Exception as e:
            self.logger.error(f"{task_display_name} API 처리 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"API 오류: {str(e)[:30]}...")
            self.historical_data_updated_signal.emit(symbol, pd.DataFrame())
            self.technical_indicators_updated_signal.emit(symbol, timeframe, pd.DataFrame())
            return None
        finally:
            self._remove_active_task(request_key)
            self.logger.debug(f"태스크 종료 (API 호출): {task_display_name}")

    def request_realtime_quote(self, symbol: str):
        request_key = f"realtime_{symbol}"
        task_display_name = f"{symbol} 실시간 호가"
        # 실시간 데이터는 캐시만 사용하고 DB는 보통 사용하지 않음 (또는 별도 로그 테이블)
        cached_data = self.get_cached_realtime_data(symbol)
        if cached_data: # 캐시 유효기간 로직 추가 가능 (예: 1분 이내 데이터면 사용)
            self.logger.debug(f"{task_display_name}: 메모리 캐시에서 로드.")
            self.realtime_quote_updated_signal.emit(symbol, cached_data)
            # self.task_feedback_signal.emit(task_display_name, "캐시 로드 완료") # 너무 빈번할 수 있음
            # return # 여기서 API 호출 없이 바로 반환할 수도 있음 (주기적 업데이트 타이머에 의존)
        
        if self._is_task_active(request_key): return

        self.task_feedback_signal.emit(task_display_name, "API 요청 시작")
        self._add_active_task(request_key)
        task = Task(
            id=request_key, name=task_display_name,
            function=self._task_fetch_realtime_quote, args=(symbol,),
            kwargs={'request_key': request_key, 'task_display_name': task_display_name}, priority=0
        )
        self.workflow_engine.submit_task(task)

    def _task_fetch_realtime_quote(self, symbol: str, request_key: str, task_display_name: str) -> Optional[Dict]:
        self.logger.debug(f"태스크 시작 (API): {task_display_name}")
        quote_data = None
        try:
            # self.task_feedback_signal.emit(task_display_name, "API 실시간 호가 요청 중...") # 너무 빈번할 수 있음
            quote_data = self.stock_collector.get_realtime_quote(symbol)
            if quote_data:
                # self.logger.info(f"{task_display_name}: API 실시간 호가 수집 성공.") # 로그 레벨 조정
                with self._cache_lock:
                    self._data_cache[f"{symbol}_realtime"] = quote_data
                self.realtime_quote_updated_signal.emit(symbol, quote_data)
                # self.task_feedback_signal.emit(task_display_name, "API 로드 완료")
                return quote_data
            else:
                self.logger.warning(f"{task_display_name}: API 실시간 호가 수집 실패 또는 데이터 없음.")
                # self.task_feedback_signal.emit(task_display_name, "API 로드 실패")
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} API 처리 중 예외: {e}", exc_info=True)
            # self.task_feedback_signal.emit(task_display_name, f"API 오류: {str(e)[:30]}...")
            return None
        finally:
            self._remove_active_task(request_key)
            self.logger.debug(f"태스크 종료 (API): {task_display_name}")

    def request_news_data(self, symbol_or_keywords: str, language: str = 'en', force_fetch_from_api: bool = False):
        request_key = f"news_{symbol_or_keywords.replace(' ','_')}_{language}"
        task_display_name = f"뉴스({symbol_or_keywords})"

        if not force_fetch_from_api:
            cached_news = self.get_cached_news_data(request_key) # 캐시 키를 request_key로 사용
            if cached_news: # 캐시가 List[Dict] 형태
                self.logger.info(f"{task_display_name}: 메모리 캐시에서 뉴스 로드.")
                self.news_data_updated_signal.emit(symbol_or_keywords, cached_news)
                self.task_feedback_signal.emit(task_display_name, "뉴스 캐시 로드 완료")
                return
            
            if self.db_manager:
                self.task_feedback_signal.emit(task_display_name, "DB 뉴스 조회 중...")
                keywords_list = [k.strip() for k in symbol_or_keywords.split("OR") if k.strip()]
                db_news_df = self.db_manager.get_news_articles(
                    keywords=keywords_list, 
                    start_date=datetime.now() - timedelta(days=3), # 최근 3일
                    limit=30 # 최대 30개
                ) 
                if db_news_df is not None and not db_news_df.empty:
                    db_news_list = db_news_df.to_dict('records') # DataFrame -> List[Dict]
                    self.logger.info(f"{task_display_name}: DB에서 뉴스 로드 ({len(db_news_list)}개).")
                    with self._cache_lock: self._data_cache[request_key] = db_news_list
                    self.news_data_updated_signal.emit(symbol_or_keywords, db_news_list)
                    self.task_feedback_signal.emit(task_display_name, "DB 뉴스 로드 완료")
                    return
        
        if self._is_task_active(request_key) and not force_fetch_from_api: return

        self.task_feedback_signal.emit(task_display_name, "API 뉴스 요청 시작")
        self._add_active_task(request_key)
        task = Task(
            id=f"{request_key}_{datetime.now().timestamp()}", name=task_display_name,
            function=self._task_fetch_news_data_from_api, args=(symbol_or_keywords, language, request_key),
            kwargs={'task_display_name': task_display_name}, priority=2
        )
        self.workflow_engine.submit_task(task)

    def _task_fetch_news_data_from_api(self, symbol_or_keywords: str, language: str, request_key: str, task_display_name: str) -> Optional[List[Dict]]:
        self.logger.debug(f"태스크 시작 (API): {task_display_name}")
        news_list = None
        try:
            self.task_feedback_signal.emit(task_display_name, "API 뉴스 데이터 요청 중...")
            news_list = self.news_collector.get_latest_news_by_keywords(symbol_or_keywords, language=language, page_size=20)

            if news_list is not None: # API가 빈 리스트를 반환할 수도 있으므로 None만 체크
                self.logger.info(f"{task_display_name}: API 뉴스 수집 완료 ({len(news_list)}개).")
                
                if self.db_manager and news_list:
                    # NewsArticle 모델에 맞게 데이터 변환 필요 (예: source, publishedAt 등)
                    processed_news_list = []
                    for item in news_list:
                        source_info = item.get('source')
                        p_item = {
                            'source_name': source_info.get('name') if isinstance(source_info, dict) else source_info,
                            'title': item.get('title'),
                            'url': item.get('url'),
                            'published_at': pd.to_datetime(item.get('publishedAt'), errors='coerce').to_pydatetime() if item.get('publishedAt') else datetime.now(),
                            'content_snippet': item.get('description') or item.get('content',''), # API 응답에 따라
                            'related_symbols': [k.strip() for k in symbol_or_keywords.split("OR")] # 간단히 키워드를 심볼로
                        }
                        if p_item['title'] and p_item['url']: # 필수값 체크
                             processed_news_list.append(p_item)
                    if processed_news_list:
                        self.db_manager.save_news_articles(processed_news_list, update_existing=True)
                
                with self._cache_lock: self._data_cache[request_key] = news_list # 원본 API 응답 캐시
                self.news_data_updated_signal.emit(symbol_or_keywords, news_list)
                self.task_feedback_signal.emit(task_display_name, f"API 뉴스 로드 완료 ({len(news_list)}개)")
                return news_list
            else:
                self.logger.warning(f"{task_display_name}: API 뉴스 수집 실패 또는 데이터 없음.")
                self.task_feedback_signal.emit(task_display_name, "API 뉴스 로드 실패")
                self.news_data_updated_signal.emit(symbol_or_keywords, [])
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} API 처리 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"API 오류: {str(e)[:30]}...")
            self.news_data_updated_signal.emit(symbol_or_keywords, [])
            return None
        finally:
            self._remove_active_task(request_key)
            self.logger.debug(f"태스크 종료 (API): {task_display_name}")

    # --- Economic Calendar Data --- (신규 추가) ---
    def request_economic_calendar_update(self, days_future: int = 7, 
                                         target_countries: Optional[List[str]] = None, 
                                         importance_min: int = 1,
                                         force_fetch_from_api: bool = False):
        request_key = f"economic_calendar_days_{days_future}"
        task_display_name = f"경제 캘린더 ({days_future}일)"

        if not force_fetch_from_api:
            # 1. 메모리 캐시 확인 (간단한 기간 기반 캐시)
            # 경제 캘린더는 자주 변하지 않으므로, 캐시 유효 기간을 설정할 수 있음 (예: 1시간)
            cached_data = self.get_cached_economic_calendar_data()
            if cached_data is not None and not cached_data.empty:
                 # 캐시된 데이터의 최신 날짜 확인 (예시)
                if not cached_data.empty and 'datetime' in cached_data.columns:
                    last_cached_date = pd.to_datetime(cached_data['datetime'].max())
                    # 캐시 데이터가 요청 범위(오늘~days_future)를 충분히 포함하고 최근에 업데이트되었다면 사용
                    if last_cached_date >= (datetime.now() + timedelta(days=days_future) - timedelta(hours=1)): # 1시간 이내 업데이트
                        self.logger.info(f"{task_display_name}: 메모리 캐시에서 로드.")
                        self.economic_data_updated_signal.emit(cached_data.copy())
                        self.task_feedback_signal.emit(task_display_name, "캐시 로드 완료")
                        return
            
            # 2. DB 확인
            if self.db_manager:
                self.task_feedback_signal.emit(task_display_name, "DB 조회 중...")
                db_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                db_end = db_start + timedelta(days=days_future)
                db_events = self.db_manager.get_economic_events(
                    start_date=db_start, end_date=db_end, 
                    country_codes=target_countries, importance_min=importance_min
                )
                if db_events is not None and not db_events.empty:
                    self.logger.info(f"{task_display_name}: DB에서 로드 ({len(db_events)} 개).")
                    with self._cache_lock:
                        self._data_cache["economic_calendar"] = db_events.copy()
                    self.economic_data_updated_signal.emit(db_events.copy())
                    self.task_feedback_signal.emit(task_display_name, "DB 로드 완료")
                    return
        
        # 3. API 호출 (Selenium 스크레이핑)
        if self._is_task_active(request_key) and not force_fetch_from_api: # 강제 호출 아니면 중복 방지
             return

        self.task_feedback_signal.emit(task_display_name, "스크레이핑 시작")
        self._add_active_task(request_key)
        start_date_param = datetime.now()
        end_date_param = start_date_param + timedelta(days=days_future)
        task = Task(
            id=request_key, name=task_display_name,
            function=self._task_fetch_economic_calendar_from_web, # 웹 스크레이핑 전용 태스크
            args=(start_date_param, end_date_param, target_countries, importance_min),
            kwargs={'request_key': request_key, 'task_display_name': task_display_name},
            priority=1 
        )
        self.workflow_engine.submit_task(task)

    def _task_fetch_economic_calendar_from_web(self, start_date: datetime, end_date: datetime, 
                                     target_countries: Optional[List[str]], importance_min: int,
                                     request_key:str, task_display_name: str) -> Optional[pd.DataFrame]:
        self.logger.debug(f"태스크 시작 (웹 스크레이핑): {task_display_name}")
        calendar_df = None
        try:
            self.task_feedback_signal.emit(task_display_name, "웹에서 경제 지표 수집 중...")
            calendar_df = self.economic_calendar_collector.fetch_calendar_data(
                start_date=start_date, end_date=end_date,
                countries=target_countries, importance_min=importance_min
            )

            if calendar_df is not None and not calendar_df.empty:
                self.logger.info(f"{task_display_name}: 웹 스크레이핑 성공 ({len(calendar_df)} 개 이벤트).")
                if self.db_manager:
                    self.db_manager.save_economic_events(calendar_df.copy(), update_existing=True)
                
                with self._cache_lock:
                    self._data_cache["economic_calendar"] = calendar_df.copy()
                
                self.economic_data_updated_signal.emit(calendar_df.copy())
                self.task_feedback_signal.emit(task_display_name, f"웹 로드 완료 ({len(calendar_df)}개)")
                return calendar_df
            # ... (이하 기존 _task_fetch_economic_calendar의 결과 처리 로직과 유사하게)
            elif calendar_df is not None and calendar_df.empty:
                self.logger.info(f"{task_display_name}: 웹에서 수집된 이벤트 없음.")
                # 빈 DataFrame이라도 캐시/UI 업데이트하여 이전 데이터가 계속 보이지 않도록
                with self._cache_lock: self._data_cache["economic_calendar"] = pd.DataFrame()
                self.economic_data_updated_signal.emit(pd.DataFrame())
                self.task_feedback_signal.emit(task_display_name, "웹 데이터 없음")
                return pd.DataFrame()
            else: # None 반환 시
                self.logger.warning(f"{task_display_name}: 웹 스크레이핑 실패.")
                self.task_feedback_signal.emit(task_display_name, "웹 로드 실패")
                self.economic_data_updated_signal.emit(pd.DataFrame())
                return None

        except Exception as e:
            self.logger.error(f"{task_display_name} 웹 스크레이핑 태스크 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"웹 오류: {str(e)[:30]}...")
            self.economic_data_updated_signal.emit(pd.DataFrame())
            return None
        finally:
            self._remove_active_task(request_key)
            self.logger.debug(f"태스크 종료 (웹 스크레이핑): {task_display_name}")
    # --- Cached Data Getters ---


    def get_cached_historical_data(self, symbol: str, timeframe: str, with_ta: bool = False) -> Optional[pd.DataFrame]:
        with self._cache_lock:
            key = f"{symbol}_{timeframe}_historical_ta" if with_ta else f"{symbol}_{timeframe}_historical"
            data = self._data_cache.get(key)
            return data.copy() if data is not None else None # 복사본 반환

    def get_cached_economic_calendar_data(self) -> Optional[pd.DataFrame]:
        with self._cache_lock:
            data = self._data_cache.get("economic_calendar")
            return data.copy() if data is not None else None

    def get_cached_news_data(self, cache_key: str) -> Optional[List[Dict]]:
        with self._cache_lock:
            # 뉴스 데이터는 리스트이므로, 내부 딕셔너리까지 깊은 복사 고려 (현재는 얕은 복사)
            data = self._data_cache.get(cache_key)
            return list(data) if data is not None else None
			
    def request_symbol_analysis(self, symbol: str, timeframe: str, data_df: Optional[pd.DataFrame] = None):
        task_display_name = f"{symbol}({timeframe}) AI 분석" # 이름 구체화
        self.task_feedback_signal.emit(task_display_name, "요청 시작")

        if data_df is None: # 분석용 데이터가 전달되지 않았다면 캐시에서 로드 시도
            with self._cache_lock:
                # 기술적 지표가 포함된 데이터를 우선적으로 찾음
                data_df = self._data_cache.get(f"{symbol}_{timeframe}_historical_ta")
            if data_df is None: # 지표 포함 데이터가 없으면 원본이라도 찾음 (이 경우 분석 모델이 내부적으로 지표 계산해야 함)
                 with self._cache_lock:
                    data_df = self._data_cache.get(f"{symbol}_{timeframe}_historical")
        
        if data_df is None or data_df.empty:
            msg = f"{task_display_name} 실패: AI 분석용 데이터 없음."
            self.status_message_signal.emit(msg, 5000)
            self.logger.warning(msg)
            self.task_feedback_signal.emit(task_display_name, "AI 분석 데이터 부족")
            return

        # attrs는 복사 시 유지되지 않을 수 있으므로, 다시 설정하거나 확인
        if not hasattr(data_df, 'attrs'):
            data_df.attrs = {}
        data_df.attrs['timeframe'] = timeframe
        data_df.attrs['symbol'] = symbol

        task = Task(
            id=f"analyze_symbol_{symbol}_{timeframe.replace(' ','_')}", name=task_display_name,
            function=self._task_run_analysis, args=(symbol, data_df.copy()), # 데이터 복사본 전달
            kwargs={'task_display_name': task_display_name}, priority=2 # 데이터 수집보다 낮은 우선순위
        )
        self.workflow_engine.submit_task(task)

    def _task_run_analysis(self, symbol: str, data_df: pd.DataFrame, task_display_name: str) -> Optional[Dict]:
        # ... (이하 동일, data_df는 기술적 지표가 이미 포함된 것으로 가정하고 사용)
        self.logger.debug(f"태스크 시작: {task_display_name} (데이터 컬럼: {data_df.columns.tolist()[:5]}...)")
        analysis_results = {}
        try:
            self.task_feedback_signal.emit(task_display_name, "AI 모델 예측 중...")
            # TempModelManager의 predict는 'close' 컬럼을 사용함. df_for_ta에서 소문자로 변경했으므로 괜찮음.
            prediction = self.model_manager.predict(symbol, data_df) 
            if prediction:
                analysis_results['ml_prediction'] = prediction
                self.logger.debug(f"{task_display_name} - ML 예측 완료: {prediction}")
            
            # 여기에 뉴스 감성, 경제 지표 등을 종합하는 로직 추가 예정
            # 예: analysis_results['news_sentiment_score'] = self.get_latest_news_sentiment(symbol)
            # 예: analysis_results['economic_impact_assessment'] = self.assess_economic_events(symbol, data_df.index[-1])

            if analysis_results:
                self.analysis_result_updated_signal.emit(symbol, analysis_results)
                self.task_feedback_signal.emit(task_display_name, "AI 분석 완료")
                self.logger.info(f"{task_display_name} 완료. 결과: {analysis_results}")
                return analysis_results
            else:
                self.logger.info(f"{task_display_name}: AI 분석 결과 없음.")
                self.task_feedback_signal.emit(task_display_name, "AI 분석 결과 없음")
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} 태스크 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"AI 분석 오류: {str(e)[:30]}...")
            return None
        finally:
            self.logger.debug(f"태스크 종료: {task_display_name}")

    def get_cached_realtime_data(self, symbol: str) -> Optional[Dict]:
        with self._cache_lock:
            data = self._data_cache.get(f"{symbol}_realtime")
            return dict(data) if data is not None else None # 딕셔너리 복사본

    def _on_task_started(self, event_data: Dict[str, Any]):
        task_id = event_data.get('task_id')
        task = self.workflow_engine.tasks.get(task_id)
        task_name = task.name if task else task_id
        self.logger.info(f"이벤트 수신: 작업 시작됨 - '{task_name}' (ID: {task_id})")

    def _on_task_completed(self, event_data: Dict[str, Any]):
        task_id = event_data.get('task_id')
        task = self.workflow_engine.tasks.get(task_id)
        task_name = task.name if task else task_id
        self.logger.info(f"이벤트 수신: 작업 완료 - '{task_name}' (ID: {task_id})")

    def _on_task_failed(self, event_data: Dict[str, Any]):
        task_id = event_data.get('task_id')
        task = self.workflow_engine.tasks.get(task_id)
        task_name = task.name if task else task_id
        error_msg = event_data.get('error')
        self.logger.error(f"이벤트 수신: 작업 실패 - '{task_name}' (ID: {task_id}) - 오류: {error_msg}")

    def apply_updated_settings(self):
        self.logger.info("변경된 설정 적용 시도...")
        try:
            # API 키 변경 시 수집기 재초기화
            self.stock_collector = StockDataCollector(self.settings.api_config)
            self.news_collector = NewsCollector(self.settings.api_config)
            
            if self.db_manager: # DB 매니저가 있다면 DB 설정도 업데이트 시도
                self.logger.info("DB 설정 변경 감지. DatabaseManager 재초기화 시도...")
                self.db_manager.close() # 기존 연결 종료
                self.db_manager = DatabaseManager(database_config=self.settings.db_config)
                self.logger.info("DatabaseManager 재초기화 완료.")

            self.logger.info("설정 적용 완료: 데이터 수집기 및 DB 매니저 재초기화됨.")
            self.status_message_signal.emit("설정이 성공적으로 업데이트 및 적용되었습니다.", 3000)
        except Exception as e:
            self.logger.error(f"설정 적용 중 오류 발생: {e}", exc_info=True)
            self.status_message_signal.emit("설정 적용 중 오류 발생.", 5000)