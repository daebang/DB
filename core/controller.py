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
from analysis.integrated_analyzer import IntegratedAnalyzer # 추가
from analysis.sentiment.news_sentiment import NewsSentimentAnalyzer # IntegratedAnalyzer에 주입하기 위해
from analysis.prediction.timeframe_predictor import TimeframePredictor # 만약 위 파일이 analysis/prediction 폴더로 이동한다면



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

    # --- Current Symbol Tracking ---
    # This needs to be set by the UI when the symbol changes.
    # MainWindow._on_symbol_changed should call a method here.
    _current_selected_symbol: Optional[str] = None

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        self.event_manager = EventManager()
        self.workflow_engine = WorkflowEngine(
            self.event_manager,
            num_workers=None,  # 자동 설정
            min_workers=2,     # 최소 2개
            max_workers=10     # 최대 10개
        )
        self.scheduler = Scheduler()

        try:
            self.db_manager = DatabaseManager(database_config=settings.db_config)
            self.logger.info("데이터베이스 매니저 초기화 완료")
        except Exception as e:
            self.logger.critical(f"데이터베이스 매니저 초기화 실패: {e}", exc_info=True)
            self.db_manager = None
        self.stock_collector = StockDataCollector(settings.api_config) # yfinance 기반 Collector
        self.news_sentiment_analyzer = NewsSentimentAnalyzer() # APIConfig 주입 고려
        self.news_collector = NewsCollector(settings.api_config)
        self.economic_calendar_collector = EconomicCalendarCollector() # api_config는 현재 불필요
        self.model_manager = TempModelManager()
        self.integrated_analyzer = IntegratedAnalyzer(news_sentiment_analyzer=self.news_sentiment_analyzer)

        self._data_cache: Dict[str, Any] = {}
        self._cache_lock = Lock()
        self._active_data_requests: set[str] = set() # 모든 데이터 요청 키 관리 (중복 방지용)

        # --- Initial Data Load Status Flags ---
        self._initial_news_loaded_for_symbol: Dict[str, bool] = {}
        self._initial_econ_cal_loaded: bool = False
        self._pending_analysis_for_symbol: Dict[str, bool] = {} # Track if analysis is pending due to missing data
        self._register_event_handlers()
        self._setup_scheduled_tasks()
        


        
        self.logger.info("메인 컨트롤러 초기화 완료. DB 연동 및 캐싱 로직 포함.")
    def set_current_symbol(self, symbol: str):
        """MainWindow에서 호출하여 현재 UI에 선택된 심볼을 설정합니다."""
        self.logger.debug(f"Controller: 현재 선택된 심볼 변경됨 -> {symbol}")
        self._current_selected_symbol = symbol
        # 심볼 변경 시, 해당 심볼에 대한 분석 재요청 로직도 고려 가능 (만약 모든 데이터가 준비되었다면)
        if self.is_all_initial_data_loaded_for_symbol(symbol):
            self.logger.info(f"심볼 변경({symbol}): 모든 초기 데이터 로드됨. 분석 재요청 시도.")
            # 과거 데이터가 이미 로드되어 있고, TA도 있다면 바로 분석 요청
            cached_ta_data = self.get_cached_historical_data(symbol, self._get_current_timeframe_for_symbol(symbol), with_ta=True, copy=False)
            if cached_ta_data is not None and not cached_ta_data.empty:
                self.request_symbol_analysis(symbol, self._get_current_timeframe_for_symbol(symbol), cached_ta_data.copy())
            else:
                self.logger.warning(f"심볼 변경({symbol}): TA 데이터 캐시에 없어 분석 요청 보류. 과거 데이터 로드 후 자동 실행될 것임.")
        else:
            self.logger.info(f"심볼 변경({symbol}): 일부 초기 데이터 미로드 상태. 자동 분석 대기.")


    def _get_current_timeframe_for_symbol(self, symbol: str) -> str:
        """
        현재 심볼에 대한 차트 위젯의 타임프레임을 가져오는 헬퍼.
        실제 구현에서는 MainWindow에서 이 정보를 받아오거나, Controller가 상태를 알아야 합니다.
        여기서는 임시로 기본값을 반환합니다.
        """
        # TODO: MainWindow와 연동하여 실제 현재 선택된 timeframe을 가져오도록 수정
        # 예를 들어, self.main_window.chart_widget.timeframe_combo.currentText() 와 같이
        # 지금은 임시로 '1일'을 사용
        if hasattr(self, 'main_window_ref') and self.main_window_ref and hasattr(self.main_window_ref, 'chart_widget'):
             return self.main_window_ref.chart_widget.timeframe_combo.currentText()
        return '1일' # 기본값

    def is_all_initial_data_loaded_for_symbol(self, symbol: str) -> bool:
        """특정 심볼에 대한 모든 주요 초기 데이터(주가, 뉴스)와 공통 경제 데이터가 로드되었는지 확인합니다."""
        stock_data_key = f"{symbol}_{self._get_current_timeframe_for_symbol(symbol)}_historical_ta"
        is_stock_loaded = stock_data_key in self._data_cache and not self._data_cache[stock_data_key].empty
        is_news_loaded = self._initial_news_loaded_for_symbol.get(symbol, False)
        return is_stock_loaded and is_news_loaded and self._initial_econ_cal_loaded

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
                # 예: 30분마다 현재 선택된 심볼 뉴스 업데이트 (만약 _current_selected_symbol이 있다면)
                self.scheduler.add_job(
                    self.request_news_update_for_current_symbol,
                    'interval', minutes=30, id="periodic_news_update", replace_existing=True
                )
                self.logger.info("현재 심볼 뉴스 주기적 업데이트 작업 스케줄됨 (30분 간격).")
            except Exception as e:
                self.logger.error(f"스케줄된 작업 설정 중 오류: {e}", exc_info=True)
        else:
            self.logger.warning("스케줄러 객체가 초기화되지 않아 스케줄 작업을 설정할 수 없습니다.")

    def request_news_update_for_current_symbol(self):
        """현재 UI에서 선택된 심볼에 대한 뉴스를 강제로 업데이트합니다."""
        if self._current_selected_symbol:
            self.logger.info(f"스케줄러: '{self._current_selected_symbol}'에 대한 뉴스 업데이트 요청 (강제 API 호출).")
            # MainWindow에서 symbol_combo.currentText()를 가져와서 사용
            self.request_news_data(f"{self._current_selected_symbol} stock news OR {self._current_selected_symbol}", force_fetch_from_api=True)
        else:
            self.logger.debug("스케줄러: 현재 선택된 심볼이 없어 뉴스 업데이트를 건너<0xEB><0x9A><0x88>니다.")

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
            self.task_feedback_signal.emit(task_display_name, "주가 데이터 없음 (처리 중단)")
            return

        self.task_feedback_signal.emit(task_display_name, "기술적 지표 계산 중...")
        df_for_ta = self._standardize_df_columns_and_index(df_original_yf, symbol, timeframe) # 표준화된 df (timestamp 컬럼, 소문자 ohlcv)
        price_col_for_ta = 'adj_close' if 'adj_close' in df_for_ta.columns and df_for_ta['adj_close'].notna().any() else 'close'
        
        required_cols_for_ta = ['open', 'high', 'low', price_col_for_ta]
        if not all(col in df_for_ta.columns for col in required_cols_for_ta) or df_for_ta[required_cols_for_ta].isnull().all().any():
            self.logger.error(f"{task_display_name}: TA용 필수 컬럼 누락 또는 모두 NaN: {df_for_ta.columns.tolist()}")
            data_with_indicators = df_for_ta.set_index('timestamp').copy() if 'timestamp' in df_for_ta else df_for_ta.copy()
        else:
            data_with_indicators = add_all_selected_indicators(df_for_ta.copy(), price_col=price_col_for_ta)
            # data_with_indicators는 timestamp 컬럼을 가짐 (add_all_selected_indicators는 인덱스 변경 안 함)
            if 'timestamp' in data_with_indicators.columns:
                 data_with_indicators = data_with_indicators.set_index('timestamp')
        
        self.logger.info(f"{task_display_name}: 기술적 지표 계산 완료.")
        
        if self.db_manager:
            indicator_cols = [col for col in data_with_indicators.columns if any(kw in col for kw in ['SMA_', 'EMA_', 'RSI_', 'MACD', 'BB_'])]
            if indicator_cols and not data_with_indicators.empty: # 이미 timestamp 인덱스
                self.db_manager.save_bulk_technical_indicators(
                    data_with_indicators.reset_index(), symbol, timeframe, indicator_cols, update_existing=True
                )

        with self._cache_lock:
            self._data_cache[f"{symbol}_{timeframe}_historical"] = df_original_yf.copy() # 원본 yf
            self._data_cache[f"{symbol}_{timeframe}_historical_ta"] = data_with_indicators.copy() # timestamp 인덱스, TA 포함

        self.historical_data_updated_signal.emit(symbol, df_original_yf.copy())
        self.technical_indicators_updated_signal.emit(symbol, timeframe, data_with_indicators.copy())
        
        # 통합 분석 요청 전, 해당 심볼에 대한 초기 데이터 로드가 완료되었는지 확인
        # 이 시점에는 주가 데이터는 로드되었으므로, is_all_initial_data_loaded_for_symbol 사용
        if self.is_all_initial_data_loaded_for_symbol(symbol):
            self.logger.info(f"{task_display_name}: 모든 관련 데이터 로드 확인, 통합 분석 요청.")
            self.request_symbol_analysis(symbol, timeframe, data_with_indicators.copy())
        else:
            self.logger.info(f"{task_display_name}: 뉴스/경제 데이터 미로드, 통합 분석 보류. '{symbol}'에 대한 분석 펜딩 설정.")
            self._pending_analysis_for_symbol[symbol] = True
            # ML 예측만이라도 먼저 수행하고 UI에 일부 정보 표시
            partial_analysis_results = {}
            ml_prediction = self.model_manager.predict(symbol, data_with_indicators.copy())
            if ml_prediction:
                partial_analysis_results['ml_prediction'] = ml_prediction
            partial_analysis_results['integrated_analysis'] = {'summary': "뉴스/경제 데이터 로딩 중..."} # 임시 메시지
            self.analysis_result_updated_signal.emit(symbol, partial_analysis_results)

    def request_historical_data(self, symbol: str, timeframe: str, force_fetch_from_api: bool = False):
        request_key = f"historical_data_{symbol}_{timeframe}"
        task_display_name = f"{symbol}({timeframe}) 과거 데이터"
        self._current_selected_symbol = symbol # 현재 작업중인 심볼 업데이트

        if not force_fetch_from_api:
            # 1. 메모리 캐시 확인
            cached_original = self.get_cached_historical_data(symbol, timeframe, with_ta=False)
            cached_ta = self.get_cached_historical_data(symbol, timeframe, with_ta=True) # timestamp 인덱스
            if cached_original is not None and not cached_original.empty and cached_ta is not None and not cached_ta.empty:
                self.logger.info(f"{task_display_name}: 메모리 캐시에서 로드.")
                self.historical_data_updated_signal.emit(symbol, cached_original.copy())
                self.technical_indicators_updated_signal.emit(symbol, timeframe, cached_ta.copy())
                # 캐시에서 로드 시에도 통합 분석 요청
                if self.is_all_initial_data_loaded_for_symbol(symbol):
                     self.request_symbol_analysis(symbol, timeframe, cached_ta.copy())
                else:
                    self.logger.info(f"{task_display_name} (캐시): 뉴스/경제 데이터 미로드, 통합 분석 보류.")
                    self._pending_analysis_for_symbol[symbol] = True
                    # ML 예측만이라도 먼저 수행
                    partial_analysis_results = {}
                    ml_prediction = self.model_manager.predict(symbol, cached_ta.copy())
                    if ml_prediction: partial_analysis_results['ml_prediction'] = ml_prediction
                    partial_analysis_results['integrated_analysis'] = {'summary': "뉴스/경제 데이터 로딩 중..."}
                    self.analysis_result_updated_signal.emit(symbol, partial_analysis_results)

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
                # 실패 시에도 _pending_analysis_for_symbol 상태를 업데이트 할 필요는 없음. 데이터 자체가 없는 것이므로.
                return None

            self.logger.info(f"{task_display_name}: API 주가 데이터 수집 성공 ({len(data_df_original_yf)} 행).")
            
            df_for_db_save = self._standardize_df_columns_and_index(data_df_original_yf, symbol, timeframe)
            if self.db_manager and not df_for_db_save.empty:
                # DB 저장 시, df_for_db_save가 timestamp 컬럼을 가지도록 확인 (현재 _standardize... 함수가 그렇게 함)
                self.db_manager.save_stock_prices(symbol, timeframe, df_for_db_save.set_index('timestamp').copy(), update_existing=True)


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
        # symbol_or_keywords에서 실제 심볼 추출 (예: "AAPL stock news OR AAPL" -> "AAPL")
        # 이 심볼을 _initial_news_loaded_for_symbol의 키로 사용
        actual_symbol_for_flag = symbol_or_keywords.split(" ")[0].upper() # 간단한 추출 방식, 개선 필요

        request_key = f"news_{actual_symbol_for_flag}_{language}" # 캐시/활성 태스크 키는 정제된 심볼 사용
        task_display_name = f"뉴스({actual_symbol_for_flag})"


        if not force_fetch_from_api:
            cached_news = self.get_cached_news_data(request_key)
            if cached_news:
                self.logger.info(f"{task_display_name}: 메모리 캐시에서 뉴스 로드.")
                self.news_data_updated_signal.emit(actual_symbol_for_flag, cached_news) # 시그널은 실제 심볼과 함께
                self._initial_news_loaded_for_symbol[actual_symbol_for_flag] = True # 캐시에서 로드해도 초기 로드 완료로 간주
                self._check_and_run_pending_analysis(actual_symbol_for_flag)
                self.task_feedback_signal.emit(task_display_name, "뉴스 캐시 로드 완료")
                return
            
            if self.db_manager:
                self.task_feedback_signal.emit(task_display_name, "DB 뉴스 조회 중...")
                db_news_df = self.db_manager.get_news_articles(
                    symbol=actual_symbol_for_flag, # DB 조회 시에는 정제된 심볼 사용
                    start_date=datetime.now() - timedelta(days=3), limit=30
                ) 
                if db_news_df is not None and not db_news_df.empty:
                    db_news_list = db_news_df.to_dict('records')
                    self.logger.info(f"{task_display_name}: DB에서 뉴스 로드 ({len(db_news_list)}개).")
                    with self._cache_lock: self._data_cache[request_key] = db_news_list
                    self.news_data_updated_signal.emit(actual_symbol_for_flag, db_news_list)
                    self._initial_news_loaded_for_symbol[actual_symbol_for_flag] = True # DB 로드도 초기 로드 완료
                    self._check_and_run_pending_analysis(actual_symbol_for_flag)
                    self.task_feedback_signal.emit(task_display_name, "DB 뉴스 로드 완료")
                    return
        
        if self._is_task_active(request_key) and not force_fetch_from_api: return

        self.task_feedback_signal.emit(task_display_name, "API 뉴스 요청 시작")
        self._add_active_task(request_key)
        task = Task(
            id=f"{request_key}_{datetime.now().timestamp()}", name=task_display_name,
            function=self._task_fetch_news_data_from_api, args=(symbol_or_keywords, language, actual_symbol_for_flag, request_key), # actual_symbol_for_flag 추가
            kwargs={'task_display_name': task_display_name}, priority=2
        )
        self.workflow_engine.submit_task(task)

    def _task_fetch_news_data_from_api(self, symbol_or_keywords: str, language: str, actual_symbol_for_flag: str, request_key: str, task_display_name: str) -> Optional[List[Dict]]:
        self.logger.debug(f"태스크 시작 (API): {task_display_name}")
        news_list = None
        try:
            self.task_feedback_signal.emit(task_display_name, "API 뉴스 데이터 요청 중...")
            news_list = self.news_collector.get_latest_news_by_keywords(symbol_or_keywords, language=language, page_size=20)

            if news_list is not None:
                self.logger.info(f"{task_display_name}: API 뉴스 수집 완료 ({len(news_list)}개).")
                
                processed_for_db_list = []
                if news_list: # 뉴스가 있을 때만 DB 저장 시도
                    for item in news_list: # DB 저장용 데이터 변환
                        source_info = item.get('source')
                        p_item = {
                            'source_name': source_info.get('name') if isinstance(source_info, dict) else source_info,
                            'title': item.get('title'),
                            'url': item.get('url'),
                            'published_at': pd.to_datetime(item.get('published_at'), errors='coerce', utc=True).to_pydatetime() if item.get('published_at') else datetime.utcnow(),
                            'content_snippet': item.get('description') or item.get('text_for_analysis', ''), # text_for_analysis 사용
                            'related_symbols': [actual_symbol_for_flag], # 실제 심볼로 저장
                            # 감성분석 결과는 NewsSentimentAnalyzer를 통해 별도로 추가되거나, NewsCollector에서 할 수도 있음.
                            # 여기서는 NewsCollector가 반환하는 text_for_analysis를 저장한다고 가정.
                            # sentiment_score, sentiment_label은 DB 저장 시점에 채워지거나, 나중에 업데이트 될 수 있음.
                        }
                        if p_item['title'] and p_item['url']:
                             # NewsSentimentAnalyzer로 각 뉴스 아이템 감성 분석 후 저장
                            if 'text_for_analysis' in item and item['text_for_analysis']:
                                sentiment_result = self.news_sentiment_analyzer.analyze_sentiment(item['text_for_analysis'], symbol=actual_symbol_for_flag)
                                p_item['sentiment_score'] = sentiment_result.get('sentiment_score')
                                p_item['sentiment_label'] = sentiment_result.get('sentiment_label')
                            processed_for_db_list.append(p_item)

                if self.db_manager and processed_for_db_list:
                    self.db_manager.save_news_articles(processed_for_db_list, update_existing=True)
                
                with self._cache_lock: self._data_cache[request_key] = news_list
                self.news_data_updated_signal.emit(actual_symbol_for_flag, news_list) # 실제 심볼과 함께
                self._initial_news_loaded_for_symbol[actual_symbol_for_flag] = True
                self._check_and_run_pending_analysis(actual_symbol_for_flag)
                self.task_feedback_signal.emit(task_display_name, f"API 뉴스 로드 완료 ({len(news_list)}개)")
                return news_list
            else: # news_list is None (API 실패) 또는 빈 리스트
                self.logger.warning(f"{task_display_name}: API 뉴스 수집 실패 또는 데이터 없음.")
                self.task_feedback_signal.emit(task_display_name, "API 뉴스 로드 실패")
                self.news_data_updated_signal.emit(actual_symbol_for_flag, [])
                # 실패 시에도 해당 심볼에 대한 뉴스 로드가 시도되었음을 알림 (무한 대기 방지)
                self._initial_news_loaded_for_symbol[actual_symbol_for_flag] = True 
                self._check_and_run_pending_analysis(actual_symbol_for_flag) # 데이터가 없어도 분석은 시도 (IntegratedAnalyzer가 처리)
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} API 처리 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"API 오류: {str(e)[:30]}...")
            self.news_data_updated_signal.emit(actual_symbol_for_flag, [])
            self._initial_news_loaded_for_symbol[actual_symbol_for_flag] = True
            self._check_and_run_pending_analysis(actual_symbol_for_flag)
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
                if 'datetime' in cached_data.columns:
                    last_cached_date_obj = pd.to_datetime(cached_data['datetime'].max())
                    # 캐시 만료 시간 (예: 6시간)
                    cache_expiry_time = datetime.now() - timedelta(hours=6)
                    # 캐시된 데이터의 마지막 날짜가 요청 범위를 커버하고, 캐시가 너무 오래되지 않았다면 사용
                    if last_cached_date_obj.tz_localize(None) >= (datetime.now() + timedelta(days=days_future)).replace(hour=0, minute=0, second=0, microsecond=0) and \
                       last_cached_date_obj.tz_localize(None) > cache_expiry_time : # 마지막 업데이트 시간 체크도 추가 가능
                        self.logger.info(f"{task_display_name}: 메모리 캐시에서 로드.")
                        self.economic_data_updated_signal.emit(cached_data.copy())
                        self._initial_econ_cal_loaded = True
                        self._check_and_run_pending_analysis_for_all_symbols() # 경제 지표는 모든 심볼에 영향
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
                    self._initial_econ_cal_loaded = True
                    self._check_and_run_pending_analysis_for_all_symbols()
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
                self._initial_econ_cal_loaded = True
                self._check_and_run_pending_analysis_for_all_symbols()
                self.task_feedback_signal.emit(task_display_name, f"웹 로드 완료 ({len(calendar_df)}개)")
                return calendar_df
            # ... (이하 기존 _task_fetch_economic_calendar의 결과 처리 로직과 유사하게)
            elif calendar_df is not None and calendar_df.empty:
                self.logger.info(f"{task_display_name}: 웹에서 수집된 이벤트 없음.")
                # 빈 DataFrame이라도 캐시/UI 업데이트하여 이전 데이터가 계속 보이지 않도록
                with self._cache_lock: self._data_cache["economic_calendar"] = pd.DataFrame()
                self.economic_data_updated_signal.emit(pd.DataFrame())
                self._initial_econ_cal_loaded = True
                self._check_and_run_pending_analysis_for_all_symbols()
                self.task_feedback_signal.emit(task_display_name, "웹 데이터 없음")
                return pd.DataFrame()
            else: # None 반환 시
                self.logger.warning(f"{task_display_name}: 웹 스크레이핑 실패.")
                self.task_feedback_signal.emit(task_display_name, "웹 로드 실패")
                self.economic_data_updated_signal.emit(pd.DataFrame()) # 빈 DF 전달
                self._initial_econ_cal_loaded = True # 실패해도 일단 로드 시도 완료로 간주 (무한 대기 방지)
                self._check_and_run_pending_analysis_for_all_symbols()
                return None

        except Exception as e:
            self.logger.error(f"{task_display_name} 웹 스크레이핑 태스크 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"웹 오류: {str(e)[:30]}...")
            self.economic_data_updated_signal.emit(pd.DataFrame())
            self._initial_econ_cal_loaded = True
            self._check_and_run_pending_analysis_for_all_symbols()
            return None
        finally:
            self._remove_active_task(request_key)
            self.logger.debug(f"태스크 종료 (웹 스크레이핑): {task_display_name}")
            
    def _check_and_run_pending_analysis(self, symbol: str):
        """특정 심볼에 대해 보류 중인 분석이 있고 모든 데이터가 준비되면 실행합니다."""
        if self._pending_analysis_for_symbol.get(symbol, False) and self.is_all_initial_data_loaded_for_symbol(symbol):
            self.logger.info(f"'{symbol}'에 대한 보류된 통합 분석 실행 조건 충족. 분석 재요청.")
            timeframe = self._get_current_timeframe_for_symbol(symbol) # 현재 심볼의 타임프레임 가져오기
            cached_ta_data = self.get_cached_historical_data(symbol, timeframe, with_ta=True, copy=False)
            if cached_ta_data is not None and not cached_ta_data.empty:
                self.request_symbol_analysis(symbol, timeframe, cached_ta_data.copy())
                self._pending_analysis_for_symbol[symbol] = False # 보류 해제
            else:
                self.logger.warning(f"'{symbol}' 보류 분석 실행 시도 중 TA 데이터 캐시에 없음. 과거 데이터 요청 선행 필요.")
                # 이 경우 request_historical_data를 호출하여 전체 플로우를 다시 타도록 할 수 있음
                # self.request_historical_data(symbol, timeframe, force_fetch_from_api=False) # force_fetch_from_api=False로 캐시/DB 먼저 시도

    def _check_and_run_pending_analysis_for_all_symbols(self):
        """경제 캘린더와 같이 모든 심볼에 영향을 줄 수 있는 데이터가 업데이트되었을 때 호출됩니다."""
        if self._current_selected_symbol: # 현재 선택된 심볼이 있다면 해당 심볼에 대해 체크
            self._check_and_run_pending_analysis(self._current_selected_symbol)
        # 또는 모든 _pending_analysis_for_symbol을 순회하며 조건 충족 시 실행
        # for symbol in list(self._pending_analysis_for_symbol.keys()):
        #     if self._pending_analysis_for_symbol.get(symbol, False):
        #         self._check_and_run_pending_analysis(symbol)

    def get_cached_historical_data(self, symbol: str, timeframe: str, with_ta: bool = False, copy: bool = True) -> Optional[pd.DataFrame]:
        """
        캐시된 과거 데이터 반환
        
        Args:
            symbol: 주식 심볼
            timeframe: 시간 프레임
            with_ta: 기술적 지표 포함 여부
            copy: DataFrame 복사본 반환 여부 (False면 원본 참조 반환)
        """
        with self._cache_lock:
            key = f"{symbol}_{timeframe}_historical_ta" if with_ta else f"{symbol}_{timeframe}_historical"
            data = self._data_cache.get(key)
            if data is None:
                return None
            return data.copy() if copy else data

    def get_cached_economic_calendar_data(self, copy: bool = True) -> Optional[pd.DataFrame]:
        """
        캐시된 경제 캘린더 데이터 반환
        
        Args:
            copy: DataFrame 복사본 반환 여부 (False면 원본 참조 반환)
        """
        with self._cache_lock:
            data = self._data_cache.get("economic_calendar")
            if data is None:
                return None
            return data.copy() if copy else data
        
    def get_cached_news_data(self, cache_key: str) -> Optional[List[Dict]]:
        with self._cache_lock:
            # 뉴스 데이터는 리스트이므로, 내부 딕셔너리까지 깊은 복사 고려 (현재는 얕은 복사)
            data = self._data_cache.get(cache_key)
            return list(data) if data is not None else None
			
    def get_cached_realtime_data(self, symbol: str) -> Optional[Dict]:
        with self._cache_lock:
            data = self._data_cache.get(f"{symbol}_realtime")
            return dict(data) if data is not None and isinstance(data, dict) else None
			
    def request_symbol_analysis(self, symbol: str, timeframe: str, data_df: Optional[pd.DataFrame] = None):
        task_display_name = f"{symbol}({timeframe}) AI 분석"
        request_key = f"analyze_symbol_{symbol}_{timeframe.replace(' ','_')}" # 고유한 ID 생성

        # 이미 활성화된 분석 작업이 있다면 중복 제출 방지 (선택적)
        if self._is_task_active(request_key):
             self.logger.debug(f"{task_display_name} 작업이 이미 진행 중입니다. 중복 요청 건너<0xEB><0x9A><0x88>니다.")
             return

        self.task_feedback_signal.emit(task_display_name, "요청 시작")
        self._add_active_task(request_key) # 작업 시작 시 활성 목록에 추가

        # 분석용 데이터 준비 (기술적 지표 포함된 데이터)
        if data_df is None or data_df.empty:
            data_df_for_analysis = self.get_cached_historical_data(symbol, timeframe, with_ta=True, copy=False)
            if data_df_for_analysis is None or data_df_for_analysis.empty:
                msg = f"{task_display_name} 실패: AI 분석용 TA 데이터 없음."
                self.status_message_signal.emit(msg, 5000)
                self.logger.warning(msg)
                self.task_feedback_signal.emit(task_display_name, "AI 분석 TA 데이터 부족")
                self._remove_active_task(request_key) # 작업 실패로 간주하고 제거
                # 빈 결과라도 시그널을 보내 UI를 초기화 할 수 있음
                self.analysis_result_updated_signal.emit(symbol, {'ml_prediction': None, 'integrated_analysis': {'summary': "분석 데이터 부족"}})
                return
        else:
            data_df_for_analysis = data_df # 제공된 데이터 사용 (복사본으로 전달됨)

        if not hasattr(data_df_for_analysis, 'attrs'): data_df_for_analysis.attrs = {}
        data_df_for_analysis.attrs['timeframe'] = timeframe
        data_df_for_analysis.attrs['symbol'] = symbol

        task = Task(
            id=request_key, name=task_display_name, # id를 request_key로 사용
            function=self._task_run_analysis, args=(symbol, data_df_for_analysis.copy()),
            kwargs={'task_display_name': task_display_name, 'request_key': request_key}, priority=2
        )
        self.workflow_engine.submit_task(task)


    def _task_run_analysis(self, symbol: str, data_df_with_ta: pd.DataFrame, task_display_name: str, request_key: str) -> Optional[Dict]:
        self.logger.info(f"태스크 시작: {task_display_name} ({symbol}) - AI 기간별 예측 실행.")
        self.task_feedback_signal.emit(task_display_name, "AI 기간별 예측 분석 중...")

        try:
            # 0. TimeframePredictor 인스턴스 생성
            # TimeframePredictor는 프로젝트 루트에 있다고 가정. 아니라면 경로 조정.
            # 예: from analysis.timeframe_predictor import TimeframePredictor
            predictor = TimeframePredictor(symbol=symbol)

            # 1. TimeframePredictor에 필요한 데이터 준비
            # data_df_with_ta는 timestamp 인덱스와 TA 컬럼들을 가지고 있음.
            
            #   1.1 historical_data: 원본 OHLCV (TimeframePredictor는 내부적으로 특징 생성 시 사용)
            #       data_df_with_ta에서 TA를 제외한 원본 OHLCV 컬럼만 추출하거나,
            #       _process_and_emit_historical_data에서 원본 yf 데이터를 캐싱했다면 그것을 사용.
            #       여기서는 data_df_with_ta에서 필요한 컬럼만 선택하여 전달.
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume'] # 표준 소문자 컬럼명 가정
            # data_df_with_ta의 인덱스가 'timestamp'인지 확인, 아니면 set_index('timestamp')
            current_historical_data_df = data_df_with_ta.copy()
            if not isinstance(current_historical_data_df.index, pd.DatetimeIndex) and 'timestamp' in current_historical_data_df.columns:
                current_historical_data_df = current_historical_data_df.set_index('timestamp')
            
            # historical_data는 순수 OHLCV를 기대할 수 있으므로, TA 컬럼 제외 (선택적, TimeframePredictor 내부 처리 방식에 따라 다름)
            # BaseTimeframePredictionModel.prepare_features가 'close'등을 직접 사용하므로,
            # 여기서는 TA가 포함된 데이터를 historical_data로 넘겨도 prepare_features에서 알아서 처리할 것으로 기대.
            # 또는, 명시적으로 OHLCV만 선택
            # base_historical_data = current_historical_data_df[ohlcv_cols].copy()


            #   1.2 technical_indicators: TA가 포함된 DataFrame
            #       data_df_with_ta 자체가 이 역할을 할 수 있음. TimeframePredictor의
            #       _prepare_unified_features가 이 데이터를 받아 TA 컬럼을 활용.
            technical_indicators_df = current_historical_data_df.copy()


            #   1.3 news_sentiment: 뉴스 데이터 (DataFrame 형태, sentiment_score 등 포함)
            news_cache_key = f"news_{symbol}_en" # 언어는 설정 또는 동적으로 조정
            recent_news_list = self.get_cached_news_data(news_cache_key)
            news_sentiment_df = pd.DataFrame(recent_news_list) if recent_news_list else pd.DataFrame()
            if not news_sentiment_df.empty and 'published_at' in news_sentiment_df.columns:
                news_sentiment_df['published_at'] = pd.to_datetime(news_sentiment_df['published_at'], errors='coerce', utc=True)
                # TimeframePredictor의 prepare_features가 news_sentiment를 날짜별로 집계하므로 DatetimeIndex 필요
                if isinstance(news_sentiment_df.index, pd.DatetimeIndex): # 이미 DatetimeIndex면 그대로 사용
                    pass
                elif 'published_at' in news_sentiment_df.columns and news_sentiment_df['published_at'].notna().any():
                    news_sentiment_df = news_sentiment_df.set_index('published_at').sort_index()
                else: # DatetimeIndex 설정 불가 시 빈 DF 전달
                    logger.warning(f"{symbol}: 뉴스 데이터에 유효한 published_at 정보가 없어 인덱싱 불가. 빈 DF로 처리.")
                    news_sentiment_df = pd.DataFrame()


            #   1.4 economic_indicators: 경제 지표 데이터 (DataFrame 형태)
            economic_indicators_df = self.get_cached_economic_calendar_data(copy=False) # 원본 참조 가능
            if economic_indicators_df is None:
                economic_indicators_df = pd.DataFrame()
            elif not economic_indicators_df.empty and 'datetime' in economic_indicators_df.columns:
                 # TimeframePredictor의 prepare_features가 economic_indicators를 날짜별로 집계하므로 DatetimeIndex 필요
                if isinstance(economic_indicators_df.index, pd.DatetimeIndex):
                    pass
                elif 'datetime' in economic_indicators_df.columns and economic_indicators_df['datetime'].notna().any():
                    economic_indicators_df = economic_indicators_df.set_index('datetime').sort_index()
                else:
                    logger.warning(f"{symbol}: 경제 지표 데이터에 유효한 datetime 정보가 없어 인덱싱 불가. 빈 DF로 처리.")
                    economic_indicators_df = pd.DataFrame()


            #   1.5 fundamental_data: 현재는 None으로 전달
            fundamental_data_df = None

            # 2. TimeframePredictor를 사용하여 예측 수행
            # current_historical_data_df는 이미 timestamp를 인덱스로 가짐
            predictions_data = predictor.predict_all_timeframes(
                historical_data=current_historical_data_df, # 원본 데이터 (또는 ohlcv_cols만 선택한 데이터)
                technical_indicators=technical_indicators_df, # TA 포함된 데이터
                news_sentiment=news_sentiment_df,
                economic_indicators=economic_indicators_df,
                fundamental_data=fundamental_data_df
            )

            if predictions_data:
                self.logger.info(f"{task_display_name} ({symbol}) - AI 기간별 예측 완료.")
                # DataWidget의 update_analysis_display는 predictions_data 전체를 받도록 설계됨
                self.analysis_result_updated_signal.emit(symbol, predictions_data.copy())
                self.task_feedback_signal.emit(task_display_name, "AI 기간별 예측 분석 완료")
                return predictions_data
            else:
                self.logger.warning(f"{task_display_name} ({symbol}) - AI 기간별 예측 결과 없음.")
                # 빈 결과 또는 기본 메시지 UI 전달
                default_empty_prediction = {
                    'symbol': symbol, 'timestamp': datetime.now(), 'current_price': None,
                    'timeframes': {}, 'overall_confidence': 0.0,
                    'recommendation': {'action': '분석 불가', 'strength': '', 'reasoning': ['예측 모델 실행에 실패했거나 결과가 없습니다.'], 'risk_level': '높음', 'suggested_strategy': ''}
                }
                self.analysis_result_updated_signal.emit(symbol, default_empty_prediction)
                self.task_feedback_signal.emit(task_display_name, "AI 기간별 예측 결과 없음")
                return None

        except Exception as e:
            self.logger.error(f"{task_display_name} ({symbol}) 태스크 중 예외 발생: {e}", exc_info=True)
            error_prediction_output = {
                'symbol': symbol, 'timestamp': datetime.now(), 'current_price': None,
                'timeframes': {}, 'overall_confidence': 0.0,
                'recommendation': {'action': '오류 발생', 'strength': '', 'reasoning': [f'분석 중 오류 발생: {str(e)[:100]}...'], 'risk_level': '알수없음', 'suggested_strategy': ''},
                'error': str(e)
            }
            self.analysis_result_updated_signal.emit(symbol, error_prediction_output)
            self.task_feedback_signal.emit(task_display_name, f"AI 분석 오류: {str(e)[:30]}...")
            return None
        finally:
            self._remove_active_task(request_key)
            self.logger.info(f"태스크 종료: {task_display_name} ({symbol})")
            

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
        error_msg = event_data.get('error')
        task = self.workflow_engine.tasks.get(task_id)
        task_name = task.name if task else task_id
        # 실패한 작업은 활성 목록에서 제거 (WorkflowEngine에서 task_id를 주므로 사용 가능)
        if task_id and task_id in self._active_data_requests:
             self._remove_active_task(task_id)
        self.logger.error(f"이벤트 수신: 작업 실패 - '{task_name}' (ID: {task_id}) - 오류: {error_msg}")

    def apply_updated_settings(self):
        self.logger.info("변경된 설정 적용 시도...")
        try:
            self.stock_collector = StockDataCollector(self.settings.api_config)
            self.news_collector = NewsCollector(self.settings.api_config)
            # IntegratedAnalyzer와 NewsSentimentAnalyzer도 API 키 변경에 영향받으면 재초기화 필요
            self.news_sentiment_analyzer = NewsSentimentAnalyzer() # settings.api_config 주입 고려
            self.integrated_analyzer = IntegratedAnalyzer(news_sentiment_analyzer=self.news_sentiment_analyzer)


            if self.db_manager:
                self.logger.info("DB 설정 변경 감지. DatabaseManager 재초기화 시도...")
                self.db_manager.close()
                self.db_manager = DatabaseManager(database_config=self.settings.db_config)
                self.logger.info("DatabaseManager 재초기화 완료.")

            self.logger.info("설정 적용 완료.")
            self.status_message_signal.emit("설정이 성공적으로 업데이트 및 적용되었습니다.", 3000)
        except Exception as e:
            self.logger.error(f"설정 적용 중 오류 발생: {e}", exc_info=True)
            self.status_message_signal.emit("설정 적용 중 오류 발생.", 5000)