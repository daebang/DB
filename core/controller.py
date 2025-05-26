# core/controller.py
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta # timedelta 추가
from threading import Thread, Lock
import queue
import pandas as pd

from PyQt5.QtCore import QObject, pyqtSignal

from utils.scheduler import Scheduler
from core.workflow_engine import WorkflowEngine, Task
from core.event_manager import EventManager
from data.collectors.stock_data_collector import StockDataCollector
from data.collectors.news_collector import NewsCollector
from data.collectors.economic_calendar_collector import EconomicCalendarCollector
from config.settings import Settings
from analysis.technical.indicators import add_all_selected_indicators

# 임시 ModelManager (이전과 동일)
class TempModelManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TempModelManager")

    def predict(self, symbol: str, data: Optional[pd.DataFrame]) -> Optional[Dict]:
        self.logger.debug(f"임시 모델 매니저: {symbol} 예측 요청 (데이터 {len(data) if data is not None else 0}개)")
        if data is not None and not data.empty and 'close' in data.columns and len(data['close']) >= 5:
            # 'close' 컬럼이 소문자로 변환되었다고 가정
            if data['close'].iloc[-1] > data['close'].iloc[-5:].mean():
                return {"action": "BUY", "confidence": 0.6, "reason": "최근 상승세 (임시)", "target_price": data['close'].iloc[-1] * 1.05}
            else:
                return {"action": "SELL", "confidence": 0.6, "reason": "최근 하락세 또는 횡보 (임시)", "target_price": data['close'].iloc[-1] * 0.95}
        self.logger.warning(f"임시 모델 매니저: {symbol} 예측 위한 데이터 부족 (close 컬럼: {'close' in data.columns if data is not None else 'N/A'})")
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
        self.workflow_engine = WorkflowEngine(self.event_manager, num_workers=3)
        self.scheduler = Scheduler()

        self.stock_collector = StockDataCollector(settings.api_config) # yfinance 기반 Collector
        self.news_collector = NewsCollector(settings.api_config)
        self.economic_calendar_collector = EconomicCalendarCollector() # api_config는 현재 불필요
        self.model_manager = TempModelManager()

        self._data_cache: Dict[str, Any] = {}
        self._cache_lock = Lock()
        self._active_symbol_data_requests: set[str] = set()
        self._active_economic_data_request = False # 경제 데이터 요청 상태 플래그

        self._register_event_handlers()
        self._setup_scheduled_tasks()
        # self._connect_controller_signals() # 이 메서드는 MainController 내부 시그널 연결용이 아니므로 주석 처리 또는 삭제
        self.logger.info("메인 컨트롤러 초기화 완료. 기술적 지표 로직 포함.")

    def _register_event_handlers(self):
        self.event_manager.subscribe('task_started', self._on_task_started)
        self.event_manager.subscribe('task_completed', self._on_task_completed)
        self.event_manager.subscribe('task_failed', self._on_task_failed)
        self.logger.debug("컨트롤러 이벤트 핸들러 등록 완료.")

    def _setup_scheduled_tasks(self):
        # 예: 매일 아침 경제 캘린더 데이터 업데이트
        try:
            self.scheduler.add_job(
                self.request_economic_calendar_update,
                'cron',
                hour=7, minute=0,
                id="daily_economic_calendar_update",
                replace_existing=True # 기존 작업이 있다면 대체
            )
            # 스케줄러 시작은 start() 메서드에서 한 번만 호출
            self.logger.info("경제 캘린더 일일 업데이트 작업 스케줄됨 (매일 07:00).")
        except Exception as e:
            self.logger.error(f"스케줄된 작업 설정 중 오류: {e}", exc_info=True)

    def start(self):
        self.logger.info("컨트롤러 시작: 워크플로우 엔진 및 스케줄러 가동 시도.")
        self.workflow_engine.start()
        if not self.scheduler.scheduler.running:
            try:
                self.scheduler.start()
                self.logger.info("스케줄러 시작됨.")
            except Exception as e: # 이미 시작된 경우 등 예외 처리
                self.logger.warning(f"스케줄러 시작 중 문제 발생 (이미 실행 중일 수 있음): {e}")
        self.connection_status_changed_signal.emit("시스템 준비됨 (컨트롤러 시작)", True)

    def stop(self):
        self.logger.info("컨트롤러 중지 요청: 워크플로우 엔진 및 스케줄러 종료 시도.")
        self.workflow_engine.stop()
        if self.scheduler.scheduler.running:
            self.scheduler.shutdown()
            self.logger.info("스케줄러 종료됨.")
        self.logger.info("워크플로우 엔진 종료됨.")
        self.connection_status_changed_signal.emit("시스템 중지됨", False)

    def request_historical_data(self, symbol: str, timeframe: str):
        request_key = f"hist_{symbol}_{timeframe}"
        task_display_name = f"{symbol}({timeframe}) 과거 데이터"

        if request_key in self._active_symbol_data_requests:
            self.status_message_signal.emit(f"{task_display_name}는 이미 요청 중입니다.", 3000)
            return

        self.task_feedback_signal.emit(task_display_name, "요청 시작")
        self._active_symbol_data_requests.add(request_key)

        task = Task(
            id=request_key, # 작업 ID로 request_key 사용
            name=task_display_name,
            function=self._task_fetch_historical_data,
            args=(symbol, timeframe), # yfinance는 period를 사용하므로, timeframe에 따라 적절한 period를 여기서 결정하거나,
                                     # _task_fetch_historical_data 내부에서 결정하도록 함.
            kwargs={'request_key': request_key, 'task_display_name': task_display_name},
            priority=1
        )
        self.workflow_engine.submit_task(task)

    def _task_fetch_historical_data(self, symbol: str, timeframe: str, request_key: str, task_display_name: str) -> Optional[pd.DataFrame]:
        self.logger.debug(f"태스크 시작: {task_display_name}")
        data_df_original = None
        try:
            self.task_feedback_signal.emit(task_display_name, "주가 데이터 요청 중...")
            
            yf_period = "1y"
            if timeframe in ['1분', '5분', '15분', '30분', '1시간', '90분']: yf_period = "60d" # 분봉은 최대 60일
            elif timeframe == '1일': yf_period = "2y" # 일봉은 최근 2년치
            elif timeframe == '1주': yf_period = "5y"
            elif timeframe == '1개월': yf_period = "max"

            data_df_original = self.stock_collector.get_historical_data(symbol, timeframe, period=yf_period)

            if data_df_original is None or data_df_original.empty:
                self.logger.warning(f"{task_display_name}: 주가 데이터 수집 실패 또는 데이터 없음.")
                self.task_feedback_signal.emit(task_display_name, "주가 데이터 없음")
                self.historical_data_updated_signal.emit(symbol, pd.DataFrame())
                self.technical_indicators_updated_signal.emit(symbol, timeframe, pd.DataFrame())
                return None

            self.logger.info(f"{task_display_name}: 주가 데이터 수집 성공 ({len(data_df_original)} 행).")
            self.task_feedback_signal.emit(task_display_name, "기술적 지표 계산 중...")

            # 기술적 지표 계산을 위해 컬럼명 소문자로 변경 및 표준화
            ohlcv_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
            }
            df_for_ta = data_df_original.rename(columns=ohlcv_mapping)
            
            # 'adj_close'가 있으면 사용하고, 없으면 'close' 사용
            price_col_for_ta = 'adj_close' if 'adj_close' in df_for_ta.columns else 'close'

            # 필수 컬럼 확인 (소문자 기준)
            required_cols_for_ta = ['open', 'high', 'low', price_col_for_ta] # volume은 선택적
            if not all(col in df_for_ta.columns for col in required_cols_for_ta):
                self.logger.error(f"{task_display_name}: 기술적 지표 계산에 필요한 OHLC 컬럼 누락. 컬럼: {df_for_ta.columns.tolist()}")
                data_with_indicators = df_for_ta.copy() # 지표 없이 전달
            else:
                data_with_indicators = add_all_selected_indicators(df_for_ta.copy(), price_col=price_col_for_ta)
            
            self.logger.info(f"{task_display_name}: 기술적 지표 계산 완료.")
            self.task_feedback_signal.emit(task_display_name, "지표 계산 완료")

            with self._cache_lock:
                self._data_cache[f"{symbol}_{timeframe}_historical"] = data_df_original.copy() # 원본 캐시
                self._data_cache[f"{symbol}_{timeframe}_historical_ta"] = data_with_indicators.copy() # 지표 포함본 캐시

            self.historical_data_updated_signal.emit(symbol, data_df_original.copy())
            self.technical_indicators_updated_signal.emit(symbol, timeframe, data_with_indicators.copy())
            
            self.task_feedback_signal.emit(task_display_name, "데이터 로드 완료")

            # ML 분석 요청 (지표가 포함된 데이터 사용)
            data_with_indicators.attrs['timeframe'] = timeframe
            data_with_indicators.attrs['symbol'] = symbol
            self.request_symbol_analysis(symbol, timeframe, data_with_indicators)
            
            return data_df_original # 태스크 결과로는 원본 DataFrame 반환

        except Exception as e:
            self.logger.error(f"{task_display_name} 처리 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"오류: {str(e)[:30]}...")
            self.historical_data_updated_signal.emit(symbol, pd.DataFrame())
            self.technical_indicators_updated_signal.emit(symbol, timeframe, pd.DataFrame())
            return None
        finally:
            if request_key in self._active_symbol_data_requests:
                self._active_symbol_data_requests.remove(request_key)
            self.logger.debug(f"태스크 종료: {task_display_name}")

    def request_realtime_quote(self, symbol: str):
        task_display_name = f"{symbol} 실시간 호가"
        self.task_feedback_signal.emit(task_display_name, "요청 시작")
        task = Task(
            id=f"fetch_realtime_{symbol}", name=task_display_name,
            function=self._task_fetch_realtime_quote, args=(symbol,),
            kwargs={'task_display_name': task_display_name}, priority=0
        )
        self.workflow_engine.submit_task(task)

    def _task_fetch_realtime_quote(self, symbol: str, task_display_name: str) -> Optional[Dict]:
        self.logger.debug(f"태스크 시작: {task_display_name}")
        quote_data = None
        try:
            self.task_feedback_signal.emit(task_display_name, "데이터 요청 중...")
            quote_data = self.stock_collector.get_realtime_quote(symbol)
            if quote_data:
                self.logger.info(f"{task_display_name}: 수집 성공.")
                with self._cache_lock:
                    self._data_cache[f"{symbol}_realtime"] = quote_data
                self.realtime_quote_updated_signal.emit(symbol, quote_data)
                self.task_feedback_signal.emit(task_display_name, "로드 완료")
                return quote_data
            else:
                self.logger.warning(f"{task_display_name}: 수집 실패 또는 데이터 없음.")
                self.task_feedback_signal.emit(task_display_name, "로드 실패")
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} 수집 태스크 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"오류: {str(e)[:50]}...")
            return None
        finally:
            self.logger.debug(f"태스크 종료: {task_display_name}")

    def request_news_data(self, symbol_or_keywords: str, language: str = 'en'):
        task_display_name = f"뉴스({symbol_or_keywords})"
        self.task_feedback_signal.emit(task_display_name, "요청 시작")
        task = Task(
            id=f"fetch_news_{symbol_or_keywords.replace(' ','_')}", name=task_display_name,
            function=self._task_fetch_news_data, args=(symbol_or_keywords, language),
            kwargs={'task_display_name': task_display_name}, priority=2
        )
        self.workflow_engine.submit_task(task)

    def _task_fetch_news_data(self, symbol_or_keywords: str, language: str, task_display_name: str) -> Optional[List[Dict]]:
        self.logger.debug(f"태스크 시작: {task_display_name}")
        news_list = None
        try:
            self.task_feedback_signal.emit(task_display_name, "데이터 요청 중...")
            news_list = self.news_collector.get_latest_news_by_keywords(symbol_or_keywords, language=language, page_size=10)
            if news_list is not None: # API가 빈 리스트를 반환할 수도 있으므로 None 체크
                self.logger.info(f"{task_display_name}: 수집 완료 ({len(news_list)}개).")
                # 뉴스 데이터는 휘발성이므로 UI에 직접 전달, 캐시는 선택적
                self.news_data_updated_signal.emit(symbol_or_keywords, news_list)
                self.task_feedback_signal.emit(task_display_name, f"로드 완료 ({len(news_list)}개)")
                return news_list
            else: # news_list가 None인 경우 (API 오류 등)
                self.logger.warning(f"{task_display_name}: 뉴스 수집 실패 (API 응답 없음).")
                self.task_feedback_signal.emit(task_display_name, "로드 실패 (API 오류)")
                self.news_data_updated_signal.emit(symbol_or_keywords, []) # 빈 리스트 전달
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} 수집 태스크 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"오류: {str(e)[:50]}...")
            self.news_data_updated_signal.emit(symbol_or_keywords, [])
            return None
        finally:
            self.logger.debug(f"태스크 종료: {task_display_name}")

    # --- Economic Calendar Data --- (신규 추가) ---
    def request_economic_calendar_update(self, days_future: int = 7, 
                                         target_countries: Optional[List[str]] = None, 
                                         importance_min: int = 1):
        """
        경제 캘린더 데이터 업데이트 요청.
        Args:
            days_future: 오늘부터 몇 일 후까지의 데이터를 가져올지 (기본값: 7일)
            target_countries: 대상 국가 코드 리스트 (예: ['US', 'KR']). None이면 주요 국가 기본값 사용.
            importance_min: 최소 중요도 (1-3).
        """
        if self._active_economic_data_request:
            self.status_message_signal.emit("경제 캘린더 데이터는 이미 요청 중입니다.", 3000)
            return

        task_display_name = f"경제 캘린더 ({days_future}일)"
        self.task_feedback_signal.emit(task_display_name, "요청 시작")
        self._active_economic_data_request = True

        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_future)

        task = Task(
            id="fetch_economic_calendar", # ID는 고유해야 함 (필요시 timestamp 추가)
            name=task_display_name,
            function=self._task_fetch_economic_calendar,
            args=(start_date, end_date, target_countries, importance_min),
            kwargs={'task_display_name': task_display_name},
            priority=1 # 주식 데이터와 유사한 우선순위
        )
        self.workflow_engine.submit_task(task)

    def _task_fetch_economic_calendar(self, start_date: datetime, end_date: datetime, 
                                     target_countries: Optional[List[str]], importance_min: int,
                                     task_display_name: str) -> Optional[pd.DataFrame]:
        self.logger.debug(f"태스크 시작: {task_display_name}")
        calendar_df = None
        try:
            self.task_feedback_signal.emit(task_display_name, "데이터 수집 중...")
            calendar_df = self.economic_calendar_collector.fetch_calendar_data(
                start_date=start_date,
                end_date=end_date,
                countries=target_countries,
                importance_min=importance_min
            )

            if calendar_df is not None and not calendar_df.empty:
                self.logger.info(f"{task_display_name}: 수집 성공 ({len(calendar_df)} 개 이벤트).")
                with self._cache_lock:
                    self._data_cache["economic_calendar"] = calendar_df.copy() # 캐시 키 정의
                
                self.economic_data_updated_signal.emit(calendar_df.copy()) # UI 업데이트용 시그널
                self.task_feedback_signal.emit(task_display_name, f"로드 완료 ({len(calendar_df)}개)")
                return calendar_df
            elif calendar_df is not None and calendar_df.empty:
                self.logger.info(f"{task_display_name}: 수집은 되었으나, 해당 조건의 이벤트가 없습니다.")
                with self._cache_lock:
                    self._data_cache["economic_calendar"] = pd.DataFrame()
                self.economic_data_updated_signal.emit(pd.DataFrame())
                self.task_feedback_signal.emit(task_display_name, "데이터 없음")
                return pd.DataFrame()
            else: # calendar_df is None (수집 중 오류 발생 가능성)
                self.logger.warning(f"{task_display_name}: 수집 실패 또는 데이터 없음.")
                self.task_feedback_signal.emit(task_display_name, "로드 실패")
                self.economic_data_updated_signal.emit(pd.DataFrame()) # 오류 시 빈 DF 전달
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} 수집 태스크 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"오류: {str(e)[:50]}...")
            self.economic_data_updated_signal.emit(pd.DataFrame())
            return None
        finally:
            self._active_economic_data_request = False
            self.logger.debug(f"태스크 종료: {task_display_name}")


    # --- Symbol Analysis --- (기존 코드 유지)
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
            return self._data_cache.get(f"{symbol}_realtime")

    def get_cached_historical_data(self, symbol: str, timeframe: str, with_ta: bool = False) -> Optional[pd.DataFrame]:
         with self._cache_lock:
            if with_ta:
                return self._data_cache.get(f"{symbol}_{timeframe}_historical_ta")
            return self._data_cache.get(f"{symbol}_{timeframe}_historical")
    
    def get_cached_economic_calendar_data(self) -> Optional[pd.DataFrame]:
        with self._cache_lock:
            return self._data_cache.get("economic_calendar")

    def _on_task_started(self, event_data: Dict[str, Any]):
        task_id = event_data.get('task_id')
        # workflow_engine의 Task 객체에서 task_name을 가져올 수 있도록 수정
        task_name = self.workflow_engine.tasks.get(task_id).name if self.workflow_engine.tasks.get(task_id) else task_id
        self.logger.info(f"이벤트 수신: 작업 시작됨 - '{task_name}' (ID: {task_id})")

    def _on_task_completed(self, event_data: Dict[str, Any]):
        task_id = event_data.get('task_id')
        task_name = self.workflow_engine.tasks.get(task_id).name if self.workflow_engine.tasks.get(task_id) else task_id
        # result = event_data.get('result') # 필요시 결과 사용
        self.logger.info(f"이벤트 수신: 작업 완료 - '{task_name}' (ID: {task_id})")

    def _on_task_failed(self, event_data: Dict[str, Any]):
        task_id = event_data.get('task_id')
        task_name = self.workflow_engine.tasks.get(task_id).name if self.workflow_engine.tasks.get(task_id) else task_id
        error_msg = event_data.get('error')
        self.logger.error(f"이벤트 수신: 작업 실패 - '{task_name}' (ID: {task_id}) - 오류: {error_msg}")

    def apply_updated_settings(self):
        self.logger.info("변경된 설정 적용 시도...")
        try:
            # API 키 변경 시 수집기 재초기화
            self.stock_collector = StockDataCollector(self.settings.api_config)
            self.news_collector = NewsCollector(self.settings.api_config)
            # economic_calendar_collector는 현재 api_config를 사용하지 않으므로 재초기화 불필요
            self.logger.info("설정 적용 완료: 데이터 수집기(주식, 뉴스) 재초기화됨.")
            self.status_message_signal.emit("설정이 성공적으로 업데이트 및 적용되었습니다.", 3000)
        except Exception as e:
            self.logger.error(f"설정 적용 중 오류 발생: {e}", exc_info=True)
            self.status_message_signal.emit("설정 적용 중 오류 발생.", 5000)