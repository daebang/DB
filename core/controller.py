# core/controller.py
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from threading import Thread, Lock
import queue
import pandas as pd

from PyQt5.QtCore import QObject, pyqtSignal
from utils.scheduler import Scheduler
from core.workflow_engine import WorkflowEngine, Task
from core.event_manager import EventManager
from data.collectors.stock_data_collector import StockDataCollector # yfinance 기반으로 수정된 것 사용
from data.collectors.news_collector import NewsCollector
from config.settings import Settings

# 임시 ModelManager
class TempModelManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TempModelManager")

    def predict(self, symbol: str, data: Optional[pd.DataFrame]) -> Optional[Dict]:
        self.logger.debug(f"임시 모델 매니저: {symbol} 예측 요청 (데이터 {len(data) if data is not None else 0}개)")
        if data is not None and not data.empty and 'close' in data.columns and len(data['close']) >= 5:
            if data['close'].iloc[-1] > data['close'].iloc[-5:].mean():
                return {"action": "BUY", "confidence": 0.6, "reason": "최근 상승세 (임시)", "target_price": data['close'].iloc[-1] * 1.05}
            else:
                return {"action": "SELL", "confidence": 0.6, "reason": "최근 하락세 또는 횡보 (임시)", "target_price": data['close'].iloc[-1] * 0.95}
        self.logger.warning(f"임시 모델 매니저: {symbol} 예측 위한 데이터 부족")
        return None

    def train_model_async(self, symbol: str, data: Optional[pd.DataFrame]):
        self.logger.info(f"임시 모델 매니저: {symbol} 모델 학습 비동기 요청 (데이터 {len(data) if data is not None else 0}개)")
        pass

class MainController(QObject):
    historical_data_updated_signal = pyqtSignal(str, object)
    realtime_quote_updated_signal = pyqtSignal(str, dict)
    news_data_updated_signal = pyqtSignal(str, list)
    analysis_result_updated_signal = pyqtSignal(str, dict)
    connection_status_changed_signal = pyqtSignal(str, bool)
    status_message_signal = pyqtSignal(str, int)
    task_feedback_signal = pyqtSignal(str, str)

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        self.event_manager = EventManager()
        self.workflow_engine = WorkflowEngine(self.event_manager, num_workers=3)
        self.scheduler = Scheduler()

        self.stock_collector = StockDataCollector(settings.api_config) # yfinance 기반 Collector
        self.news_collector = NewsCollector(settings.api_config)
        self.model_manager = TempModelManager()

        self._data_cache: Dict[str, Any] = {}
        self._cache_lock = Lock()
        self._active_symbol_data_requests: set[str] = set()

        self._register_event_handlers()
        self._setup_scheduled_tasks()
        self.logger.info("메인 컨트롤러 초기화 완료.")

    def _register_event_handlers(self):
        self.event_manager.subscribe('task_started', self._on_task_started)
        self.event_manager.subscribe('task_completed', self._on_task_completed)
        self.event_manager.subscribe('task_failed', self._on_task_failed)
        self.logger.debug("컨트롤러 이벤트 핸들러 등록 완료.")

    def _setup_scheduled_tasks(self):
        self.logger.info("스케줄된 작업 설정 완료 (현재는 비활성화된 예시 또는 실제 작업 추가 필요).")

    def start(self):
        self.logger.info("컨트롤러 시작: 워크플로우 엔진 가동 시도.")
        self.workflow_engine.start()
        self.connection_status_changed_signal.emit("시스템 준비됨 (컨트롤러 시작)", True)

    def stop(self):
        self.logger.info("컨트롤러 중지 요청: 워크플로우 엔진 종료 시도.")
        self.workflow_engine.stop()
        self.logger.info("워크플로우 엔진 종료됨.")
        self.connection_status_changed_signal.emit("시스템 중지됨", False)

    def request_historical_data(self, symbol: str, timeframe: str):
        request_key = f"{symbol}_{timeframe}_hist"
        task_display_name = f"{symbol}({timeframe}) 과거 데이터"

        if request_key in self._active_symbol_data_requests:
            self.status_message_signal.emit(f"{task_display_name}는 이미 요청 중입니다.", 3000)
            return

        self.task_feedback_signal.emit(task_display_name, "요청 시작")
        self._active_symbol_data_requests.add(request_key)

        task = Task(
            id=f"fetch_historical_{symbol}_{timeframe.replace(' ','_')}",
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
        data_df = None
        try:
            self.task_feedback_signal.emit(task_display_name, "데이터 요청 중...")
            
            # yfinance에 맞는 period 결정 로직
            yf_period = "1y" # 기본 1년치
            if timeframe in ['1분', '5분', '15분', '30분', '1시간', '90분']:
                yf_period = "7d" # 분/시간봉은 최근 7일치로 제한
            elif timeframe == '1주':
                yf_period = "5y"
            elif timeframe == '1개월' or timeframe == '3개월':
                yf_period = "max" # 또는 적절한 기간

            # StockDataCollector.get_historical_data 호출 (yfinance 버전)
            data_df = self.stock_collector.get_historical_data(
                symbol, 
                timeframe, 
                period=yf_period 
            )

            if data_df is not None and not data_df.empty:
                self.logger.info(f"{task_display_name}: 수집 성공 ({len(data_df)} 행).")
                with self._cache_lock:
                    self._data_cache[f"{symbol}_{timeframe}_historical"] = data_df.copy()
                
                self.logger.debug(f"{task_display_name} 전달될 데이터 (첫 3행):\n{data_df.head(3)}")
                self.historical_data_updated_signal.emit(symbol, data_df.copy())
                self.task_feedback_signal.emit(task_display_name, "로드 완료")
                
                data_df.attrs['timeframe'] = timeframe
                data_df.attrs['symbol'] = symbol
                self.request_symbol_analysis(symbol, timeframe, data_df)
                return data_df
            else:
                self.logger.warning(f"{task_display_name}: 최종 수집 실패 또는 데이터 없음.")
                self.task_feedback_signal.emit(task_display_name, "로드 실패 (데이터 없음)")
                self.historical_data_updated_signal.emit(symbol, pd.DataFrame())
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} 수집 태스크 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"오류: {str(e)[:50]}...")
            self.historical_data_updated_signal.emit(symbol, pd.DataFrame())
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
            if news_list is not None:
                self.logger.info(f"{task_display_name}: 수집 완료 ({len(news_list)}개).")
                self.news_data_updated_signal.emit(symbol_or_keywords, news_list)
                self.task_feedback_signal.emit(task_display_name, f"로드 완료 ({len(news_list)}개)")
                return news_list
            else:
                self.logger.warning(f"{task_display_name}: 뉴스 수집 실패.")
                self.task_feedback_signal.emit(task_display_name, "로드 실패")
                self.news_data_updated_signal.emit(symbol_or_keywords, [])
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} 수집 태스크 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"오류: {str(e)[:50]}...")
            self.news_data_updated_signal.emit(symbol_or_keywords, [])
            return None
        finally:
            self.logger.debug(f"태스크 종료: {task_display_name}")

    def request_symbol_analysis(self, symbol: str, timeframe: str, data_df: Optional[pd.DataFrame] = None):
        task_display_name = f"{symbol}({timeframe}) 분석"
        self.task_feedback_signal.emit(task_display_name, "요청 시작")

        if data_df is None:
            with self._cache_lock:
                data_df = self._data_cache.get(f"{symbol}_{timeframe}_historical")
        
        if data_df is None or data_df.empty:
            msg = f"{task_display_name} 실패: 분석용 데이터 없음."
            self.status_message_signal.emit(msg, 5000)
            self.logger.warning(msg)
            self.task_feedback_signal.emit(task_display_name, "데이터 부족")
            return

        if not hasattr(data_df, 'attrs') or data_df.attrs.get('timeframe') != timeframe:
            data_df.attrs['timeframe'] = timeframe
            data_df.attrs['symbol'] = symbol

        task = Task(
            id=f"analyze_{symbol}_{timeframe.replace(' ','_')}", name=task_display_name,
            function=self._task_run_analysis, args=(symbol, data_df.copy()),
            kwargs={'task_display_name': task_display_name}, priority=1
        )
        self.workflow_engine.submit_task(task)

    def _task_run_analysis(self, symbol: str, data_df: pd.DataFrame, task_display_name: str) -> Optional[Dict]:
        self.logger.debug(f"태스크 시작: {task_display_name}")
        analysis_results = {}
        try:
            self.task_feedback_signal.emit(task_display_name, "진행 중...")
            prediction = self.model_manager.predict(symbol, data_df)
            if prediction:
                analysis_results['ml_prediction'] = prediction
                self.logger.debug(f"{task_display_name} - ML 예측 완료: {prediction}")

            if analysis_results:
                self.analysis_result_updated_signal.emit(symbol, analysis_results)
                self.task_feedback_signal.emit(task_display_name, "완료")
                self.logger.info(f"{task_display_name} 완료. 결과: {analysis_results}")
                return analysis_results
            else:
                self.logger.info(f"{task_display_name}: 분석 결과 없음.")
                self.task_feedback_signal.emit(task_display_name, "결과 없음")
                return None
        except Exception as e:
            self.logger.error(f"{task_display_name} 태스크 중 예외: {e}", exc_info=True)
            self.task_feedback_signal.emit(task_display_name, f"오류: {str(e)[:50]}...")
            return None
        finally:
            self.logger.debug(f"태스크 종료: {task_display_name}")

    def get_cached_realtime_data(self, symbol: str) -> Optional[Dict]:
        with self._cache_lock:
            return self._data_cache.get(f"{symbol}_realtime")

    def get_cached_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
         with self._cache_lock:
            return self._data_cache.get(f"{symbol}_{timeframe}_historical")

    def _on_task_started(self, event_data: Dict[str, Any]):
        task_id = event_data.get('task_id')
        task_name = event_data.get('task_name', task_id)
        self.logger.info(f"이벤트 수신: 작업 시작됨 - '{task_name}' (ID: {task_id})")

    def _on_task_completed(self, event_data: Dict[str, Any]):
        task_id = event_data.get('task_id')
        task_name = event_data.get('task_name', task_id)
        self.logger.info(f"이벤트 수신: 작업 완료 - '{task_name}' (ID: {task_id})")

    def _on_task_failed(self, event_data: Dict[str, Any]):
        task_id = event_data.get('task_id')
        task_name = event_data.get('task_name', task_id)
        error_msg = event_data.get('error')
        self.logger.error(f"이벤트 수신: 작업 실패 - '{task_name}' (ID: {task_id}) - 오류: {error_msg}")

    def apply_updated_settings(self):
        self.logger.info("변경된 설정 적용 시도...")
        try:
            self.stock_collector = StockDataCollector(self.settings.api_config) # yfinance 기반으로 변경 시 api_config 불필요
            self.news_collector = NewsCollector(self.settings.api_config)
            self.logger.info("설정 적용 완료: 데이터 수집기 재초기화됨.")
            self.status_message_signal.emit("설정이 성공적으로 업데이트 및 적용되었습니다.", 3000)
        except Exception as e:
            self.logger.error(f"설정 적용 중 오류 발생: {e}", exc_info=True)
            self.status_message_signal.emit("설정 적용 중 오류 발생.", 5000)