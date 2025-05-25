import logging
from typing import Dict, List, Optional
from datetime import datetime
from threading import Thread
import queue

from core.workflow_engine import WorkflowEngine
from core.event_manager import EventManager
from data.collectors.stock_data_collector import StockDataCollector
from data.collectors.news_collector import NewsCollector
from analysis.models.model_manager import ModelManager
from trading.execution.order_manager import OrderManager
from trading.portfolio.portfolio_manager import PortfolioManager
from utils.scheduler import Scheduler

class MainController:
    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # 핵심 컴포넌트 초기화
        self.event_manager = EventManager()
        self.workflow_engine = WorkflowEngine(self.event_manager)
        self.scheduler = Scheduler()
        
        # 데이터 수집기
        self.stock_collector = StockDataCollector(settings.api_config)
        self.news_collector = NewsCollector(settings.api_config)
        
        # 분석 및 모델링
        self.model_manager = ModelManager()
        
        # 트레이딩
        self.order_manager = OrderManager(settings.trading_config)
        self.portfolio_manager = PortfolioManager(settings.trading_config)
        
        # 이벤트 핸들러 등록
        self._register_event_handlers()
        
        # 스케줄 작업 설정
        self._setup_scheduled_tasks()
    
    def _register_event_handlers(self):
        """이벤트 핸들러 등록"""
        self.event_manager.subscribe('data_updated', self._on_data_updated)
        self.event_manager.subscribe('signal_generated', self._on_signal_generated)
        self.event_manager.subscribe('order_filled', self._on_order_filled)
    
    def _setup_scheduled_tasks(self):
        """스케줄된 작업 설정"""
        # 실시간 데이터 수집 (1분마다)
        self.scheduler.add_job(
            self._collect_realtime_data,
            'interval',
            minutes=1,
            id='realtime_data_collection'
        )
        
        # 뉴스 데이터 수집 (5분마다)
        self.scheduler.add_job(
            self._collect_news_data,
            'interval',
            minutes=5,
            id='news_data_collection'
        )
        
        # 모델 예측 및 신호 생성 (1분마다)
        self.scheduler.add_job(
            self._generate_trading_signals,
            'interval',
            minutes=1,
            id='signal_generation'
        )
    
    def start(self):
        """애플리케이션 시작"""
        self.logger.info("애플리케이션 시작")
        self.scheduler.start()
        self.workflow_engine.start()
        
    def stop(self):
        """애플리케이션 종료"""
        self.logger.info("애플리케이션 종료")
        self.scheduler.shutdown()
        self.workflow_engine.stop()
    
    def _collect_realtime_data(self):
        """실시간 데이터 수집"""
        try:
            # 관심 종목 리스트
            symbols = self.portfolio_manager.get_watchlist()
            
            # 주식 데이터 수집
            for symbol in symbols:
                data = self.stock_collector.get_realtime_data(symbol)
                if data:
                    self.event_manager.publish('data_updated', {
                        'type': 'stock',
                        'symbol': symbol,
                        'data': data
                    })
                    
        except Exception as e:
            self.logger.error(f"실시간 데이터 수집 오류: {e}")
    
    def _collect_news_data(self):
        """뉴스 데이터 수집"""
        try:
            news_data = self.news_collector.get_latest_news()
            if news_data:
                self.event_manager.publish('data_updated', {
                    'type': 'news',
                    'data': news_data
                })
        except Exception as e:
            self.logger.error(f"뉴스 데이터 수집 오류: {e}")
    
    def _generate_trading_signals(self):
        """트레이딩 신호 생성"""
        try:
            symbols = self.portfolio_manager.get_watchlist()
            
            for symbol in symbols:
                # 모델 예측 실행
                prediction = self.model_manager.predict(symbol)
                
                if prediction:
                    signal = {
                        'symbol': symbol,
                        'action': prediction['action'],  # 'BUY', 'SELL', 'HOLD'
                        'confidence': prediction['confidence'],
                        'price_target': prediction.get('price_target'),
                        'timestamp': datetime.now()
                    }
                    
                    self.event_manager.publish('signal_generated', signal)
                    
        except Exception as e:
            self.logger.error(f"신호 생성 오류: {e}")
    
    def _on_data_updated(self, event_data):
        """데이터 업데이트 이벤트 핸들러"""
        data_type = event_data['type']
        
        if data_type == 'stock':
            symbol = event_data['symbol']
            data = event_data['data']
            # 데이터베이스에 저장
            # UI 업데이트 신호 발송
            
        elif data_type == 'news':
            news_data = event_data['data']
            # 뉴스 감성 분석 시작
            # 분석 결과를 모델에 반영
    
    def _on_signal_generated(self, signal):
        """트레이딩 신호 생성 이벤트 핸들러"""
        if signal['confidence'] > 0.7:  # 신뢰도 임계값
            # 주문 생성 및 실행
            order = self.order_manager.create_order(
                symbol=signal['symbol'],
                action=signal['action'],
                quantity=self._calculate_position_size(signal)
            )
            
            if order:
                self.order_manager.submit_order(order)
    
    def _on_order_filled(self, order_data):
        """주문 체결 이벤트 핸들러"""
        # 포트폴리오 업데이트
        self.portfolio_manager.update_position(order_data)
        
        # 성과 추적 업데이트
        # UI 업데이트 신호 발송
    
    def _calculate_position_size(self, signal):
        """포지션 크기 계산"""
        # 켈리 공식 또는 고정 비율 방식 사용
        portfolio_value = self.portfolio_manager.get_total_value()
        max_position_value = portfolio_value * self.settings.trading_config.max_position_size
        
        current_price = signal.get('current_price', 100)  # 현재 가격
        position_size = int(max_position_value / current_price)
        
        return position_size