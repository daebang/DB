from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
import uuid
import threading
from queue import Queue

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class Order:
    """주문 클래스"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    action: str = ""  # BUY, SELL
    quantity: int = 0
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    strategy_name: str = ""
    metadata: Dict = field(default_factory=dict)

class OrderManager:
    """주문 관리자"""
    
    def __init__(self, trading_config, broker_api=None):
        self.trading_config = trading_config
        self.broker_api = broker_api
        
        # 주문 관리
        self.orders: Dict[str, Order] = {}
        self.order_queue = Queue()
        self.active_orders: Dict[str, Order] = {}
        
        # 이벤트 콜백
        self.order_callbacks: List[Callable] = []
        
        # 주문 처리 스레드
        self.order_processor_running = False
        self.order_processor_thread = None
        
        # 리스크 관리
        self.daily_loss_limit = trading_config.max_position_size * 5  # 일일 손실 한도
        self.daily_pnl = 0.0
        
    def start(self):
        """주문 처리 시작"""
        if not self.order_processor_running:
            self.order_processor_running = True
            self.order_processor_thread = threading.Thread(target=self._process_orders)
            self.order_processor_thread.start()
    
    def stop(self):
        """주문 처리 중단"""
        self.order_processor_running = False
        if self.order_processor_thread:
            self.order_processor_thread.join()
    
    def create_order(self, symbol: str, action: str, quantity: int, 
                    order_type: OrderType = OrderType.MARKET, 
                    price: Optional[float] = None,
                    strategy_name: str = "") -> Optional[Order]:
        """주문 생성"""
        
        # 리스크 체크
        if not self._risk_check(symbol, action, quantity, price):
            return None
        
        order = Order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type=order_type,
            price=price,
            strategy_name=strategy_name
        )
        
        self.orders[order.id] = order
        return order
    
    def submit_order(self, order: Order) -> bool:
        """주문 제출"""
        if order.status != OrderStatus.PENDING:
            return False
        
        # 주문 큐에 추가
        self.order_queue.put(order)
        order.status = OrderStatus.SUBMITTED
        order.submitted_time = datetime.now()
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            # 브로커 API를 통한 취소 요청
            if self.broker_api:
                success = self.broker_api.cancel_order(order_id)
                if success:
                    order.status = OrderStatus.CANCELLED
                    self._remove_from_active_orders(order_id)
                    self._notify_callbacks('order_cancelled', order)
                return success
            else:
                # 시뮬레이션 모드에서는 바로 취소
                order.status = OrderStatus.CANCELLED
                self._remove_from_active_orders(order_id)
                self._notify_callbacks('order_cancelled', order)
                return True
        
        return False
    
    def _process_orders(self):
        """주문 처리 스레드"""
        while self.order_processor_running:
            try:
                if not self.order_queue.empty():
                    order = self.order_queue.get(timeout=1)
                    self._execute_order(order)
            except:
                continue
    
    def _execute_order(self, order: Order):
        """주문 실행"""
        try:
            if self.broker_api:
                # 실제 브로커 API 사용
                result = self.broker_api.submit_order(order)
                if result['success']:
                    order.status = OrderStatus.FILLED
                    order.filled_time = datetime.now()
                    order.filled_quantity = result['filled_quantity']
                    order.filled_price = result['filled_price']
                else:
                    order.status = OrderStatus.REJECTED
            else:
                # 시뮬레이션 모드
                self._simulate_order_execution(order)
            
            # 활성 주문에서 제거 (체결 또는 거부된 경우)
            if order.status in [OrderStatus.FILLED, OrderStatus.REJECTED]:
                self._remove_from_active_orders(order.id)
            
            # 콜백 호출
            self._notify_callbacks('order_executed', order)
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.metadata['error'] = str(e)
            self._notify_callbacks('order_error', order)
    
    def _simulate_order_execution(self, order: Order):
        """주문 실행 시뮬레이션"""
        # 간단한 시뮬레이션: 시장가 주문은 즉시 체결
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.FILLED
            order.filled_time = datetime.now()
            order.filled_quantity = order.quantity
            # 실제로는 현재 시장가를 가져와야 함
            order.filled_price = order.price or 100.0  # 임시 가격
        else:
            # 지정가 주문은 조건 체크 후 체결
            order.status = OrderStatus.FILLED  # 단순화
            order.filled_time = datetime.now()
            order.filled_quantity = order.quantity
            order.filled_price = order.price
    
    def _risk_check(self, symbol: str, action: str, quantity: int, price: Optional[float]) -> bool:
        """리스크 체크"""
        # 일일 손실 한도 체크
        if self.daily_pnl < -self.daily_loss_limit:
            print(f"일일 손실 한도 초과: {self.daily_pnl}")
            return False
        
        # 포지션 크기 체크
        position_value = quantity * (price or 100)
        max_position_value = self.trading_config.max_position_size * 1000000  # 가정: 100만원 포트폴리오
        
        if position_value > max_position_value:
            print(f"최대 포지션 크기 초과: {position_value} > {max_position_value}")
            return False
        
        return True
    
    def _remove_from_active_orders(self, order_id: str):
        """활성 주문에서 제거"""
        if order_id in self.active_orders:
            del self.active_orders[order_id]
    
    def _notify_callbacks(self, event_type: str, order: Order):
        """콜백 함수들에 이벤트 알림"""
        for callback in self.order_callbacks:
            try:
                callback(event_type, order)
            except Exception as e:
                print(f"콜백 실행 오류: {e}")
    
    def add_callback(self, callback: Callable):
        """콜백 함수 추가"""
        self.order_callbacks.append(callback)
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """주문 상태 조회"""
        if order_id in self.orders:
            return self.orders[order_id].status
        return None
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """종목별 주문 조회"""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """상태별 주문 조회"""
        return [order for order in self.orders.values() if order.status == status]