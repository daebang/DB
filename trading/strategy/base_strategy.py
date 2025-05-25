from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class Signal:
    """거래 신호"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    quantity: int
    confidence: float
    timestamp: datetime
    strategy_name: str
    metadata: Dict = None

@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    position_type: str  # 'LONG', 'SHORT'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class BaseStrategy(ABC):
    """거래 전략 베이스 클래스"""
    
    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
        self.positions: Dict[str, Position] = {}
        self.signals_history: List[Signal] = []
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """거래 신호 생성 (추상 메서드)"""
        pass
    
    @abstractmethod
    def update_parameters(self, new_parameters: Dict):
        """전략 파라미터 업데이트 (추상 메서드)"""
        pass
    
    def add_position(self, position: Position):
        """포지션 추가"""
        self.positions[position.symbol] = position
    
    def remove_position(self, symbol: str):
        """포지션 제거"""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def update_position_price(self, symbol: str, current_price: float):
        """포지션 현재가 업데이트"""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
    
    def check_exit_conditions(self, symbol: str) -> Optional[Signal]:
        """청산 조건 확인"""
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        current_price = position.current_price
        
        # 손절 확인
        if position.stop_loss and current_price <= position.stop_loss:
            return Signal(
                symbol=symbol,
                action='SELL',
                price=current_price,
                quantity=position.quantity,
                confidence=0.9,
                timestamp=datetime.now(),
                strategy_name=self.name,
                metadata={'reason': 'stop_loss'}
            )
        
        # 익절 확인
        if position.take_profit and current_price >= position.take_profit:
            return Signal(
                symbol=symbol,
                action='SELL',
                price=current_price,
                quantity=position.quantity,
                confidence=0.9,
                timestamp=datetime.now(),
                strategy_name=self.name,
                metadata={'reason': 'take_profit'}
            )
        
        return None
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float) -> int:
        """포지션 크기 계산"""
        risk_per_trade = self.parameters.get('risk_per_trade', 0.02)  # 2% 리스크
        max_position_size = portfolio_value * risk_per_trade
        quantity = int(max_position_size / price)
        return max(1, quantity)  # 최소 1주
    
    def get_performance_summary(self) -> Dict:
        """성과 요약"""
        if not self.signals_history:
            return {}
        
        total_signals = len(self.signals_history)
        buy_signals = len([s for s in self.signals_history if s.action == 'BUY'])
        sell_signals = len([s for s in self.signals_history if s.action == 'SELL'])
        
        avg_confidence = sum(s.confidence for s in self.signals_history) / total_signals
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_confidence': avg_confidence,
            'active_positions': len(self.positions)
        }