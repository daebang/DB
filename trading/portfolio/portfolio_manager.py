import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

class PortfolioManager:
    """포트폴리오 관리자"""
    
    def __init__(self, trading_config, database_manager=None):
        self.trading_config = trading_config
        self.db_manager = database_manager
        self.positions: Dict[str, Dict] = {}
        self.cash = 100000.0  # 초기 현금
        self.watchlist = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        # 포트폴리오 로드
        self.load_portfolio()
    
    def load_portfolio(self):
        """포트폴리오 로드"""
        if self.db_manager:
            try:
                portfolio_df = self.db_manager.get_portfolio()
                for _, row in portfolio_df.iterrows():
                    self.positions[row['symbol']] = {
                        'quantity': row['quantity'],
                        'avg_price': row['avg_price'],
                        'current_price': row['avg_price'],  # 실시간 업데이트 필요
                        'last_updated': row['last_updated']
                    }
            except Exception as e:
                print(f"포트폴리오 로드 오류: {e}")
    
    def add_position(self, symbol: str, quantity: int, price: float):
        """포지션 추가"""
        if symbol in self.positions:
            # 기존 포지션 업데이트 (평균 단가 계산)
            existing = self.positions[symbol]
            total_quantity = existing['quantity'] + quantity
            total_value = (existing['quantity'] * existing['avg_price']) + (quantity * price)
            avg_price = total_value / total_quantity
            
            self.positions[symbol].update({
                'quantity': total_quantity,
                'avg_price': avg_price,
                'last_updated': datetime.now()
            })
        else:
            # 새 포지션 생성
            self.positions[symbol] = {
                'quantity': quantity,
                'avg_price': price,
                'current_price': price,
                'last_updated': datetime.now()
            }
        
        # 현금 업데이트
        self.cash -= quantity * price
        
        # 데이터베이스 업데이트
        if self.db_manager:
            self.db_manager.update_portfolio(symbol, self.positions[symbol]['quantity'], 
                                           self.positions[symbol]['avg_price'])
    
    def remove_position(self, symbol: str, quantity: int, price: float):
        """포지션 제거"""
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if quantity >= position['quantity']:
                # 전체 포지션 청산
                self.cash += position['quantity'] * price
                del self.positions[symbol]
                
                # 데이터베이스 업데이트
                if self.db_manager:
                    self.db_manager.update_portfolio(symbol, 0, 0)
            else:
                # 부분 청산
                position['quantity'] -= quantity
                self.cash += quantity * price
                position['last_updated'] = datetime.now()
                
                # 데이터베이스 업데이트
                if self.db_manager:
                    self.db_manager.update_portfolio(symbol, position['quantity'], position['avg_price'])
    
    def update_position(self, order_data):
        """주문 체결에 따른 포지션 업데이트"""
        symbol = order_data['symbol']
        action = order_data['action']
        quantity = order_data['filled_quantity']
        price = order_data['filled_price']
        
        if action == 'BUY':
            self.add_position(symbol, quantity, price)
        elif action == 'SELL':
            self.remove_position(symbol, quantity, price)
    
    def update_current_prices(self, price_data: Dict[str, float]):
        """현재가 업데이트"""
        for symbol, price in price_data.items():
            if symbol in self.positions:
                self.positions[symbol]['current_price'] = price
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """특정 종목 포지션 조회"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """모든 포지션 조회"""
        return self.positions.copy()
    
    def get_portfolio_value(self) -> float:
        """포트폴리오 총 가치"""
        total_value = self.cash
        
        for position in self.positions.values():
            total_value += position['quantity'] * position['current_price']
        
        return total_value
    
    def get_total_value(self) -> float:
        """총 자산 가치 (포트폴리오 가치와 동일)"""
        return self.get_portfolio_value()
    
    def get_position_pnl(self, symbol: str) -> Dict:
        """포지션 손익"""
        if symbol not in self.positions:
            return {'pnl': 0, 'pnl_percent': 0}
        
        position = self.positions[symbol]
        unrealized_pnl = (position['current_price'] - position['avg_price']) * position['quantity']
        pnl_percent = (position['current_price'] / position['avg_price'] - 1) * 100
        
        return {
            'pnl': unrealized_pnl,
            'pnl_percent': pnl_percent
        }
    
    def get_total_pnl(self) -> Dict:
        """총 손익"""
        total_pnl = 0
        total_invested = 0
        
        for symbol, position in self.positions.items():
            pnl_data = self.get_position_pnl(symbol)
            total_pnl += pnl_data['pnl']
            total_invested += position['avg_price'] * position['quantity']
        
        pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'pnl_percent': pnl_percent,
            'total_invested': total_invested
        }
    
    def get_watchlist(self) -> List[str]:
        """관심 종목 리스트"""
        return self.watchlist.copy()
    
    def add_to_watchlist(self, symbol: str):
        """관심 종목 추가"""
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
    
    def remove_from_watchlist(self, symbol: str):
        """관심 종목 제거"""
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
    
    def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약"""
        total_value = self.get_portfolio_value()
        pnl_data = self.get_total_pnl()
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'invested_value': total_value - self.cash,
            'positions_count': len(self.positions),
            'total_pnl': pnl_data['total_pnl'],
            'pnl_percent': pnl_data['pnl_percent'],
            'largest_position': self._get_largest_position(),
            'allocation': self._get_allocation()
        }
    
    def _get_largest_position(self) -> Dict:
        """최대 포지션"""
        if not self.positions:
            return {}
        
        largest_symbol = max(self.positions.keys(), 
                           key=lambda x: self.positions[x]['quantity'] * self.positions[x]['current_price'])
        
        position = self.positions[largest_symbol]
        value = position['quantity'] * position['current_price']
        
        return {
            'symbol': largest_symbol,
            'value': value,
            'percentage': value / self.get_portfolio_value() * 100
        }
    
    def _get_allocation(self) -> Dict[str, float]:
        """자산 배분"""
        total_value = self.get_portfolio_value()
        allocation = {'CASH': self.cash / total_value * 100}
        
        for symbol, position in self.positions.items():
            value = position['quantity'] * position['current_price']
            allocation[symbol] = value / total_value * 100
        
        return allocation