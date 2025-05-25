from typing import Dict, List, Callable, Any
import threading
from datetime import datetime

class EventManager:
    """이벤트 관리자 - 발행/구독 패턴 구현"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()
        
    def subscribe(self, event_type: str, callback: Callable):
        """이벤트 구독"""
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """이벤트 구독 취소"""
        with self.lock:
            if event_type in self.subscribers:
                try:
                    self.subscribers[event_type].remove(callback)
                except ValueError:
                    pass
    
    def publish(self, event_type: str, data: Any):
        """이벤트 발행"""
        with self.lock:
            subscribers = self.subscribers.get(event_type, []).copy()
        
        # 구독자들에게 비동기적으로 이벤트 전달
        for callback in subscribers:
            try:
                callback(data)
            except Exception as e:
                print(f"이벤트 처리 오류 ({event_type}): {e}")