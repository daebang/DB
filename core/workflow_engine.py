import threading
import time
import queue
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from collections.abc import Callable
from core.event_manager import EventManager
import os
import multiprocessing

class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class Task:
    id: str
    name: str
    function: Callable
    args: tuple = ()
    kwargs: dict = None
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None


class WorkflowEngine:
    """워크플로우 엔진 - 작업 스케줄링 및 실행"""
    
    def __init__(self, event_manager: EventManager, num_workers: Optional[int] = None, 
                 min_workers: int = 2, max_workers: int = 8):
        self.event_manager = event_manager
        self.task_queue = queue.PriorityQueue()
        
        # CPU 코어 수에 기반한 워커 수 자동 설정
        if num_workers is None:
            cpu_count = multiprocessing.cpu_count()
            # CPU 코어의 50% 사용 (최소 min_workers, 최대 max_workers)
            num_workers = max(min_workers, min(cpu_count // 2, max_workers))
            
        self.num_workers = num_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.workers = []
        self.running = False
        self.tasks: Dict[str, Task] = {}
        
        # 성능 모니터링을 위한 속성
        self.task_completion_times = []
        self._last_adjustment_time = time.time()
        
        print(f"WorkflowEngine 초기화: {self.num_workers}개 워커 (CPU 코어: {multiprocessing.cpu_count()})")
    
    def adjust_worker_count(self):
        """작업 부하에 따라 워커 수 동적 조정"""
        current_time = time.time()
        
        # 5분마다 조정 검토
        if current_time - self._last_adjustment_time < 300:
            return
            
        queue_size = self.task_queue.qsize()
        avg_completion_time = (sum(self.task_completion_times[-10:]) / len(self.task_completion_times[-10:]) 
                              if self.task_completion_times else 0)
        
        # 큐가 크고 처리 시간이 길면 워커 추가
        if queue_size > self.num_workers * 2 and avg_completion_time > 5:
            if self.num_workers < self.max_workers:
                self._add_worker()
                print(f"워커 추가: 현재 {self.num_workers}개")
        
        # 큐가 작고 처리 시간이 짧으면 워커 감소
        elif queue_size < self.num_workers and avg_completion_time < 1:
            if self.num_workers > self.min_workers:
                self._remove_worker()
                print(f"워커 감소: 현재 {self.num_workers}개")
        
        self._last_adjustment_time = current_time
        self.task_completion_times = self.task_completion_times[-50:]  # 최근 50개만 유지
    
    def _add_worker(self):
        """워커 추가"""
        if self.running and self.num_workers < self.max_workers:
            worker = threading.Thread(target=self._worker_loop, name=f"Worker-{self.num_workers}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.num_workers += 1
    
    def _remove_worker(self):
        """워커 제거 (안전하게 종료되도록 플래그 설정)"""
        if self.num_workers > self.min_workers:
            self.num_workers -= 1
            # 실제 워커 종료는 _worker_loop에서 처리
            
    def start(self):
        """워크플로우 엔진 시작"""
        if self.running:
            return
            
        self.running = True
        
        # 워커 스레드 시작
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"Worker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """워크플로우 엔진 중단"""
        self.running = False
        
        # 워커들이 종료될 때까지 기다림
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                self.event_manager.publish('task_warning', {'message': f"Worker thread {worker.name} did not terminate in time."})
                print(f"Warning: Worker thread {worker.name} did not terminate in time.") # Or use logger
    
    def submit_task(self, task: Task):
        """작업 제출"""
        self.tasks[task.id] = task
        # 우선순위가 높을수록 먼저 처리 (음수로 변환)
        self.task_queue.put((-task.priority, task.id))
    
    def _worker_loop(self):
        """워커 루프"""
        while self.running:
            try:
                priority, task_id = self.task_queue.get(timeout=1)
                task = self.tasks.get(task_id)
                
                if task:
                    self._execute_task(task)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"워커 오류: {e}")
    
    def _execute_task(self, task: Task):
        """작업 실행"""
        try:
            task.status = TaskStatus.RUNNING
            self.event_manager.publish('task_started', {'task_id': task.id})
            
            # 작업 실행
            kwargs = task.kwargs or {}
            task.result = task.function(*task.args, **kwargs)
            task.status = TaskStatus.COMPLETED
            
            self.event_manager.publish('task_completed', {
                'task_id': task.id,
                'result': task.result
            })
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            self.event_manager.publish('task_failed', {
                'task_id': task.id,
                'error': task.error
            })