import threading
import time
import queue
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from collections.abc import Callable
from core.event_manager import EventManager

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
    
    def __init__(self, event_manager: EventManager, num_workers: int = 3):
        self.event_manager = event_manager
        self.task_queue = queue.PriorityQueue()
        self.num_workers = num_workers
        self.workers = []
        self.running = False
        self.tasks: Dict[str, Task] = {}
        
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