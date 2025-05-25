from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import logging

class Scheduler:
    """작업 스케줄러"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """스케줄러 시작"""
        try:
            self.scheduler.start()
            self.logger.info("스케줄러 시작됨")
        except Exception as e:
            self.logger.error(f"스케줄러 시작 오류: {e}")
    
    def shutdown(self):
        """스케줄러 종료"""
        try:
            self.scheduler.shutdown()
            self.logger.info("스케줄러 종료됨")
        except Exception as e:
            self.logger.error(f"스케줄러 종료 오류: {e}")
    
    def add_job(self, func, trigger_type, **kwargs):
        """작업 추가"""
        try:
            if trigger_type == 'interval':
                trigger = IntervalTrigger(**{k: v for k, v in kwargs.items() 
                                          if k in ['weeks', 'days', 'hours', 'minutes', 'seconds']})
            elif trigger_type == 'cron':
                trigger = CronTrigger(**{k: v for k, v in kwargs.items() 
                                       if k in ['year', 'month', 'day', 'week', 'day_of_week', 
                                              'hour', 'minute', 'second']})
            else:
                raise ValueError(f"지원하지 않는 트리거 타입: {trigger_type}")
            
            job_id = kwargs.get('id', f"{func.__name__}_{trigger_type}")
            
            self.scheduler.add_job(
                func=func,
                trigger=trigger,
                id=job_id,
                replace_existing=True
            )
            
            self.logger.info(f"작업 추가됨: {job_id}")
            
        except Exception as e:
            self.logger.error(f"작업 추가 오류: {e}")
    
    def remove_job(self, job_id):
        """작업 제거"""
        try:
            self.scheduler.remove_job(job_id)
            self.logger.info(f"작업 제거됨: {job_id}")
        except Exception as e:
            self.logger.error(f"작업 제거 오류: {e}")
    
    def get_jobs(self):
        """등록된 작업 목록 반환"""
        return self.scheduler.get_jobs()
# 누락된 핵심 모듈들 구현