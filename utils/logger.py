# utils/logger.py
import logging
import logging.handlers
import os
from datetime import datetime

def setup_logger(name='trading_app', level=logging.DEBUG, log_to_console=True, log_to_file=True): # 기본 레벨 DEBUG로 변경, 콘솔/파일 로깅 제어 추가
    """로거 설정"""

    # 최상위 로거 (루트 로거 또는 지정된 이름의 로거)
    logger = logging.getLogger(name if name != 'trading_app' else None) # 최상위 로거를 잡기 위해 None 사용 고려
    logger.setLevel(level) # 로거 자체의 레벨 설정이 중요

    # 핸들러가 이미 있으면 모두 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # 파일명, 라인번호 추가
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handlers_added = False

    if log_to_console:
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level) # 핸들러에도 레벨 설정
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        handlers_added = True
        # print(f"콘솔 핸들러 추가됨 (레벨: {level})") # 초기 테스트용

    if log_to_file:
        # 로그 디렉토리 생성
        log_dir = 'logs'
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            print(f"로그 디렉토리 생성 실패: {e}") # 로거 설정 전이므로 print 사용
            # 파일 로깅 없이 진행하거나, 다른 경로 시도

        # 파일 핸들러 (회전 로그)
        # 파일 이름에 name 인자를 사용하여 로거별 파일 분리 가능 (선택적)
        log_filename = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
            file_handler.setLevel(level) # 핸들러에도 레벨 설정
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            handlers_added = True
            # print(f"파일 핸들러 추가됨: {log_filename} (레벨: {level})") # 초기 테스트용
        except Exception as e:
            print(f"파일 핸들러 설정 실패 ({log_filename}): {e}")


    if not handlers_added and name is None: # 루트 로거에 핸들러가 하나도 없을 경우 기본 핸들러 추가 (최소한의 출력 보장)
        logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
        # print("기본 로깅 핸들러 (basicConfig) 사용됨.")


    # 다른 라이브러리의 로그 레벨도 제어하고 싶다면 (예: requests, pyqtgraph)
    # logging.getLogger("requests").setLevel(logging.WARNING)
    # logging.getLogger("pyqtgraph").setLevel(logging.INFO)

    # 로거 반환 (필요 시)
    return logger