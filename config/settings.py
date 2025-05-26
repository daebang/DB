# config/settings.py
import sys
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict # asdict 추가
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "stock_trading_db" # 이름 변경
    username: str = "trader_user" # 이름 변경
    password: str = "your_password" # 기본값 설정 (보안 주의)

@dataclass
class APIConfig:
    alpha_vantage_key: str = ""
    news_api_key: str = ""
    broker_api_key: str = "" # 실제 거래용 API
    broker_secret: str = ""  # 실제 거래용 Secret
    # 추가 API 키 (예: FRED, 경제지표용 등)
    # fred_api_key: str = ""

@dataclass
class TradingConfig:
    max_position_size: float = 0.1  # 포트폴리오의 % (0.0 ~ 1.0)
    stop_loss_percent: float = 0.05      # % (0.0 ~ 1.0)
    take_profit_percent: float = 0.15    # % (0.0 ~ 1.0)
    risk_free_rate: float = 0.02         # % (0.0 ~ 1.0)
    default_order_quantity: int = 1      # 기본 주문 수량 (주식 수)
    # slippage_allowance: float = 0.001  # 슬리피지 허용 범위 (0.1%)

@dataclass
class AppConfig: # 애플리케이션 일반 설정
    log_level: str = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
    default_theme: str = "dark" # "dark", "light"
    # ... 기타 앱 설정 ...

class Settings:
    CONFIG_FILE_NAME = "config.json" # 기본 설정 파일 이름
    DEFAULT_CONFIG_DIR_NAME = "user_config" # 사용자 설정 디렉토리 이름

    def __init__(self, base_path: Optional[str] = None):
        # base_path: 애플리케이션 루트 경로 또는 쓰기 가능한 경로
        if base_path is None:
            # 스크립트 실행 위치를 기준으로 하거나, 사용자 디렉토리 사용
            # 여기서는 실행 파일이 있는 곳에 user_config 디렉토리 생성 시도
            try:
                # PyInstaller 등으로 패키징되었을 때 sys.executable 사용
                if getattr(sys, 'frozen', False):
                    app_root = os.path.dirname(sys.executable)
                else:
                    # 일반 스크립트 실행 시
                    app_root = os.path.dirname(os.path.abspath(__file__)) # settings.py 위치 기준
                    app_root = os.path.abspath(os.path.join(app_root, "..")) # 프로젝트 루트로 가정
            except NameError: # sys 임포트 안된 경우 등 (테스트 환경)
                 app_root = os.getcwd()

            self.config_dir = os.path.join(app_root, self.DEFAULT_CONFIG_DIR_NAME)
        else:
            self.config_dir = os.path.join(base_path, self.DEFAULT_CONFIG_DIR_NAME)

        self.config_file_path = os.path.join(self.config_dir, self.CONFIG_FILE_NAME)

        # 기본 설정값으로 초기화
        self.db_config = DatabaseConfig()
        self.api_config = APIConfig()
        self.trading_config = TradingConfig()
        self.app_config = AppConfig()

        self.ensure_config_dir_exists()
        if not self.load_config(self.config_file_path): # 기본 경로에서 로드 시도
            logger.warning(f"기본 설정 파일({self.config_file_path})을 찾을 수 없거나 로드에 실패했습니다. 기본값으로 새 설정을 저장합니다.")
            self.save_config(self.config_file_path) # 기본값으로 파일 생성


    def ensure_config_dir_exists(self):
        if not os.path.exists(self.config_dir):
            try:
                os.makedirs(self.config_dir, exist_ok=True)
                logger.info(f"설정 디렉토리 생성: {self.config_dir}")
            except OSError as e:
                logger.error(f"설정 디렉토리 생성 실패 ({self.config_dir}): {e}", exc_info=True)
                # 대체 경로 사용 또는 에러 처리


    def get_config_directory(self) -> str:
        """설정 파일이 저장될 디렉토리 경로를 반환합니다."""
        return self.config_dir


    def load_config(self, file_path: Optional[str] = None) -> bool:
        """
        설정 파일 로드. file_path가 None이면 기본 경로 사용.
        성공 시 True, 실패 시 False 반환.
        """
        target_file = file_path or self.config_file_path
        logger.info(f"설정 파일 로드 시도: {target_file}")
         
        if not os.path.exists(target_file):
            logger.debug(f"설정 파일 없음: {target_file}")
            return False

        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 각 설정 클래스에 데이터 할당 (존재하는 키만 업데이트)
            if 'database' in config_data:
                for key, value in config_data['database'].items():
                    if hasattr(self.db_config, key):
                        setattr(self.db_config, key, value)
            if 'api' in config_data:
                for key, value in config_data['api'].items():
                    if hasattr(self.api_config, key):
                        setattr(self.api_config, key, value)
            if 'trading' in config_data:
                for key, value in config_data['trading'].items():
                    if hasattr(self.trading_config, key):
                        setattr(self.trading_config, key, value)
            if 'application' in config_data:
                for key, value in config_data['application'].items():
                    if hasattr(self.app_config, key):
                        setattr(self.app_config, key, value)

            logger.info(f"설정 로드 완료: {target_file}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"설정 파일 JSON 디코딩 오류 ({target_file}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"설정 파일 로드 중 예외 발생 ({target_file}): {e}", exc_info=True)
            return False


    def save_config(self, file_path: Optional[str] = None) -> bool:
        """
        현재 설정을 파일에 저장. file_path가 None이면 기본 경로 사용.
        성공 시 True, 실패 시 False 반환.
        """
        target_file = file_path or self.config_file_path
        self.ensure_config_dir_exists() # 저장 전 디렉토리 존재 확인

        config_to_save = {
            'database': asdict(self.db_config),
            'api': asdict(self.api_config),
            'trading': asdict(self.trading_config),
            'application': asdict(self.app_config)
        }

        try:
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4, ensure_ascii=False)
            logger.info(f"설정 저장 완료: {target_file}")
            return True
        except Exception as e:
            logger.error(f"설정 파일 저장 중 예외 발생 ({target_file}): {e}", exc_info=True)
            return False

# 사용 예시 (main.py 등에서)
# import sys
# settings = Settings(base_path=os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else None)
# print(settings.api_config.alpha_vantage_key)