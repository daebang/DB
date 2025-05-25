import os
import json
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "stock_trading"
    username: str = "trader"
    password: str = ""

@dataclass
class APIConfig:
    alpha_vantage_key: str = ""
    news_api_key: str = ""
    broker_api_key: str = ""
    broker_secret: str = ""

@dataclass
class TradingConfig:
    max_position_size: float = 0.1  # 포트폴리오의 10%
    stop_loss_percent: float = 0.05  # 5% 손절
    take_profit_percent: float = 0.15  # 15% 익절
    risk_free_rate: float = 0.02  # 무위험 수익률
    
class Settings:
    def __init__(self):
        self.config_file = "config/config.json"
        self.db_config = DatabaseConfig()
        self.api_config = APIConfig()
        self.trading_config = TradingConfig()
        self.load_config()
    
    def load_config(self):
        """설정 파일 로드"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            # 데이터베이스 설정
            if 'database' in config:
                db = config['database']
                self.db_config = DatabaseConfig(**db)
                
            # API 설정
            if 'api' in config:
                api = config['api']
                self.api_config = APIConfig(**api)
                
            # 트레이딩 설정
            if 'trading' in config:
                trading = config['trading']
                self.trading_config = TradingConfig(**trading)
    
    def save_config(self):
        """설정 파일 저장"""
        config = {
            'database': self.db_config.__dict__,
            'api': self.api_config.__dict__,
            'trading': self.trading_config.__dict__
        }
        
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)