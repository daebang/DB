# data/collectors/stock_data_collector.py
import logging
import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

# Tiingo API 요청 제한 (무료 요금제)
TIINGO_HOURLY_LIMIT = 50  # 시간당 최대 50회
TIINGO_DAILY_LIMIT = 1000  # 일일 최대 1,000회
TIINGO_REQUEST_DELAY = 3.6  # 시간당 50회 = 평균 72초에 1회, 여유를 두어 3.6초


class StockDataCollector:
    def __init__(self, api_config: Any):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_config.tiingo_api_key if api_config and hasattr(api_config, 'tiingo_api_key') else None
        
        if not self.api_key:
            self.logger.error("Tiingo API 키가 설정되지 않았습니다. 주식 데이터 수집이 제한됩니다.")
        
        self.base_url = "https://api.tiingo.com/tiingo/daily"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}' if self.api_key else ''
        }
        
        self.realtime_quote_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = pd.Timedelta(seconds=60)  # 실시간 데이터 캐시 시간
        self._last_request_time = 0  # 마지막 요청 시간 기록
        self._request_count_hour = 0  # 시간당 요청 수
        self._request_count_day = 0  # 일일 요청 수
        self._hour_start_time = time.time()
        self._day_start_time = time.time()
        
        self.logger.info("Tiingo StockDataCollector 초기화됨.")
    
    def _check_and_wait_rate_limit(self):
        """API 요청 제한 확인 및 대기"""
        current_time = time.time()
        
        # 시간 기준 리셋 (1시간 경과)
        if current_time - self._hour_start_time >= 3600:
            self._request_count_hour = 0
            self._hour_start_time = current_time
        
        # 일일 기준 리셋 (24시간 경과)
        if current_time - self._day_start_time >= 86400:
            self._request_count_day = 0
            self._day_start_time = current_time
        
        # 제한 확인
        if self._request_count_hour >= TIINGO_HOURLY_LIMIT:
            wait_time = 3600 - (current_time - self._hour_start_time)
            self.logger.warning(f"시간당 요청 한도 도달. {wait_time:.1f}초 대기합니다.")
            time.sleep(wait_time)
            self._request_count_hour = 0
            self._hour_start_time = time.time()
        
        if self._request_count_day >= TIINGO_DAILY_LIMIT:
            wait_time = 86400 - (current_time - self._day_start_time)
            self.logger.warning(f"일일 요청 한도 도달. {wait_time:.1f}초 대기합니다.")
            time.sleep(wait_time)
            self._request_count_day = 0
            self._day_start_time = time.time()
        
        # 최소 대기 시간
        elapsed = current_time - self._last_request_time
        if elapsed < TIINGO_REQUEST_DELAY:
            wait_time = TIINGO_REQUEST_DELAY - elapsed
            self.logger.debug(f"Tiingo API 요청 제한을 위해 {wait_time:.2f}초 대기합니다.")
            time.sleep(wait_time)
        
        self._last_request_time = time.time()
        self._request_count_hour += 1
        self._request_count_day += 1
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Tiingo API 요청 실행"""
        if not self.api_key:
            self.logger.error("Tiingo API 키가 없어 요청할 수 없습니다.")
            return None
        
        self._check_and_wait_rate_limit()
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                self.logger.warning(f"Tiingo: 심볼을 찾을 수 없습니다 - {url}")
            elif response.status_code == 429:
                self.logger.error("Tiingo: API 요청 한도 초과")
            else:
                self.logger.error(f"Tiingo HTTP 오류: {e}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Tiingo 요청 오류: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = '1일', period: str = '1y') -> Optional[pd.DataFrame]:
        """
        과거 주가 데이터 조회
        
        Args:
            symbol: 주식 심볼
            timeframe: 시간 프레임 (Tiingo는 일봉만 지원, 분/시간봉은 미지원)
            period: 조회 기간
        
        Returns:
            주가 데이터 DataFrame
        """
        self.logger.info(f"Tiingo로 '{symbol}' 과거 데이터 요청 (시간대: {timeframe}, 기간: {period})")
        
        # Tiingo는 일봉 데이터만 지원
        if timeframe not in ['1일', '1주', '1개월']:
            self.logger.warning(f"Tiingo는 일봉 데이터만 지원합니다. '{timeframe}'은 지원하지 않습니다.")
            return None
        
        # 기간 계산
        end_date = datetime.now()
        if period == '1d':
            start_date = end_date - timedelta(days=1)
        elif period == '5d':
            start_date = end_date - timedelta(days=5)
        elif period == '1mo':
            start_date = end_date - timedelta(days=30)
        elif period == '3mo':
            start_date = end_date - timedelta(days=90)
        elif period == '6mo':
            start_date = end_date - timedelta(days=180)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '2y':
            start_date = end_date - timedelta(days=730)
        elif period == '5y':
            start_date = end_date - timedelta(days=1825)
        elif period == 'max':
            start_date = datetime(2000, 1, 1)  # Tiingo 데이터 시작일
        else:
            start_date = end_date - timedelta(days=365)  # 기본값 1년
        
        # API 요청
        url = f"{self.base_url}/{symbol}/prices"
        params = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'format': 'json'
        }
        
        # 주간/월간 데이터를 위한 리샘플링 파라미터
        if timeframe == '1주':
            params['resampleFreq'] = 'weekly'
        elif timeframe == '1개월':
            params['resampleFreq'] = 'monthly'
        
        try:
            data = self._make_request(url, params)
            
            if not data:
                self.logger.warning(f"Tiingo: '{symbol}' 데이터 없음")
                return None
            
            # DataFrame 변환
            df = pd.DataFrame(data)
            
            if df.empty:
                self.logger.warning(f"Tiingo: '{symbol}' 빈 데이터프레임")
                return None
            
            # 날짜 인덱스 설정
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 타임존 정보 제거 (pyqtgraph 호환성)
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            
            # 컬럼명 매핑 (Tiingo -> yfinance 형식)
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adjOpen': 'Open',  # 조정 가격 사용
                'adjHigh': 'High',
                'adjLow': 'Low',
                'adjClose': 'Close',
                'adjVolume': 'Volume'
            }, inplace=True)
            
            # 필요한 컬럼만 선택
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[required_cols]
            
            # NaN 제거
            df.dropna(subset=required_cols, how='all', inplace=True)
            
            if df.empty:
                self.logger.warning(f"Tiingo: '{symbol}' 유효한 데이터 없음")
                return None
            
            # 타임스탬프 추가
            df['timestamp'] = df.index.astype(np.int64) // 10**9  # Unix timestamp (초)
            
            # 소문자 컬럼명으로 변환 (내부 사용)
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            df_to_return = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            self.logger.info(f"Tiingo: '{symbol}' ({timeframe}) 과거 데이터 {len(df_to_return)}개 수집 완료.")
            return df_to_return
            
        except Exception as e:
            self.logger.error(f"Tiingo로 '{symbol}' 과거 데이터 수집 중 오류: {e}", exc_info=True)
            return None
    
    def get_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        현재 가격 정보 조회
        
        Args:
            symbol: 주식 심볼
        
        Returns:
            현재 가격 정보 딕셔너리
        """
        self.logger.info(f"Tiingo로 '{symbol}' 현재 정보 요청")
        
        # 캐시 확인
        cached_item = self.realtime_quote_cache.get(symbol)
        if cached_item and (pd.Timestamp.now(tz='UTC') - cached_item['retrieved_at'] < self.cache_duration):
            self.logger.debug(f"{symbol} 실시간 호가 캐시 사용 (Tiingo)")
            return cached_item
        
        # 최신 가격 데이터 요청
        url = f"{self.base_url}/{symbol}/prices"
        
        try:
            data = self._make_request(url)
            
            if not data or len(data) == 0:
                self.logger.warning(f"Tiingo: {symbol} 가격 정보 없음")
                return None
            
            # 가장 최근 데이터 사용
            latest_data = data[-1]
            
            # 메타 정보 요청 (시가총액 등을 위해)
            meta_url = f"{self.base_url}/{symbol}"
            meta_data = self._make_request(meta_url)
            
            current_price = latest_data.get('adjClose', latest_data.get('close'))
            open_price = latest_data.get('adjOpen', latest_data.get('open'))
            high_price = latest_data.get('adjHigh', latest_data.get('high'))
            low_price = latest_data.get('adjLow', latest_data.get('low'))
            volume = latest_data.get('adjVolume', latest_data.get('volume'))
            
            # 이전 거래일 종가 (전일 데이터 요청)
            yesterday = datetime.now() - timedelta(days=7)  # 주말/휴일 고려하여 7일 전부터
            prev_url = f"{self.base_url}/{symbol}/prices"
            prev_params = {
                'startDate': yesterday.strftime('%Y-%m-%d'),
                'endDate': datetime.now().strftime('%Y-%m-%d'),
                'format': 'json'
            }
            prev_data = self._make_request(prev_url, prev_params)
            
            previous_close = current_price  # 기본값
            if prev_data and len(prev_data) >= 2:
                previous_close = prev_data[-2].get('adjClose', prev_data[-2].get('close'))
            
            change = float(current_price) - float(previous_close) if current_price and previous_close else 0.0
            change_percent = (change / float(previous_close)) if previous_close and float(previous_close) != 0 else 0.0
            
            quote = {
                'symbol': symbol,
                'price': float(current_price) if current_price else 0.0,
                'open': float(open_price) if open_price else 0.0,
                'high': float(high_price) if high_price else 0.0,
                'low': float(low_price) if low_price else 0.0,
                'volume': int(volume) if volume else 0,
                'previous_close': float(previous_close) if previous_close else 0.0,
                'change': float(change),
                'change_percent': float(change_percent),
                'latest_trading_day': latest_data.get('date'),
                'market_cap': None,  # Tiingo 무료 플랜에서는 시가총액 미제공
                'retrieved_at': pd.Timestamp.now(tz='UTC')
            }
            
            self.logger.info(f"Tiingo: {symbol} 현재 정보 수집 완료 (가격: {quote['price']}).")
            self.realtime_quote_cache[symbol] = quote
            return quote
            
        except Exception as e:
            self.logger.error(f"Tiingo로 '{symbol}' 현재 정보 수집 중 오류: {e}", exc_info=True)
            return None


# 테스트용 코드 (선택적으로 실행)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    
    # 테스트를 위한 더미 API 설정
    class DummyAPIConfig:
        tiingo_api_key = "YOUR_TIINGO_API_KEY"  # 실제 Tiingo API 키로 교체
    
    if DummyAPIConfig.tiingo_api_key == "YOUR_TIINGO_API_KEY":
        print("테스트를 위해 DummyAPIConfig.tiingo_api_key에 실제 Tiingo API 키를 입력하세요.")
    else:
        collector = StockDataCollector(DummyAPIConfig())
        
        # 과거 데이터 테스트
        aapl_daily = collector.get_historical_data("AAPL", timeframe='1일', period='1mo')
        if aapl_daily is not None:
            print("\n--- AAPL 일봉 (1개월) ---")
            print(aapl_daily.tail())
        
        # Tiingo는 분봉을 지원하지 않으므로 이 테스트는 제거
        # msft_5min = collector.get_historical_data("MSFT", timeframe='5분', period='5d')
        
        # 주간 데이터 테스트
        msft_weekly = collector.get_historical_data("MSFT", timeframe='1주', period='3mo')
        if msft_weekly is not None:
            print("\n--- MSFT 주봉 (3개월) ---")
            print(msft_weekly.tail())
        
        # 실시간 호가 테스트
        tsla_quote = collector.get_realtime_quote("TSLA")
        if tsla_quote:
            print("\n--- TSLA 실시간 호가 ---")
            for k, v in tsla_quote.items():
                print(f"{k}: {v}")