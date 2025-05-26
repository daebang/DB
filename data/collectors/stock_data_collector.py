# data/collectors/stock_data_collector.py
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, List, Any # Any 추가

logger = logging.getLogger(__name__)

class StockDataCollector:
    def __init__(self, api_config: Any = None): # api_config는 yfinance에 직접 필요하지 않음
        self.logger = logging.getLogger(__name__)
        self.logger.info("yfinance StockDataCollector 초기화됨.")
        self.realtime_quote_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = pd.Timedelta(seconds=60) # 실시간 데이터 캐시 시간

    def get_historical_data(self, symbol: str, timeframe: str = '1일', period: str = '1y') -> Optional[pd.DataFrame]:
        self.logger.info(f"yfinance로 '{symbol}' 과거 데이터 요청 (시간대: {timeframe}, 기간: {period})")

        interval_map = {
            '1분': '1m', '5분': '5m', '15분': '15m', '30분': '30m',
            '1시간': '1h', '90분': '90m',
            '1일': '1d', '5일': '5d',
            '1주': '1wk', '1개월': '1mo', '3개월': '3mo'
        }
        yf_interval = interval_map.get(timeframe)
        if not yf_interval:
            self.logger.warning(f"yfinance에서 지원하지 않거나 매핑되지 않은 시간대: {timeframe}. '1d'로 기본 설정.")
            yf_interval = '1d'
            if period not in ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']:
                 period = '1y'

        if yf_interval.endswith('m') or yf_interval.endswith('h'):
            valid_intraday_periods = ['1d', '5d', '7d', '1mo', '3mo', '60d'] # yfinance는 1m의 경우 max 7d, 그 외 분봉은 60d까지 가능
            if period not in valid_intraday_periods:
                if yf_interval == '1m' and period not in ['1d','5d','7d']:
                    period_for_intraday = '7d'
                    self.logger.debug(f"1분봉 데이터 요청 시 기간을 '{period}'에서 '{period_for_intraday}'(으)로 자동 조정.")
                    period = period_for_intraday
                elif yf_interval != '1m' and period not in ['1d','5d','7d','1mo','3mo','60d']:
                    period_for_intraday = '60d'
                    self.logger.debug(f"분/시간봉 데이터 요청 시 기간을 '{period}'에서 '{period_for_intraday}'(으)로 자동 조정.")
                    period = period_for_intraday


        try:
            ticker = yf.Ticker(symbol)
            self.logger.debug(f"yf.Ticker('{symbol}').history(period='{period}', interval='{yf_interval}', auto_adjust=True, prepost=False) 호출")
            hist_df = ticker.history(period=period, interval=yf_interval, auto_adjust=True, prepost=False)

            if hist_df.empty:
                self.logger.warning(f"yfinance: '{symbol}' ({timeframe} @ {yf_interval}, 기간: {period}) 데이터 없음.")
                return None

            hist_df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }, inplace=True)

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in hist_df.columns for col in required_cols):
                self.logger.error(f"yfinance: '{symbol}' 데이터에 필수 컬럼 부족. 현재: {hist_df.columns.tolist()}, 필요: {required_cols}")
                return None
            
            hist_df.dropna(subset=required_cols, how='all', inplace=True)
            if hist_df.empty:
                self.logger.warning(f"yfinance: '{symbol}' 데이터가 모두 NaN 값을 포함하여 비어있게 되었습니다.")
                return None

            # 타임존 정보 제거 또는 UTC로 통일 (pyqtgraph DateAxisItem 호환성)
            if hist_df.index.tz is not None:
                hist_df.index = hist_df.index.tz_convert('UTC') # UTC로 통일
                # hist_df.index = hist_df.index.tz_localize(None) # 타임존 정보 제거

            hist_df['timestamp'] = hist_df.index.astype(np.int64) // 10**9 # UTC 기준 Unix timestamp (초)
            
            df_to_return = hist_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            self.logger.info(f"yfinance: '{symbol}' ({timeframe}) 과거 데이터 {len(df_to_return)}개 수집 완료.")
            return df_to_return

        except Exception as e:
            self.logger.error(f"yfinance로 '{symbol}' 과거 데이터 수집 중 오류: {e}", exc_info=True)
            return None

    def get_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        self.logger.info(f"yfinance로 '{symbol}' 현재 정보 요청")

        cached_item = self.realtime_quote_cache.get(symbol)
        if cached_item and (pd.Timestamp.now(tz='UTC') - cached_item['retrieved_at'] < self.cache_duration):
            self.logger.debug(f"{symbol} 실시간 호가 캐시 사용 (yfinance)")
            return cached_item

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_price = info.get('currentPrice', info.get('regularMarketPrice'))
            open_price = info.get('regularMarketOpen', info.get('open'))
            high_price = info.get('regularMarketDayHigh', info.get('dayHigh'))
            low_price = info.get('regularMarketDayLow', info.get('dayLow'))
            volume = info.get('regularMarketVolume', info.get('volume'))
            previous_close = info.get('previousClose', info.get('regularMarketPreviousClose'))

            if current_price is None and previous_close is not None:
                 current_price = previous_close

            if current_price is None:
                self.logger.warning(f"yfinance: {symbol} 현재가 정보 없음.")
                # 최근 1일치 데이터로 시도
                hist_1d = ticker.history(period="1d", interval="1d", auto_adjust=True)
                if not hist_1d.empty:
                    current_price = hist_1d['Close'].iloc[-1]
                    open_price = hist_1d['Open'].iloc[-1]
                    high_price = hist_1d['High'].iloc[-1]
                    low_price = hist_1d['Low'].iloc[-1]
                    volume = hist_1d['Volume'].iloc[-1]
                    if len(hist_1d) > 1:
                        previous_close = hist_1d['Close'].iloc[-2]
                    else: # 전일 데이터가 없으면 현재가로 이전 종가 설정 (등락 0)
                        previous_close = current_price
                else:
                    return None


            change = (float(current_price) - float(previous_close)) if current_price is not None and previous_close is not None else 0.0
            change_percent = (change / float(previous_close)) if previous_close and float(previous_close) != 0 else 0.0

            quote = {
                'symbol': info.get('symbol', symbol),
                'price': float(current_price) if current_price is not None else 0.0,
                'open': float(open_price) if open_price is not None else 0.0,
                'high': float(high_price) if high_price is not None else 0.0,
                'low': float(low_price) if low_price is not None else 0.0,
                'volume': int(volume) if volume is not None else 0,
                'previous_close': float(previous_close) if previous_close is not None else 0.0,
                'change': float(change),
                'change_percent': float(change_percent), # yfinance는 %가 아님
                'latest_trading_day': None, 
                'market_cap': info.get('marketCap'),
                'retrieved_at': pd.Timestamp.now(tz='UTC')
            }
            self.logger.info(f"yfinance: {symbol} 현재 정보 수집 완료 (가격: {quote['price']}).")
            self.realtime_quote_cache[symbol] = quote
            return quote
        except Exception as e:
            self.logger.error(f"yfinance로 '{symbol}' 현재 정보 수집 중 오류: {e}", exc_info=True)
            return None

# 테스트용 코드 (선택적으로 실행)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    collector = StockDataCollector()

    # 과거 데이터 테스트
    aapl_daily = collector.get_historical_data("AAPL", timeframe='1일', period='1mo')
    if aapl_daily is not None:
        print("\n--- AAPL 일봉 (1개월) ---")
        print(aapl_daily.tail())

    msft_5min = collector.get_historical_data("MSFT", timeframe='5분', period='5d')
    if msft_5min is not None:
        print("\n--- MSFT 5분봉 (5일) ---")
        print(msft_5min.tail())
    
    # 실시간 호가 테스트
    tsla_quote = collector.get_realtime_quote("TSLA")
    if tsla_quote:
        print("\n--- TSLA 실시간 호가 ---")
        for k, v in tsla_quote.items():
            print(f"{k}: {v}")