# analysis/technical/indicators.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
# from scipy import stats # slope 함수에 필요
# import scipy.optimize # fourier, sine 함수에 필요
# from scipy.optimize import OptimizeWarning # fourier, sine 함수에 필요
# import warnings # fourier, sine 함수에 필요

logger = logging.getLogger(__name__)

# --- 이동평균 ---
def add_sma(df: pd.DataFrame, window: int, price_col: str = 'close', output_col_name: Optional[str] = None) -> pd.DataFrame:
    """DataFrame에 단순 이동 평균(SMA) 컬럼을 추가합니다."""
    if price_col not in df.columns:
        logger.error(f"'{price_col}' 컬럼이 DataFrame에 없습니다. SMA를 계산할 수 없습니다.")
        return df
    if output_col_name is None:
        output_col_name = f'SMA_{window}'
    
    df[output_col_name] = df[price_col].rolling(window=window, min_periods=1).mean() # min_periods=1 추가하여 초반 NaN 줄임
    logger.debug(f"'{output_col_name}' 컬럼 추가됨 (window: {window})")
    return df

def add_ema(df: pd.DataFrame, window: int, price_col: str = 'close', output_col_name: Optional[str] = None) -> pd.DataFrame:
    """DataFrame에 지수 이동 평균(EMA) 컬럼을 추가합니다."""
    if price_col not in df.columns:
        logger.error(f"'{price_col}' 컬럼이 DataFrame에 없습니다. EMA를 계산할 수 없습니다.")
        return df
    if output_col_name is None:
        output_col_name = f'EMA_{window}'
        
    df[output_col_name] = df[price_col].ewm(span=window, adjust=False, min_periods=1).mean() # min_periods=1 추가
    logger.debug(f"'{output_col_name}' 컬럼 추가됨 (span: {window})")
    return df

# --- RSI ---
def add_rsi(df: pd.DataFrame, window: int = 14, price_col: str = 'close', output_col_name: Optional[str] = None) -> pd.DataFrame:
    """DataFrame에 상대강도지수(RSI) 컬럼을 추가합니다. (Wilder's Smoothing 방식)"""
    if price_col not in df.columns:
        logger.error(f"'{price_col}' 컬럼이 DataFrame에 없습니다. RSI를 계산할 수 없습니다.")
        return df
    if output_col_name is None:
        output_col_name = f'RSI_{window}'

    delta = df[price_col].diff(1)
    
    gain = delta.copy()
    gain[gain < 0] = 0.0
    
    loss = delta.copy()
    loss[loss > 0] = 0.0
    loss = abs(loss)

    # 첫 번째 평균은 단순 이동 평균으로 계산 (Wilder's method)
    first_avg_gain = gain[:window].mean()
    first_avg_loss = loss[:window].mean()
    
    # Wilder's smoothing을 위한 초기값 설정
    avg_gain = gain.copy()
    avg_loss = loss.copy()
    
    avg_gain.iloc[window-1] = first_avg_gain
    avg_loss.iloc[window-1] = first_avg_loss
    
    # Wilder's smoothing 적용
    for i in range(window, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window - 1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window - 1) + loss.iloc[i]) / window
    
    # 첫 window-1 개는 NaN으로 설정
    avg_gain[:window-1] = np.nan
    avg_loss[:window-1] = np.nan
    
    # RSI 계산
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # 0으로 나누기 처리
    rsi[avg_loss == 0] = 100
    
    df[output_col_name] = rsi
    df[output_col_name] = df[output_col_name].fillna(50)  # 초기 NaN 값을 중립값 50으로

    logger.debug(f"'{output_col_name}' 컬럼 추가됨 (window: {window})")
    return df
    
# --- MACD ---
def add_macd(df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9, price_col: str = 'close') -> pd.DataFrame:
    """DataFrame에 MACD 관련 컬럼들(MACD, Signal, Histogram)을 추가합니다."""
    if price_col not in df.columns:
        logger.error(f"'{price_col}' 컬럼이 DataFrame에 없습니다. MACD를 계산할 수 없습니다.")
        return df

    ema_short_col = f'EMA_{short_window}'
    ema_long_col = f'EMA_{long_window}'
    macd_col = 'MACD'
    signal_col = 'MACD_Signal'
    histogram_col = 'MACD_Histogram'

    # EMA 계산 시 기존 add_ema 함수 활용 또는 직접 계산
    ema_short = df[price_col].ewm(span=short_window, adjust=False, min_periods=1).mean()
    ema_long = df[price_col].ewm(span=long_window, adjust=False, min_periods=1).mean()
    
    df[macd_col] = ema_short - ema_long
    df[signal_col] = df[macd_col].ewm(span=signal_window, adjust=False, min_periods=1).mean()
    df[histogram_col] = df[macd_col] - df[signal_col]
    
    logger.debug(f"MACD 관련 컬럼들 ('{macd_col}', '{signal_col}', '{histogram_col}') 추가됨")
    return df

# --- Bollinger Bands ---
def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std_dev: int = 2, price_col: str = 'close') -> pd.DataFrame:
    """DataFrame에 볼린저 밴드 관련 컬럼들(Upper, Middle, Lower)을 추가합니다."""
    if price_col not in df.columns:
        logger.error(f"'{price_col}' 컬럼이 DataFrame에 없습니다. 볼린저 밴드를 계산할 수 없습니다.")
        return df

    middle_band_col = f'BB_Middle_{window}'
    upper_band_col = f'BB_Upper_{window}'
    lower_band_col = f'BB_Lower_{window}'
    
    # SMA (중간 밴드)
    df[middle_band_col] = df[price_col].rolling(window=window, min_periods=1).mean()
    
    # 표준편차
    std_dev = df[price_col].rolling(window=window, min_periods=1).std()
    
    df[upper_band_col] = df[middle_band_col] + (std_dev * num_std_dev)
    df[lower_band_col] = df[middle_band_col] - (std_dev * num_std_dev)
    
    logger.debug(f"볼린저 밴드 관련 컬럼들 ('{upper_band_col}', '{middle_band_col}', '{lower_band_col}') 추가됨")
    return df

# --- (선택) 사용자가 제공한 다른 지표 함수들을 이와 유사한 방식으로 수정하여 추가 가능 ---
# 예: Momentum (사용자 코드 기반)
def add_momentum(df: pd.DataFrame, window: int, price_col: str = 'close', output_col_name: Optional[str] = None) -> pd.DataFrame:
    if price_col not in df.columns:
        logger.error(f"'{price_col}' 컬럼이 DataFrame에 없습니다. Momentum을 계산할 수 없습니다.")
        return df
    if output_col_name is None:
        output_col_name = f'Momentum_{window}'
    
    # .values를 사용하지 않고 Series 간의 연산으로 변경하여 인덱스 문제를 피함
    df[output_col_name] = df[price_col].diff(periods=window) 
    # 과거 코드: prices.close.iloc[periods[i]:]-prices.close.iloc[:-periods[i]].values
    # df.index를 기준으로 정렬되어 있다고 가정할 때 diff(period)가 동일한 효과
    logger.debug(f"'{output_col_name}' 컬럼 추가됨 (window: {window})")
    return df


# --- 종합적인 지표 추가 함수 ---
def add_all_selected_indicators(df: pd.DataFrame, price_col: str = 'close', 
                                sma_windows: List[int] = [20, 60],
                                ema_windows: List[int] = [12, 26],
                                rsi_window: int = 14,
                                macd_params: tuple = (12, 26, 9), # short, long, signal
                                bb_params: tuple = (20, 2) # window, std_dev
                                ) -> pd.DataFrame:
    """선택된 주요 기술적 지표들을 DataFrame에 추가합니다."""
    if df.empty:
        logger.warning("입력 DataFrame이 비어 있어 기술적 지표를 추가할 수 없습니다.")
        return df
    
    df_processed = df.copy()

    for window in sma_windows:
        df_processed = add_sma(df_processed, window=window, price_col=price_col)
    
    for window in ema_windows:
        df_processed = add_ema(df_processed, window=window, price_col=price_col)
        
    if rsi_window > 0:
        df_processed = add_rsi(df_processed, window=rsi_window, price_col=price_col)
        
    if macd_params and len(macd_params) == 3:
        df_processed = add_macd(df_processed, short_window=macd_params[0], long_window=macd_params[1], signal_window=macd_params[2], price_col=price_col)

    if bb_params and len(bb_params) == 2:
        df_processed = add_bollinger_bands(df_processed, window=bb_params[0], num_std_dev=bb_params[1], price_col=price_col)
        
    # 필요한 경우 여기에 사용자의 다른 지표 함수 호출 추가
    # 예: df_processed = add_momentum(df_processed, window=10, price_col=price_col)

    logger.info("선택된 기술적 지표들 계산 완료.")
    return df_processed


if __name__ == '__main__':
    # 테스트용 코드
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
                                '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
                                '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15',
                                '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20',
                                '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25',
                                '2023-01-26', '2023-01-27', '2023-01-28', '2023-01-29', '2023-01-30']),
        'open':  [10, 11, 12, 11, 13, 15, 14, 16, 17, 18, 16, 15, 17, 19, 20, 18, 17, 16, 18, 20, 22, 21, 23, 22, 24, 25, 24, 23, 25, 26],
        'high':  [11, 12, 13, 13, 15, 16, 16, 18, 19, 18, 17, 18, 20, 20, 20, 19, 18, 19, 21, 22, 23, 23, 24, 25, 25, 26, 25, 24, 26, 27],
        'low':   [9,  10, 11, 10, 12, 14, 13, 15, 16, 16, 14, 14, 16, 18, 18, 16, 15, 15, 17, 19, 20, 20, 21, 22, 23, 24, 22, 22, 24, 25],
        'close': [10, 12, 11, 13, 15, 14, 16, 17, 18, 16, 15, 17, 19, 20, 18, 17, 16, 18, 20, 22, 21, 23, 22, 24, 25, 24, 23, 22, 25, 27],
        'volume':[100,120,110,130,150,140,160,170,180,160,150,170,190,200,180,170,160,180,200,220,210,230,220,240,250,240,230,220,250,270]

    }
    sample_df = pd.DataFrame(data)
    sample_df.set_index('date', inplace=True)

    # 개별 지표 테스트
    # df_with_sma = add_sma(sample_df.copy(), window=5)
    # print("\nSMA:\n", df_with_sma.tail())
    # df_with_ema = add_ema(sample_df.copy(), window=5)
    # print("\nEMA:\n", df_with_ema.tail())
    # df_with_rsi = add_rsi(sample_df.copy(), window=6)
    # print("\nRSI:\n", df_with_rsi.tail())
    # df_with_macd = add_macd(sample_df.copy(), short_window=5, long_window=10, signal_window=4)
    # print("\nMACD:\n", df_with_macd.tail())
    # df_with_bb = add_bollinger_bands(sample_df.copy(), window=5, num_std_dev=2)
    # print("\nBollinger Bands:\n", df_with_bb.tail())
    # df_with_mom = add_momentum(sample_df.copy(), window=5)
    # print("\nMomentum:\n", df_with_mom.tail())


    # 모든 선택된 지표 추가 테스트
    df_with_all_indicators = add_all_selected_indicators(sample_df.copy())
    print("\n=== 모든 선택된 기술적 지표 계산 결과 ===")
    print(df_with_all_indicators.to_string())