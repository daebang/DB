import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Attention
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BaseTimeframePredictionModel:
    """기간별 예측 모델의 기본 클래스"""
    
    def __init__(self, model_name: str, timeframe: str):
        self.model_name = model_name
        self.timeframe = timeframe
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.last_train_date = None
        
    def prepare_features(self, data: pd.DataFrame, 
                        news_sentiment: Optional[pd.DataFrame] = None,
                        economic_indicators: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """예측을 위한 특징 준비"""
        features = pd.DataFrame(index=data.index)
        
        # 기술적 지표 특징
        if 'close' in data.columns:
            # 가격 변화율
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            # 이동평균 관련
            for period in [5, 10, 20, 50]:
                if f'SMA_{period}' in data.columns:
                    features[f'sma_{period}_ratio'] = data['close'] / data[f'SMA_{period}']
                    
            # RSI
            if 'RSI_14' in data.columns:
                features['rsi'] = data['RSI_14']
                features['rsi_oversold'] = (data['RSI_14'] < 30).astype(int)
                features['rsi_overbought'] = (data['RSI_14'] > 70).astype(int)
            
            # MACD
            if 'MACD' in data.columns:
                features['macd'] = data['MACD']
                features['macd_signal'] = data['MACD_Signal']
                features['macd_diff'] = data['MACD'] - data['MACD_Signal']
            
            # 볼린저 밴드
            if 'BB_Upper_20' in data.columns:
                features['bb_position'] = (data['close'] - data['BB_Lower_20']) / (data['BB_Upper_20'] - data['BB_Lower_20'])
                
            # 거래량 관련
            if 'volume' in data.columns:
                features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
                features['volume_trend'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
        
        # 뉴스 감성 특징
        if news_sentiment is not None and not news_sentiment.empty:
            # 날짜별 감성 점수 집계
            daily_sentiment = news_sentiment.groupby(pd.Grouper(freq='D')).agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'sentiment_label': lambda x: (x == 'positive').sum() - (x == 'negative').sum()
            })
            daily_sentiment.columns = ['sentiment_mean', 'sentiment_std', 'news_count', 'sentiment_balance']
            features = features.join(daily_sentiment, how='left')
            features.fillna(0, inplace=True)
        
        # 경제 지표 특징
        if economic_indicators is not None and not economic_indicators.empty:
            # 중요 경제 지표 통합
            econ_features = self._process_economic_indicators(economic_indicators)
            features = features.join(econ_features, how='left')
        
        # 시간 관련 특징
        features['day_of_week'] = pd.to_datetime(features.index).dayofweek
        features['month'] = pd.to_datetime(features.index).month
        features['quarter'] = pd.to_datetime(features.index).quarter
        
        return features.dropna()
    
    def _process_economic_indicators(self, economic_data: pd.DataFrame) -> pd.DataFrame:
        """경제 지표 처리 및 특징 생성"""
        econ_features = pd.DataFrame()
        
        # 주요 지표별 처리
        important_indicators = {
            'interest_rate': ['Federal Funds Rate', 'Fed Interest Rate Decision'],
            'inflation': ['CPI', 'Core CPI', 'PPI'],
            'employment': ['Non-Farm Payrolls', 'Unemployment Rate'],
            'gdp': ['GDP Growth Rate', 'GDP'],
            'consumer': ['Consumer Confidence', 'Retail Sales']
        }
        
        for category, indicators in important_indicators.items():
            category_data = economic_data[economic_data['event'].str.contains('|'.join(indicators), case=False, na=False)]
            if not category_data.empty:
                # 실제값과 예상값의 차이 (서프라이즈)
                surprises = []
                for _, row in category_data.iterrows():
                    if pd.notna(row['actual']) and pd.notna(row['forecast']):
                        surprises.append((row['actual'] - row['forecast']) / abs(row['forecast']))
                
                if surprises:
                    econ_features[f'{category}_surprise'] = np.mean(surprises)
                    econ_features[f'{category}_volatility'] = np.std(surprises) if len(surprises) > 1 else 0
        
        return econ_features
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """모델 학습 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """예측 수행 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def calculate_prediction_ranges(self, base_prediction: float, 
                                  historical_volatility: float,
                                  confidence_level: float) -> Tuple[float, float]:
        """예측 범위 계산"""
        # 신뢰도와 변동성을 고려한 범위 계산
        range_multiplier = 1.96 if confidence_level > 0.8 else 1.645  # 95% or 90% CI
        range_width = historical_volatility * range_multiplier * (1 - confidence_level + 0.5)
        
        upper_bound = base_prediction * (1 + range_width)
        lower_bound = base_prediction * (1 - range_width)
        
        return lower_bound, upper_bound


class ShortTermModel(BaseTimeframePredictionModel):
    """단기 예측 모델 (당일 ~ 수일)"""
    
    def __init__(self):
        super().__init__("ShortTermLSTM", "short")
        self.sequence_length = 20  # 최근 20일 데이터 사용
        self.prediction_days = [1, 2, 3]  # 1일, 2일, 3일 예측
        
    def build_model(self, input_shape: Tuple):
        """LSTM 기반 단기 예측 모델 구축"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(len(self.prediction_days))  # 다중 기간 예측
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 데이터 준비"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(features) - max(self.prediction_days)):
            # 입력 시퀀스
            seq = features.iloc[i-self.sequence_length:i].values
            sequences.append(seq)
            
            # 다중 기간 타겟
            target_values = []
            for days in self.prediction_days:
                future_return = (target.iloc[i+days-1] / target.iloc[i-1]) - 1
                target_values.append(future_return)
            targets.append(target_values)
        
        return np.array(sequences), np.array(targets)
    
    def train(self, data: pd.DataFrame, features: pd.DataFrame):
        """단기 예측 모델 학습"""
        # 특징 정규화
        features_scaled = self.scaler.fit_transform(features)
        features_df = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
        
        # 시퀀스 데이터 준비
        X, y = self.prepare_sequences(features_df, data['close'])
        
        # 학습/검증 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 모델 구축 및 학습
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.last_train_date = datetime.now()
        logger.info(f"단기 예측 모델 학습 완료. 최종 검증 손실: {history.history['val_loss'][-1]:.4f}")
        
    def predict(self, current_data: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """단기 예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 특징 정규화
        features_scaled = self.scaler.transform(features.iloc[-self.sequence_length:])
        sequence = features_scaled.reshape(1, self.sequence_length, -1)
        
        # 예측
        predictions = self.model.predict(sequence, verbose=0)[0]
        current_price = current_data['close'].iloc[-1]
        
        results = {}
        for i, days in enumerate(self.prediction_days):
            predicted_return = predictions[i]
            predicted_price = current_price * (1 + predicted_return)
            
            # 과거 변동성 계산
            historical_returns = current_data['close'].pct_change().dropna()
            volatility = historical_returns.rolling(20).std().iloc[-1]
            
            # 예측 범위 계산
            lower, upper = self.calculate_prediction_ranges(
                predicted_price, 
                volatility * np.sqrt(days),  # 시간에 따른 변동성 조정
                0.85  # 단기 예측 신뢰도
            )
            
            results[f'{days}일'] = {
                'predicted_price': predicted_price,
                'predicted_return': predicted_return * 100,  # 퍼센트로 변환
                'price_range': (lower, upper),
                'confidence': 0.85 - (days * 0.05)  # 기간이 길수록 신뢰도 감소
            }
        
        return results


class MidTermModel(BaseTimeframePredictionModel):
    """중기 예측 모델 (1주일 ~ 2개월)"""
    
    def __init__(self):
        super().__init__("MidTermEnsemble", "medium")
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'arima': None  # ARIMA는 데이터에 맞게 동적으로 생성
        }
        self.prediction_periods = {
            '1주일': 5,
            '2주일': 10,
            '1개월': 20,
            '2개월': 40
        }
        
    def train(self, data: pd.DataFrame, features: pd.DataFrame, 
              economic_indicators: Optional[pd.DataFrame] = None):
        """중기 예측 모델 학습"""
        # 중기 예측을 위한 추가 특징 생성
        extended_features = self._create_extended_features(data, features, economic_indicators)
        
        # 타겟 생성 (다양한 기간의 수익률)
        targets = {}
        for period_name, days in self.prediction_periods.items():
            targets[period_name] = data['close'].shift(-days) / data['close'] - 1
        
        # 유효한 데이터만 사용
        valid_idx = extended_features.index.intersection(targets['1주일'].dropna().index)
        X = extended_features.loc[valid_idx]
        
        # 각 기간별로 모델 학습
        for period_name, days in self.prediction_periods.items():
            y = targets[period_name].loc[valid_idx].dropna()
            X_valid = X.loc[y.index]
            
            # 앙상블 모델 학습
            for model_name, model in self.models.items():
                if model_name != 'arima' and model is not None:
                    model.fit(self.scaler.fit_transform(X_valid), y)
                    
                    # 특징 중요도 저장
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[f'{model_name}_{period_name}'] = pd.Series(
                            model.feature_importances_,
                            index=X_valid.columns
                        ).sort_values(ascending=False)
        
        # ARIMA 모델은 시계열 데이터에 대해 별도 학습
        self._train_arima_model(data['close'])
        
        self.last_train_date = datetime.now()
        logger.info("중기 예측 모델 학습 완료")
        
    def _create_extended_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                 economic_indicators: Optional[pd.DataFrame]) -> pd.DataFrame:
        """중기 예측을 위한 확장 특징 생성"""
        extended = features.copy()
        
        # 장기 추세 특징
        for period in [50, 100, 200]:
            if len(data) > period:
                extended[f'trend_{period}d'] = data['close'].rolling(period).mean().pct_change(20)
        
        # 변동성 특징
        extended['volatility_20d'] = data['close'].pct_change().rolling(20).std()
        extended['volatility_60d'] = data['close'].pct_change().rolling(60).std()
        
        # 경제 지표 통합
        if economic_indicators is not None:
            # 주요 경제 지표의 추세
            macro_features = self._process_macro_indicators(economic_indicators)
            extended = extended.join(macro_features, how='left')
        
        # 계절성 특징
        extended['yearly_high_ratio'] = data['close'] / data['close'].rolling(252).max()
        extended['yearly_low_ratio'] = data['close'] / data['close'].rolling(252).min()
        
        return extended.dropna()
    
    def _process_macro_indicators(self, economic_data: pd.DataFrame) -> pd.DataFrame:
        """거시 경제 지표 처리"""
        macro_features = pd.DataFrame()
        
        # GDP, 금리, 인플레이션 등 주요 지표 추세
        key_indicators = {
            'gdp_trend': 'GDP',
            'interest_trend': 'Interest Rate',
            'inflation_trend': 'CPI',
            'unemployment_trend': 'Unemployment'
        }
        
        for feature_name, indicator_keyword in key_indicators.items():
            relevant_data = economic_data[
                economic_data['event'].str.contains(indicator_keyword, case=False, na=False)
            ]
            
            if not relevant_data.empty and 'actual' in relevant_data.columns:
                # 시계열 추세 계산
                trend_data = relevant_data.groupby('datetime')['actual'].mean()
                if len(trend_data) > 1:
                    macro_features[feature_name] = trend_data.pct_change().rolling(3).mean()
        
        return macro_features
    
    def _train_arima_model(self, price_series: pd.Series):
        """ARIMA 모델 학습"""
        try:
            # 로그 변환으로 안정화
            log_prices = np.log(price_series.dropna())
            
            # ARIMA 모델 fitting
            self.models['arima'] = ARIMA(log_prices, order=(2, 1, 2))
            self.models['arima'] = self.models['arima'].fit()
            
            logger.info("ARIMA 모델 학습 완료")
        except Exception as e:
            logger.warning(f"ARIMA 모델 학습 실패: {e}")
            self.models['arima'] = None
    
    def predict(self, current_data: pd.DataFrame, features: pd.DataFrame,
                economic_indicators: Optional[pd.DataFrame] = None) -> Dict:
        """중기 예측 수행"""
        extended_features = self._create_extended_features(current_data, features, economic_indicators)
        current_features = extended_features.iloc[-1:] if not extended_features.empty else pd.DataFrame()
        
        if current_features.empty:
            logger.warning("예측을 위한 특징 데이터가 부족합니다.")
            return {}
        
        current_price = current_data['close'].iloc[-1]
        results = {}
        
        for period_name, days in self.prediction_periods.items():
            predictions = []
            
            # 앙상블 모델 예측
            for model_name, model in self.models.items():
                if model_name != 'arima' and model is not None:
                    try:
                        pred = model.predict(self.scaler.transform(current_features))[0]
                        predictions.append(pred)
                    except Exception as e:
                        logger.warning(f"{model_name} 예측 실패: {e}")
            
            # ARIMA 예측 추가
            if self.models['arima'] is not None:
                try:
                    arima_forecast = self.models['arima'].forecast(steps=days)
                    arima_return = np.exp(arima_forecast.iloc[-1]) / current_price - 1
                    predictions.append(arima_return)
                except Exception as e:
                    logger.warning(f"ARIMA 예측 실패: {e}")
            
            if predictions:
                # 앙상블 예측 (평균)
                ensemble_return = np.mean(predictions)
                predicted_price = current_price * (1 + ensemble_return)
                
                # 중기 변동성 계산
                historical_returns = current_data['close'].pct_change().dropna()
                base_volatility = historical_returns.rolling(60).std().iloc[-1]
                period_volatility = base_volatility * np.sqrt(days)
                
                # 예측 범위
                lower, upper = self.calculate_prediction_ranges(
                    predicted_price,
                    period_volatility,
                    0.70  # 중기 예측 신뢰도
                )
                
                # 주요 근거 분석
                reasons = self._analyze_prediction_reasons(
                    current_data, features, extended_features, period_name
                )
                
                results[period_name] = {
                    'predicted_price': predicted_price,
                    'predicted_return': ensemble_return * 100,
                    'price_range': (lower, upper),
                    'confidence': 0.70 - (days / 100),  # 기간에 따른 신뢰도 조정
                    'key_factors': reasons
                }
        
        return results
    
    def _analyze_prediction_reasons(self, data: pd.DataFrame, features: pd.DataFrame,
                                   extended_features: pd.DataFrame, period: str) -> List[str]:
        """예측의 주요 근거 분석"""
        reasons = []
        
        # 기술적 지표 분석
        if 'rsi' in features.columns:
            current_rsi = features['rsi'].iloc[-1]
            if current_rsi < 30:
                reasons.append("RSI 과매도 구간 진입")
            elif current_rsi > 70:
                reasons.append("RSI 과매수 구간 진입")
        
        # 추세 분석
        if 'trend_50d' in extended_features.columns:
            trend = extended_features['trend_50d'].iloc[-1]
            if trend > 0.05:
                reasons.append("중기 상승 추세 지속")
            elif trend < -0.05:
                reasons.append("중기 하락 추세 진행")
        
        # 뉴스 감성
        if 'sentiment_mean' in features.columns:
            sentiment = features['sentiment_mean'].iloc[-1]
            if sentiment > 0.5:
                reasons.append("긍정적 뉴스 감성 우세")
            elif sentiment < -0.5:
                reasons.append("부정적 뉴스 감성 증가")
        
        # 특징 중요도 기반 분석
        if f'rf_{period}' in self.feature_importance:
            top_features = self.feature_importance[f'rf_{period}'].head(3)
            for feature, importance in top_features.items():
                if importance > 0.1:  # 중요도 10% 이상
                    feature_value = extended_features[feature].iloc[-1] if feature in extended_features.columns else None
                    if feature_value is not None:
                        if 'volume' in feature and feature_value > 1.5:
                            reasons.append("거래량 급증")
                        elif 'volatility' in feature and feature_value > extended_features[feature].mean():
                            reasons.append("변동성 확대")
        
        return reasons[:5]  # 최대 5개 이유만 반환


class LongTermModel(BaseTimeframePredictionModel):
    """장기 예측 모델 (3개월 이상)"""
    
    def __init__(self):
        super().__init__("LongTermFundamental", "long")
        # 장기 예측은 펀더멘털 분석 중심
        self.fundamental_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            random_state=42
        )
        
    def train(self, data: pd.DataFrame, features: pd.DataFrame,
              fundamental_data: Optional[pd.DataFrame] = None):
        """장기 예측 모델 학습"""
        # 장기 예측을 위한 특징 생성
        long_term_features = self._create_long_term_features(
            data, features, fundamental_data
        )
        
        # 3개월, 6개월 수익률 타겟
        targets = {
            '3개월': data['close'].shift(-60) / data['close'] - 1,
            '6개월': data['close'].shift(-120) / data['close'] - 1
        }
        
        # 학습 데이터 준비
        for period, target in targets.items():
            valid_idx = long_term_features.index.intersection(target.dropna().index)
            if len(valid_idx) > 50:  # 충분한 데이터가 있을 때만 학습
                X = self.scaler.fit_transform(long_term_features.loc[valid_idx])
                y = target.loc[valid_idx]
                
                self.fundamental_model.fit(X, y)
                
                # 특징 중요도 저장
                self.feature_importance[period] = pd.Series(
                    self.fundamental_model.feature_importances_,
                    index=long_term_features.columns
                ).sort_values(ascending=False)
        
        self.last_train_date = datetime.now()
        logger.info("장기 예측 모델 학습 완료")
    
    def _create_long_term_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                  fundamental_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """장기 예측을 위한 펀더멘털 중심 특징"""
        long_features = pd.DataFrame(index=features.index)
        
        # 장기 가격 추세
        for period in [120, 252]:  # 6개월, 1년
            if len(data) > period:
                long_features[f'return_{period}d'] = data['close'].pct_change(period)
                long_features[f'max_drawdown_{period}d'] = (
                    data['close'] / data['close'].rolling(period).max() - 1
                )
        
        # 변동성 체제
        long_features['volatility_regime'] = (
            data['close'].pct_change().rolling(60).std() / 
            data['close'].pct_change().rolling(252).std()
        )
        
        # 시장 사이클 지표
        long_features['market_cycle'] = self._calculate_market_cycle(data)
        
        # 펀더멘털 데이터 통합 (예: P/E, P/B, ROE 등)
        if fundamental_data is not None:
            # 여기에 펀더멘털 데이터 처리 로직 추가
            pass
        
        return long_features.dropna()
    
    def _calculate_market_cycle(self, data: pd.DataFrame) -> pd.Series:
        """시장 사이클 지표 계산"""
        # 간단한 사이클 지표: 200일 이동평균 대비 위치
        if len(data) > 200:
            ma200 = data['close'].rolling(200).mean()
            cycle = (data['close'] - ma200) / ma200
            return cycle
        return pd.Series(index=data.index)
    
    def predict(self, current_data: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """장기 예측 수행"""
        # 장기 예측은 더 보수적이고 넓은 범위 제공
        # 구현 생략 (MidTermModel과 유사한 방식)
        pass