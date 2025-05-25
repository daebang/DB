import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy, Signal
from analysis.models.ml_models import MLTradingModel
from analysis.models.deep_learning import LSTMTradingModel

class MLTradingStrategy(BaseStrategy):
    """머신러닝 기반 거래 전략"""
    
    def __init__(self, name: str = "ML_Strategy", parameters: Dict = None):
        super().__init__(name, parameters)
        
        # 기본 파라미터 설정
        default_params = {
            'confidence_threshold': 0.7,
            'model_type': 'classification',  # 'classification' or 'regression'
            'use_lstm': False,
            'min_data_points': 100,
            'retrain_frequency': 30  # 30일마다 재훈련
        }
        
        self.parameters = {**default_params, **(parameters or {})}
        
        # 모델 초기화
        if self.parameters['use_lstm']:
            self.model = LSTMTradingModel()
        else:
            self.model = MLTradingModel(self.parameters['model_type'])
        
        self.last_retrain_date = None
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """ML 모델 기반 신호 생성"""
        if len(data) < self.parameters['min_data_points']:
            return None
        
        try:
            # 모델이 훈련되지 않았거나 재훈련이 필요한 경우
            if not self.model.is_trained or self._needs_retraining():
                self._train_model(data, symbol)
            
            # 예측 수행
            if self.parameters['use_lstm']:
                prediction_result = self.model.predict_next(data)
                
                # LSTM 예측 결과 해석
                price_change = prediction_result['price_change_percent']
                confidence = prediction_result['confidence']
                
                if price_change > 2.0 and confidence > self.parameters['confidence_threshold']:
                    action = 'BUY'
                elif price_change < -2.0 and confidence > self.parameters['confidence_threshold']:
                    action = 'SELL'
                else:
                    action = 'HOLD'
                    
            else:
                # 분류/회귀 모델 예측
                X, _ = self.model.prepare_features(data.tail(1))
                if len(X) == 0:
                    return None
                    
                prediction_result = self.model.predict(X)
                
                if self.parameters['model_type'] == 'classification':
                    predicted_class = prediction_result['predictions'][0]
                    confidence = prediction_result['confidence'][0]
                    
                    if predicted_class == 'BUY' and confidence > self.parameters['confidence_threshold']:
                        action = 'BUY'
                    elif predicted_class == 'SELL' and confidence > self.parameters['confidence_threshold']:
                        action = 'SELL'
                    else:
                        action = 'HOLD'
                else:
                    predicted_return = prediction_result['predictions'][0]
                    confidence = prediction_result['confidence'][0]
                    
                    if predicted_return > 0.02 and confidence > self.parameters['confidence_threshold']:
                        action = 'BUY'
                    elif predicted_return < -0.02 and confidence > self.parameters['confidence_threshold']:
                        action = 'SELL'
                    else:
                        action = 'HOLD'
            
            # 신호 생성
            if action != 'HOLD':
                current_price = data['close'].iloc[-1]
                
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    quantity=0,  # 포지션 크기는 나중에 계산
                    confidence=confidence,
                    timestamp=datetime.now(),
                    strategy_name=self.name,
                    metadata={
                        'model_type': self.parameters['model_type'],
                        'prediction_details': prediction_result
                    }
                )
                
                self.signals_history.append(signal)
                return signal
                
        except Exception as e:
            print(f"ML 신호 생성 오류 ({symbol}): {e}")
            
        return None
    
    def _train_model(self, data: pd.DataFrame, symbol: str):
        """모델 훈련"""
        try:
            if self.parameters['use_lstm']:
                # LSTM 모델 훈련
                X, y = self.model.prepare_sequences(data)
                if len(X) > 0:
                    self.model.train(X, y)
            else:
                # 분류/회귀 모델 훈련
                target_col = self._create_target_variable(data)
                X, y = self.model.prepare_features(data, target_col)
                
                if len(X) > 0 and len(y) > 0:
                    self.model.train(X, y)
            
            self.last_retrain_date = datetime.now()
            print(f"모델 훈련 완료: {symbol}")
            
        except Exception as e:
            print(f"모델 훈련 오류 ({symbol}): {e}")
    
    def _create_target_variable(self, data: pd.DataFrame) -> str:
        """타겟 변수 생성"""
        df = data.copy()
        
        if self.parameters['model_type'] == 'classification':
            # 분류: 다음 날 수익률이 양수면 BUY, 음수면 SELL
            df['future_return'] = df['close'].shift(-1) / df['close'] - 1
            df['target'] = df['future_return'].apply(
                lambda x: 'BUY' if x > 0.01 else ('SELL' if x < -0.01 else 'HOLD')
            )
            return 'target'
        else:
            # 회귀: 다음 날 수익률 예측
            df['target'] = df['close'].shift(-1) / df['close'] - 1
            return 'target'
    
    def _needs_retraining(self) -> bool:
        """재훈련 필요 여부 확인"""
        if self.last_retrain_date is None:
            return True
        
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        return days_since_retrain >= self.parameters['retrain_frequency']
    
    def update_parameters(self, new_parameters: Dict):
        """파라미터 업데이트"""
        self.parameters.update(new_parameters)
        
        # 모델 타입이 변경된 경우 새 모델 생성
        if 'model_type' in new_parameters or 'use_lstm' in new_parameters:
            if self.parameters['use_lstm']:
                self.model = LSTMTradingModel()
            else:
                self.model = MLTradingModel(self.parameters['model_type'])