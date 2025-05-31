import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ConfidenceCalculator:
    """예측 신뢰도 계산기"""
    
    def __init__(self):
        self.confidence_weights = {
            'model_agreement': 0.3,  # 모델 간 일치도
            'data_quality': 0.2,     # 데이터 품질
            'market_condition': 0.2, # 시장 상황
            'prediction_horizon': 0.2, # 예측 기간
            'historical_accuracy': 0.1 # 과거 정확도
        }
        self.historical_predictions = []
        
    def calculate_overall_confidence(self, predictions: Dict) -> float:
        """전체 예측에 대한 종합 신뢰도 계산"""
        confidence_scores = []
        
        # 각 기간별 신뢰도 수집
        for timeframe, periods in predictions.items():
            if isinstance(periods, dict):
                for period, pred in periods.items():
                    if isinstance(pred, dict) and 'confidence' in pred:
                        confidence_scores.append(pred['confidence'])
        
        if not confidence_scores:
            return 0.5  # 기본값
        
        # 가중 평균 (단기 예측에 더 높은 가중치)
        weights = np.linspace(2, 1, len(confidence_scores))
        weights = weights / weights.sum()
        
        overall_confidence = np.average(confidence_scores, weights=weights)
        
        return round(overall_confidence, 3)
    
    def calculate_prediction_confidence(self, 
                                      prediction_data: Dict,
                                      historical_data: pd.DataFrame,
                                      model_predictions: List[float]) -> float:
        """개별 예측에 대한 신뢰도 계산"""
        
        confidence_components = {}
        
        # 1. 모델 간 일치도
        if len(model_predictions) > 1:
            pred_std = np.std(model_predictions)
            pred_mean = np.mean(model_predictions)
            if pred_mean != 0:
                coefficient_of_variation = pred_std / abs(pred_mean)
                confidence_components['model_agreement'] = max(0, 1 - coefficient_of_variation)
            else:
                confidence_components['model_agreement'] = 0.5
        else:
            confidence_components['model_agreement'] = 0.7  # 단일 모델
        
        # 2. 데이터 품질
        data_quality = self._assess_data_quality(historical_data)
        confidence_components['data_quality'] = data_quality
        
        # 3. 시장 상황
        market_condition = self._assess_market_condition(historical_data)
        confidence_components['market_condition'] = market_condition
        
        # 4. 예측 기간에 따른 신뢰도
        prediction_horizon = prediction_data.get('horizon_days', 1)
        horizon_confidence = 1 / (1 + np.log(prediction_horizon))
        confidence_components['prediction_horizon'] = horizon_confidence
        
        # 5. 과거 예측 정확도 (있는 경우)
        historical_accuracy = self._get_historical_accuracy(prediction_data.get('symbol'))
        confidence_components['historical_accuracy'] = historical_accuracy
        
        # 가중 평균 계산
        total_confidence = 0
        for component, weight in self.confidence_weights.items():
            total_confidence += confidence_components.get(component, 0.5) * weight
        
        return round(total_confidence, 3)
    
    def _assess_data_quality(self, historical_data: pd.DataFrame) -> float:
        """데이터 품질 평가"""
        quality_score = 1.0
        
        # 데이터 길이
        if len(historical_data) < 100:
            quality_score *= 0.7
        elif len(historical_data) < 50:
            quality_score *= 0.5
        
        # 결측치 비율
        missing_ratio = historical_data.isnull().sum().sum() / historical_data.size
        quality_score *= (1 - missing_ratio)
        
        # 최근 데이터 여부
        if isinstance(historical_data.index, pd.DatetimeIndex):
            last_date = historical_data.index[-1]
            days_old = (datetime.now() - last_date).days
            if days_old > 1:
                quality_score *= 0.9 ** days_old
        
        return quality_score
    
    def _assess_market_condition(self, historical_data: pd.DataFrame) -> float:
        """시장 상황 평가 (변동성 기반)"""
        if 'close' not in historical_data.columns:
            return 0.5
        
        # 최근 변동성
        recent_returns = historical_data['close'].pct_change().tail(20)
        volatility = recent_returns.std()
        
        # 정상 변동성 범위 (연간 10-30%)
        daily_normal_vol = 0.2 / np.sqrt(252)  # 연간 20% 변동성
        
        if volatility < daily_normal_vol * 0.5:
            # 너무 낮은 변동성
            return 0.7
        elif volatility > daily_normal_vol * 2:
            # 너무 높은 변동성
            return 0.6
        else:
            # 정상 범위
            return 0.9
    
    def _get_historical_accuracy(self, symbol: Optional[str]) -> float:
        """과거 예측 정확도 조회"""
        if not symbol or not self.historical_predictions:
            return 0.7  # 기본값
        
        # 해당 종목의 과거 예측 필터링
        symbol_predictions = [p for p in self.historical_predictions 
                            if p.get('symbol') == symbol]
        
        if len(symbol_predictions) < 5:
            return 0.7  # 충분한 데이터 없음
        
        # 최근 10개 예측의 정확도 평가
        recent_predictions = symbol_predictions[-10:]
        accuracies = []
        
        for pred in recent_predictions:
            if 'actual_return' in pred and 'predicted_return' in pred:
                error = abs(pred['actual_return'] - pred['predicted_return'])
                accuracy = max(0, 1 - error / abs(pred['predicted_return']))
                accuracies.append(accuracy)
        
        if accuracies:
            return np.mean(accuracies)
        else:
            return 0.7
    
    def update_prediction_history(self, symbol: str, 
                                predicted_return: float,
                                actual_return: float,
                                prediction_date: datetime):
        """예측 이력 업데이트"""
        self.historical_predictions.append({
            'symbol': symbol,
            'predicted_return': predicted_return,
            'actual_return': actual_return,
            'prediction_date': prediction_date,
            'error': abs(predicted_return - actual_return)
        })
        
        # 최대 1000개 예측만 유지
        if len(self.historical_predictions) > 1000:
            self.historical_predictions = self.historical_predictions[-1000:]