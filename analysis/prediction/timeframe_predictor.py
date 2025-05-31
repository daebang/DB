import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
from .prediction_models import ShortTermModel, MidTermModel, LongTermModel
from .confidence_calculator import ConfidenceCalculator

logger = logging.getLogger(__name__)

class TimeframePredictor:
    """통합 기간별 예측 시스템"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.models = {
            'short': ShortTermModel(),
            'medium': MidTermModel(),
            'long': LongTermModel()
        }
        self.confidence_calculator = ConfidenceCalculator()
        self.last_prediction_time = None
        self.prediction_cache = {}
        
    def predict_all_timeframes(self, 
                             historical_data: pd.DataFrame,
                             technical_indicators: pd.DataFrame,
                             news_sentiment: Optional[pd.DataFrame] = None,
                             economic_indicators: Optional[pd.DataFrame] = None,
                             fundamental_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """모든 기간에 대한 예측 수행"""
        
        logger.info(f"{self.symbol}: 기간별 예측 시작")
        
        # 캐시 확인
        cache_key = self._generate_cache_key(historical_data)
        if cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            if (datetime.now() - cached_result['timestamp']).seconds < 3600:  # 1시간 캐시
                logger.info(f"{self.symbol}: 캐시된 예측 결과 사용")
                return cached_result['predictions']
        
# 모든 모델에 대한 특징 준비
        features = self._prepare_unified_features(
            historical_data, technical_indicators, news_sentiment, economic_indicators
        )
        
        predictions = {
            'symbol': self.symbol,
            'timestamp': datetime.now(),
            'current_price': historical_data['close'].iloc[-1],
            'timeframes': {}
        }
        
        # 병렬로 각 기간별 예측 수행
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 단기 예측
            short_future = executor.submit(
                self._predict_short_term,
                historical_data, features, news_sentiment
            )
            
            # 중기 예측
            medium_future = executor.submit(
                self._predict_medium_term,
                historical_data, features, economic_indicators
            )
            
            # 장기 예측
            long_future = executor.submit(
                self._predict_long_term,
                historical_data, features, fundamental_data
            )
            
            # 결과 수집
            predictions['timeframes']['short_term'] = short_future.result()
            predictions['timeframes']['medium_term'] = medium_future.result()
            predictions['timeframes']['long_term'] = long_future.result()
        
        # 종합 신뢰도 계산
        predictions['overall_confidence'] = self.confidence_calculator.calculate_overall_confidence(
            predictions['timeframes']
        )
        
        # 투자 추천 생성
        predictions['recommendation'] = self._generate_recommendation(predictions)
        
        # 캐시 저장
        self.prediction_cache[cache_key] = {
            'timestamp': datetime.now(),
            'predictions': predictions
        }
        
        self.last_prediction_time = datetime.now()
        logger.info(f"{self.symbol}: 기간별 예측 완료")
        
        return predictions
    
    def _prepare_unified_features(self, historical_data: pd.DataFrame,
                                technical_indicators: pd.DataFrame,
                                news_sentiment: Optional[pd.DataFrame],
                                economic_indicators: Optional[pd.DataFrame]) -> pd.DataFrame:
        """통합 특징 데이터 준비"""
        # 기본 특징 준비
        features = self.models['short'].prepare_features(
            historical_data, news_sentiment, economic_indicators
        )
        
        # 기술적 지표 통합
        if technical_indicators is not None:
            # 이미 계산된 지표들을 features에 병합
            ta_features = technical_indicators[[col for col in technical_indicators.columns 
                                              if any(ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD', 'BB'])]]
            features = features.join(ta_features, how='left')
        
        return features.fillna(method='ffill').fillna(0)
    
    def _predict_short_term(self, historical_data: pd.DataFrame,
                          features: pd.DataFrame,
                          news_sentiment: Optional[pd.DataFrame]) -> Dict:
        """단기 예측 수행"""
        try:
            # 모델이 학습되지 않았으면 학습
            if self.models['short'].model is None:
                logger.info(f"{self.symbol}: 단기 모델 학습 시작")
                self.models['short'].train(historical_data, features)
            
            # 예측 수행
            predictions = self.models['short'].predict(historical_data, features)
            
            # 뉴스 영향 분석
            news_impact = self._analyze_news_impact(news_sentiment) if news_sentiment is not None else {}
            
            # 각 기간별 상세 정보 추가
            for period in predictions:
                predictions[period]['factors'] = self._get_short_term_factors(
                    historical_data, features, period
                )
                predictions[period]['news_impact'] = news_impact
            
            return predictions
            
        except Exception as e:
            logger.error(f"{self.symbol} 단기 예측 실패: {e}")
            return self._get_default_prediction('short')
    
    def _predict_medium_term(self, historical_data: pd.DataFrame,
                           features: pd.DataFrame,
                           economic_indicators: Optional[pd.DataFrame]) -> Dict:
        """중기 예측 수행"""
        try:
            # 모델 학습 확인
            if self.models['medium'].last_train_date is None:
                logger.info(f"{self.symbol}: 중기 모델 학습 시작")
                self.models['medium'].train(historical_data, features, economic_indicators)
            
            # 예측 수행
            predictions = self.models['medium'].predict(
                historical_data, features, economic_indicators
            )
            
            # 경제 지표 영향 분석
            if economic_indicators is not None:
                for period in predictions:
                    predictions[period]['economic_impact'] = self._analyze_economic_impact(
                        economic_indicators, period
                    )
            
            return predictions
            
        except Exception as e:
            logger.error(f"{self.symbol} 중기 예측 실패: {e}")
            return self._get_default_prediction('medium')
    
    def _predict_long_term(self, historical_data: pd.DataFrame,
                         features: pd.DataFrame,
                         fundamental_data: Optional[pd.DataFrame]) -> Dict:
        """장기 예측 수행"""
        # 장기 예측은 더 보수적인 접근
        try:
            # 단순히 과거 장기 수익률 기반 예측
            predictions = {}
            
            # 3개월 예측
            if len(historical_data) > 60:
                historical_3m_returns = historical_data['close'].pct_change(60).dropna()
                mean_return = historical_3m_returns.mean()
                std_return = historical_3m_returns.std()
                
                current_price = historical_data['close'].iloc[-1]
                predicted_price = current_price * (1 + mean_return)
                
                predictions['3개월'] = {
                    'predicted_price': predicted_price,
                    'predicted_return': mean_return * 100,
                    'price_range': (
                        current_price * (1 + mean_return - 2*std_return),
                        current_price * (1 + mean_return + 2*std_return)
                    ),
                    'confidence': 0.60,
                    'factors': ["과거 3개월 평균 수익률 기반", "장기 추세 분석"]
                }
            
            # 6개월 예측
            if len(historical_data) > 120:
                historical_6m_returns = historical_data['close'].pct_change(120).dropna()
                mean_return = historical_6m_returns.mean()
                std_return = historical_6m_returns.std()
                
                current_price = historical_data['close'].iloc[-1]
                predicted_price = current_price * (1 + mean_return)
                
                predictions['6개월'] = {
                    'predicted_price': predicted_price,
                    'predicted_return': mean_return * 100,
                    'price_range': (
                        current_price * (1 + mean_return - 2*std_return),
                        current_price * (1 + mean_return + 2*std_return)
                    ),
                    'confidence': 0.50,
                    'factors': ["과거 6개월 평균 수익률 기반", "시장 사이클 고려"]
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"{self.symbol} 장기 예측 실패: {e}")
            return self._get_default_prediction('long')
    
    def _analyze_news_impact(self, news_sentiment: pd.DataFrame) -> Dict:
        """뉴스 감성이 주가에 미치는 영향 분석"""
        if news_sentiment.empty:
            return {'impact': 'neutral', 'score': 0, 'key_news': []}
        
        # 최근 3일간 뉴스 감성 분석
        recent_news = news_sentiment[
            news_sentiment.index >= (datetime.now() - timedelta(days=3))
        ]
        
        if recent_news.empty:
            return {'impact': 'neutral', 'score': 0, 'key_news': []}
        
        # 평균 감성 점수
        avg_sentiment = recent_news['sentiment_score'].mean()
        
        # 주요 뉴스 추출
        key_news = []
        if 'title' in recent_news.columns:
            # 가장 긍정적/부정적인 뉴스 선택
            sorted_news = recent_news.sort_values('sentiment_score', ascending=False)
            
            # 상위 2개, 하위 2개
            if len(sorted_news) > 0:
                for idx in [0, 1, -2, -1]:
                    if 0 <= idx < len(sorted_news) or -len(sorted_news) <= idx < 0:
                        news_item = sorted_news.iloc[idx]
                        key_news.append({
                            'title': news_item.get('title', 'N/A'),
                            'sentiment': news_item.get('sentiment_score', 0),
                            'date': news_item.name if isinstance(news_item.name, datetime) else 'N/A'
                        })
        
        # 영향 판단
        if avg_sentiment > 0.5:
            impact = 'positive'
        elif avg_sentiment < -0.5:
            impact = 'negative'
        else:
            impact = 'neutral'
        
        return {
            'impact': impact,
            'score': avg_sentiment,
            'news_count': len(recent_news),
            'key_news': key_news[:4]  # 최대 4개
        }
    
    def _analyze_economic_impact(self, economic_indicators: pd.DataFrame, period: str) -> Dict:
        """경제 지표가 주가에 미치는 영향 분석"""
        if economic_indicators.empty:
            return {'impact': 'neutral', 'key_indicators': []}
        
        # 기간에 따른 경제 지표 필터링
        days_map = {'1주일': 7, '2주일': 14, '1개월': 30, '2개월': 60}
        days = days_map.get(period, 30)
        
        future_indicators = economic_indicators[
            (economic_indicators['datetime'] >= datetime.now()) &
            (economic_indicators['datetime'] <= datetime.now() + timedelta(days=days))
        ]
        
        if future_indicators.empty:
            return {'impact': 'neutral', 'key_indicators': []}
        
        # 중요 지표 분석
        high_impact = future_indicators[future_indicators['importance'] >= 3]
        
        key_indicators = []
        impact_score = 0
        
        for _, indicator in high_impact.iterrows():
            # 예상치와 이전값 비교로 영향 추정
            if pd.notna(indicator['forecast']) and pd.notna(indicator['previous']):
                if indicator['forecast'] > indicator['previous']:
                    impact_direction = 'positive' if 'GDP' in indicator['event'] or 'Employment' in indicator['event'] else 'negative'
                else:
                    impact_direction = 'negative' if 'GDP' in indicator['event'] or 'Employment' in indicator['event'] else 'positive'
                
                key_indicators.append({
                    'event': indicator['event'],
                    'date': indicator['datetime'].strftime('%Y-%m-%d'),
                    'impact': impact_direction,
                    'importance': indicator['importance']
                })
                
                impact_score += 1 if impact_direction == 'positive' else -1
        
        # 전체 영향 판단
        if impact_score > 2:
            overall_impact = 'positive'
        elif impact_score < -2:
            overall_impact = 'negative'
        else:
            overall_impact = 'neutral'
        
        return {
            'impact': overall_impact,
            'key_indicators': key_indicators[:5]  # 최대 5개
        }
    
    def _get_short_term_factors(self, historical_data: pd.DataFrame,
                              features: pd.DataFrame, period: str) -> List[str]:
        """단기 예측의 주요 요인 분석"""
        factors = []
        
        # 기술적 지표 기반 요인
        if 'rsi' in features.columns:
            current_rsi = features['rsi'].iloc[-1]
            if current_rsi < 30:
                factors.append("RSI 과매도 신호 (매수 기회)")
            elif current_rsi > 70:
                factors.append("RSI 과매수 신호 (조정 가능성)")
        
        # 이동평균 크로스
        if 'sma_5_ratio' in features.columns and 'sma_20_ratio' in features.columns:
            sma5_ratio = features['sma_5_ratio'].iloc[-1]
            sma20_ratio = features['sma_20_ratio'].iloc[-1]
            if sma5_ratio > 1 and sma20_ratio > 1:
                factors.append("단기 및 중기 이동평균선 상향 돌파")
            elif sma5_ratio < 1 and sma20_ratio < 1:
                factors.append("단기 및 중기 이동평균선 하향 이탈")
        
        # 거래량 분석
        if 'volume_ratio' in features.columns:
            vol_ratio = features['volume_ratio'].iloc[-1]
            if vol_ratio > 1.5:
                factors.append("평균 대비 거래량 급증 (150%+)")
        
        # 모멘텀
        recent_return = historical_data['close'].pct_change(5).iloc[-1]
        if recent_return > 0.05:
            factors.append("최근 5일 강한 상승 모멘텀")
        elif recent_return < -0.05:
            factors.append("최근 5일 하락 압력 지속")
        
        return factors[:4]  # 최대 4개 요인
    
    def _generate_recommendation(self, predictions: Dict) -> Dict:
        """종합 예측 기반 투자 추천 생성"""
        recommendation = {
            'action': 'HOLD',  # BUY, SELL, HOLD
            'strength': 'medium',  # strong, medium, weak
            'reasoning': [],
            'risk_level': 'medium',  # low, medium, high
            'suggested_strategy': ''
        }
        
        # 각 기간별 예측 점수 계산
        scores = {'short': 0, 'medium': 0, 'long': 0}
        
        # 단기 예측 평가
        if 'short_term' in predictions['timeframes']:
            for period, pred in predictions['timeframes']['short_term'].items():
                if pred['predicted_return'] > 2:
                    scores['short'] += 1
                elif pred['predicted_return'] < -2:
                    scores['short'] -= 1
        
        # 중기 예측 평가
        if 'medium_term' in predictions['timeframes']:
            for period, pred in predictions['timeframes']['medium_term'].items():
                if pred['predicted_return'] > 5:
                    scores['medium'] += 2
                elif pred['predicted_return'] < -5:
                    scores['medium'] -= 2
        
        # 종합 점수
        total_score = sum(scores.values())
        
        # 추천 결정
        if total_score >= 3:
            recommendation['action'] = 'BUY'
            recommendation['strength'] = 'strong' if total_score >= 5 else 'medium'
            recommendation['reasoning'].append("단기 및 중기 전망 모두 긍정적")
        elif total_score <= -3:
            recommendation['action'] = 'SELL'
            recommendation['strength'] = 'strong' if total_score <= -5 else 'medium'
            recommendation['reasoning'].append("전반적인 하락 압력 예상")
        else:
            recommendation['action'] = 'HOLD'
            recommendation['reasoning'].append("혼재된 신호로 관망 권고")
        
        # 리스크 수준 평가
        confidence_avg = predictions['overall_confidence']
        if confidence_avg < 0.6:
            recommendation['risk_level'] = 'high'
            recommendation['reasoning'].append("예측 신뢰도가 낮아 리스크 높음")
        elif confidence_avg > 0.8:
            recommendation['risk_level'] = 'low'
        
        # 전략 제안
        if recommendation['action'] == 'BUY':
            if scores['short'] > scores['medium']:
                recommendation['suggested_strategy'] = "단기 트레이딩 전략 (1-3일 보유)"
            else:
                recommendation['suggested_strategy'] = "스윙 트레이딩 전략 (1-4주 보유)"
        elif recommendation['action'] == 'SELL':
            recommendation['suggested_strategy'] = "즉시 매도 또는 숏 포지션 고려"
        else:
            recommendation['suggested_strategy'] = "추가 신호 확인 후 진입"
        
        return recommendation
    
    def _generate_cache_key(self, historical_data: pd.DataFrame) -> str:
        """캐시 키 생성"""
        last_date = historical_data.index[-1] if isinstance(historical_data.index, pd.DatetimeIndex) else str(historical_data.index[-1])
        last_price = historical_data['close'].iloc[-1]
        return f"{self.symbol}_{last_date}_{last_price:.2f}"
    
    def _get_default_prediction(self, term: str) -> Dict:
        """예측 실패 시 기본값 반환"""
        default_predictions = {
            'short': {
                '1일': {'predicted_return': 0, 'confidence': 0.3, 'factors': ["예측 실패"]},
                '2일': {'predicted_return': 0, 'confidence': 0.2, 'factors': ["예측 실패"]},
                '3일': {'predicted_return': 0, 'confidence': 0.1, 'factors': ["예측 실패"]}
            },
            'medium': {
                '1주일': {'predicted_return': 0, 'confidence': 0.3, 'factors': ["예측 실패"]},
                '1개월': {'predicted_return': 0, 'confidence': 0.2, 'factors': ["예측 실패"]}
            },
            'long': {
                '3개월': {'predicted_return': 0, 'confidence': 0.2, 'factors': ["예측 실패"]},
                '6개월': {'predicted_return': 0, 'confidence': 0.1, 'factors': ["예측 실패"]}
            }
        }
        
        return default_predictions.get(term, {})