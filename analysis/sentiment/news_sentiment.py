import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
from typing import List, Dict, Optional
import numpy as np

# 금융 특화 감성 사전
FINANCIAL_SENTIMENT_WORDS = {
    'positive': [
        'bullish', 'rally', 'surge', 'soar', 'breakthrough', 'outperform',
        'beat expectations', 'strong growth', 'record high', 'upgrade'
    ],
    'negative': [
        'bearish', 'plunge', 'crash', 'decline', 'underperform', 'disappoint',
        'miss expectations', 'weak', 'downgrade', 'concerns', 'risks'
    ]
}

class NewsSentimentAnalyzer:
    def __init__(self):
        # NLTK 감성 분석기
        self.sia = SentimentIntensityAnalyzer()
        
        # 트랜스포머 기반 감성 분석 파이프라인
        self.transformer_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",  # 금융 특화 BERT 모델
            return_all_scores=True
        )
        
        # 금융 키워드 가중치
        self.financial_weights = self._create_financial_weights()
    
    def _create_financial_weights(self) -> Dict[str, float]:
        """금융 키워드 가중치 생성"""
        weights = {}
        
        for word in FINANCIAL_SENTIMENT_WORDS['positive']:
            weights[word.lower()] = 1.5
            
        for word in FINANCIAL_SENTIMENT_WORDS['negative']:
            weights[word.lower()] = -1.5
            
        return weights
    
    def clean_text(self, text: str) -> str:
        """텍스트 전처리"""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수 문자 정리
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def analyze_sentiment(self, text: str, symbol: str = None) -> Dict:
        """감성 분석 실행"""
        # 텍스트 전처리
        cleaned_text = self.clean_text(text)
        
        # NLTK 감성 점수
        nltk_scores = self.sia.polarity_scores(cleaned_text)
        
        # FinBERT 감성 점수
        finbert_result = self.transformer_analyzer(cleaned_text[:512])  # 최대 길이 제한
        finbert_scores = {item['label'].lower(): item['score'] for item in finbert_result[0]}
        
        # 금융 키워드 보정
        financial_adjustment = self._calculate_financial_adjustment(cleaned_text)
        
        # 종합 감성 점수 계산
        composite_score = self._calculate_composite_score(
            nltk_scores, finbert_scores, financial_adjustment
        )
        
        return {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'symbol': symbol,
            'sentiment_score': composite_score,
            'sentiment_label': self._get_sentiment_label(composite_score),
            'confidence': self._calculate_confidence(nltk_scores, finbert_scores),
            'nltk_scores': nltk_scores,
            'finbert_scores': finbert_scores,
            'financial_adjustment': financial_adjustment
        }
    
    def _calculate_financial_adjustment(self, text: str) -> float:
        """금융 키워드 기반 점수 보정"""
        text_lower = text.lower()
        adjustment = 0.0
        
        for keyword, weight in self.financial_weights.items():
            if keyword in text_lower:
                adjustment += weight
        
        # 정규화
        return np.tanh(adjustment / 10.0)
    
    def _calculate_composite_score(self, nltk_scores: Dict, finbert_scores: Dict, 
                                 financial_adj: float) -> float:
        """종합 감성 점수 계산"""
        # NLTK 복합 점수
        nltk_compound = nltk_scores['compound']
        
        # FinBERT 점수 (positive - negative)
        finbert_score = finbert_scores.get('positive', 0) - finbert_scores.get('negative', 0)
        
        # 가중 평균 (FinBERT에 더 높은 가중치)
        composite = (0.3 * nltk_compound + 0.5 * finbert_score + 0.2 * financial_adj)
        
        # -1 ~ 1 범위로 정규화
        return np.tanh(composite)
    
    def _get_sentiment_label(self, score: float) -> str:
        """감성 라벨 반환"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, nltk_scores: Dict, finbert_scores: Dict) -> float:
        """분석 신뢰도 계산"""
        # NLTK 신뢰도 (복합 점수의 절댓값)
        nltk_confidence = abs(nltk_scores['compound'])
        
        # FinBERT 신뢰도 (최고 점수)
        finbert_confidence = max(finbert_scores.values())
        
        # 평균 신뢰도
        return (nltk_confidence + finbert_confidence) / 2
    
    def analyze_news_batch(self, news_list: List[Dict]) -> pd.DataFrame:
        """뉴스 리스트 일괄 감성 분석"""
        results = []
        
        for news in news_list:
            result = self.analyze_sentiment(news['text'], news.get('symbol'))
            result.update({
                'title': news.get('title', ''),
                'source': news.get('source', ''),
                'published_at': news.get('published_at'),
                'url': news.get('url', '')
            })
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_symbol_sentiment_summary(self, symbol: str, df: pd.DataFrame, 
                                   hours: int = 24) -> Dict:
        """특정 종목의 감성 요약"""
        # 시간 필터링
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        recent_news = df[
            (df['symbol'] == symbol) & 
            (pd.to_datetime(df['published_at']) >= cutoff_time)
        ]
        
        if recent_news.empty:
            return {
                'symbol': symbol,
                'news_count': 0,
                'avg_sentiment': 0.0,
                'sentiment_trend': 'neutral',
                'confidence': 0.0
            }
        
        # 감성 통계
        sentiment_scores = recent_news['sentiment_score']
        avg_sentiment = sentiment_scores.mean()
        sentiment_std = sentiment_scores.std()
        
        # 트렌드 분석 (시간순 감성 변화)
        recent_news_sorted = recent_news.sort_values('published_at')
        recent_sentiment = recent_news_sorted['sentiment_score'].tail(5).mean()
        earlier_sentiment = recent_news_sorted['sentiment_score'].head(5).mean()
        
        if recent_sentiment > earlier_sentiment + 0.1:
            trend = 'improving'
        elif recent_sentiment < earlier_sentiment - 0.1:
            trend = 'deteriorating'
        else:
            trend = 'stable'
        
        return {
            'symbol': symbol,
            'news_count': len(recent_news),
            'avg_sentiment': avg_sentiment,
            'sentiment_std': sentiment_std,
            'sentiment_trend': trend,
            'confidence': recent_news['confidence'].mean(),
            'positive_news_ratio': len(recent_news[recent_news['sentiment_score'] > 0.1]) / len(recent_news),
            'negative_news_ratio': len(recent_news[recent_news['sentiment_score'] < -0.1]) / len(recent_news)
        }