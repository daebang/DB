# analysis/integrated_analyzer.py
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import numpy as np

# 기존 분석 모듈 import (경로는 실제 프로젝트 구조에 맞게 조정 필요)
from .sentiment.news_sentiment import NewsSentimentAnalyzer
# EconomicCalendarCollector는 MainController를 통해 데이터가 전달되므로 직접 import는 불필요할 수 있음
# from ..data.collectors.economic_calendar_collector import EconomicCalendarCollector

logger = logging.getLogger(__name__)

class IntegratedAnalyzer:
    """
    뉴스 감성, 경제 지표, (향후) 기술적 지표 등을 통합적으로 분석하여
    시장 또는 개별 종목에 대한 단기적 전망 및 주요 영향 요인을 평가하는 모듈.
    """

    def __init__(self, news_sentiment_analyzer: Optional[NewsSentimentAnalyzer] = None):
        """
        통합 분석기 초기화.

        Args:
            news_sentiment_analyzer: 이미 초기화된 NewsSentimentAnalyzer 인스턴스.
                                     None이면 내부적으로 생성 (API 키 설정 필요).
        """
        if news_sentiment_analyzer:
            self.news_analyzer = news_sentiment_analyzer
        else:
            # APIConfig가 없으므로, NewsSentimentAnalyzer가 API 키 없이도
            # 제한적으로 동작하거나, 외부에서 APIConfig를 주입받도록 설계 변경 필요.
            # 여기서는 NewsSentimentAnalyzer가 API 키 없이 초기화되면
            # Transformers 모델(finbert)만 사용 가능하다고 가정.
            logger.warning("NewsSentimentAnalyzer가 제공되지 않아 내부적으로 생성합니다. "
                           "NewsAPI를 사용하는 기능은 제한될 수 있습니다.")
            self.news_analyzer = NewsSentimentAnalyzer() # APIConfig 주입 필요

        # 경제 지표 관련 설정값 (예: 중요도 임계값, 영향력 가중치 등)
        self.economic_event_importance_threshold = 2 # 중간 중요도 이상
        # 국가별 시장 영향력 가중치 (예시, 실제로는 더 정교한 모델 필요)
        self.country_market_impact_weights = {
            'US': 1.0, 'EU': 0.7, 'CN': 0.8, 'JP': 0.5, 'GB': 0.4, 'KR': 0.3
        }
        # 이벤트 카테고리별 예상 영향 방향 (예시)
        self.event_category_impact_direction = {
            'ppi': {'positive_surprise': 'positive', 'negative_surprise': 'negative'}, # 생산자물가지수
            'cpi': {'positive_surprise': 'negative_stock', 'negative_surprise': 'positive_stock'}, # 소비자물가지수 (주식시장 관점)
            'interest_rate_decision': {'hike': 'negative_stock', 'cut': 'positive_stock', 'hold': 'neutral'}, # 금리 결정
            'gdp': {'positive_surprise': 'positive', 'negative_surprise': 'negative'}, # GDP 성장률
            'unemployment_rate': {'positive_surprise': 'negative', 'negative_surprise': 'positive'}, # 실업률 (실업률 상승이 부정적)
            'non_farm_payrolls': {'positive_surprise': 'positive', 'negative_surprise': 'negative'}, # 비농업고용지수
            'retail_sales': {'positive_surprise': 'positive', 'negative_surprise': 'negative'}, # 소매판매
            'industrial_production': {'positive_surprise': 'positive', 'negative_surprise': 'negative'}, # 산업생산
            'trade_balance': {'positive_surprise': 'positive', 'negative_surprise': 'negative'}, # 무역수지 (흑자 확대가 긍정적)
            'consumer_confidence': {'positive_surprise': 'positive', 'negative_surprise': 'negative'}, # 소비자신뢰지수
            'pmi': {'positive_surprise': 'positive', 'negative_surprise': 'negative'} # 구매관리자지수
            # 추가적인 이벤트 카테고리와 규칙 정의 필요
        }

        logger.info("IntegratedAnalyzer 초기화 완료.")

    def analyze_symbol_impact(self,
                              symbol: str,
                              recent_news_df: Optional[pd.DataFrame],
                              upcoming_economic_events_df: Optional[pd.DataFrame],
                              current_price_data: Optional[Dict[str, Any]] = None,
                              historical_data_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        특정 심볼에 대한 뉴스 감성 및 주요 경제 지표의 통합적 영향을 분석합니다.

        Args:
            symbol (str): 분석 대상 심볼 (예: "AAPL").
            recent_news_df (pd.DataFrame): 최근 뉴스 데이터 (NewsSentimentAnalyzer 분석 결과 포함).
                                           필수 컬럼: 'symbol', 'published_at', 'sentiment_score', 'sentiment_label', 'title', 'url', 'source_name'.
            upcoming_economic_events_df (pd.DataFrame): 향후 주요 경제 지표 데이터.
                                                        필수 컬럼: 'datetime', 'country_code', 'event', 'importance', 'actual_raw', 'forecast_raw', 'previous_raw'.
            current_price_data (Dict, optional): 현재 주가 정보 (실시간 호가).
            historical_data_df (pd.DataFrame, optional): 기술적 지표 포함된 과거 주가 데이터.

        Returns:
            Dict[str, Any]: 통합 분석 결과.
                - 'symbol': 분석 대상 심볼
                - 'overall_sentiment_score': 종합 감성 점수 (-1.0 ~ 1.0)
                - 'overall_sentiment_label': 종합 감성 라벨 ('positive', 'neutral', 'negative')
                - 'news_analysis': 뉴스 분석 요약
                - 'economic_event_analysis': 경제 지표 분석 요약
                - 'key_risk_factors': 주요 리스크 요인 (텍스트 설명)
                - 'key_positive_factors': 주요 긍정적 요인 (텍스트 설명)
                - 'short_term_outlook_label': 단기 전망 라벨 ('매우 긍정적', '긍정적', '중립적', '부정적', '매우 부정적')
                - 'confidence': 분석 신뢰도 (0.0 ~ 1.0)
        """
        logger.info(f"'{symbol}'에 대한 통합 영향 분석 시작...")

        analysis_result = {
            'symbol': symbol,
            'overall_sentiment_score': 0.0,
            'overall_sentiment_label': 'neutral',
            'news_analysis': {'summary': "뉴스 데이터 부족", 'sentiment_score': 0.0, 'recent_headlines': []},
            'economic_event_analysis': {'summary': "경제 이벤트 데이터 부족", 'impact_score': 0.0, 'upcoming_critical_events': []},
            'key_risk_factors': [],
            'key_positive_factors': [],
            'short_term_outlook_label': '데이터 부족',
            'confidence': 0.0
        }

        # 1. 뉴스 감성 분석 요약
        news_sentiment_score = 0.0
        news_confidence_total = 0.0
        news_count = 0
        
        if recent_news_df is not None and not recent_news_df.empty:
            # sentiment_score 컬럼 존재 여부 확인
            if 'sentiment_score' not in recent_news_df.columns:
                logger.warning(f"뉴스 데이터에 'sentiment_score' 컬럼이 없습니다. 감성 분석을 수행합니다.")
                # 감성 분석 수행
                if self.news_analyzer and hasattr(recent_news_df, 'text_for_analysis'):
                    analyzed_news = []
                    for idx, row in recent_news_df.iterrows():
                        text = row.get('text_for_analysis') or row.get('description') or row.get('title', '')
                        if text:
                            sentiment_result = self.news_analyzer.analyze_sentiment(text, symbol)
                            row['sentiment_score'] = sentiment_result.get('sentiment_score', 0.0)
                            row['sentiment_label'] = sentiment_result.get('sentiment_label', 'neutral')
                            row['confidence'] = sentiment_result.get('confidence', 0.5)
                        else:
                            row['sentiment_score'] = 0.0
                            row['sentiment_label'] = 'neutral'
                            row['confidence'] = 0.0
                        analyzed_news.append(row)
                    recent_news_df = pd.DataFrame(analyzed_news)
            
            # symbol 필터링
            if 'symbol' in recent_news_df.columns:
                symbol_news_df = recent_news_df[recent_news_df['symbol'] == symbol].copy()
            else:
                symbol_news_df = recent_news_df.copy()

            if not symbol_news_df.empty and 'sentiment_score' in symbol_news_df.columns:
                # sentiment_score가 있는 행만 선택
                valid_sentiment_df = symbol_news_df.dropna(subset=['sentiment_score'])
                if not valid_sentiment_df.empty:
                    news_sentiment_score = valid_sentiment_df['sentiment_score'].mean()
                    if 'confidence' in valid_sentiment_df.columns:
                        news_confidence_total = valid_sentiment_df['confidence'].sum()
                    else:
                        news_confidence_total = 0.5 * len(valid_sentiment_df)
                    news_count = len(valid_sentiment_df)
                    
                    analysis_result['news_analysis']['sentiment_score'] = round(news_sentiment_score, 3)
                    analysis_result['news_analysis']['summary'] = f"최근 {news_count}개 뉴스 평균 감성: {news_sentiment_score:.2f} ({self._get_sentiment_label_from_score(news_sentiment_score)})"
                    # 최근 주요 헤드라인 (최대 3개)
                    recent_headlines = []
                    for _, row in symbol_news_df.sort_values(by='published_at', ascending=False).head(3).iterrows():
                        recent_headlines.append({
                            'title': row.get('title', 'N/A'),
                            'url': row.get('url', ''),
                            'source': row.get('source_name', row.get('source', {}).get('name', 'N/A') if isinstance(row.get('source'), dict) else 'N/A'),
                            'sentiment': row.get('sentiment_label', 'neutral'),
                            'score': round(row.get('sentiment_score', 0.0), 2)
                        })
                    analysis_result['news_analysis']['recent_headlines'] = recent_headlines
                    logger.debug(f"'{symbol}' 뉴스 분석: 평균 감성 {news_sentiment_score:.2f}, 뉴스 {news_count}개.")
                else:
                    analysis_result['news_analysis']['summary'] = f"'{symbol}' 관련 최근 뉴스 없음."
                    logger.debug(f"'{symbol}' 관련 최근 뉴스 없음.")
            else:
                logger.debug("입력된 뉴스 데이터 없음.")

        # 2. 경제 지표 영향 분석
        economic_impact_score = 0.0
        economic_confidence_total = 0.0
        economic_event_count = 0
        if upcoming_economic_events_df is not None and not upcoming_economic_events_df.empty:
            critical_events_info = []
            # 중요도(importance)와 국가(country_code)를 기준으로 필터링 및 가중치 적용
            for _, event_row in upcoming_economic_events_df.iterrows():
                if event_row['importance'] >= self.economic_event_importance_threshold:
                    country = event_row['country_code']
                    country_weight = self.country_market_impact_weights.get(country, 0.1) # 정의되지 않은 국가는 낮은 가중치
                    event_name_lower = str(event_row['event']).lower()

                    # 이벤트 카테고리 또는 이름으로 예상 영향 추론
                    # 이 부분은 더 정교한 규칙 기반 또는 모델 기반 시스템으로 발전 가능
                    event_impact_direction = 0 # 1: positive, -1: negative, 0: neutral/unknown
                    surprise_factor = 0.0 # -1 (negative surprise) to 1 (positive surprise)

                    # 실제값이 있고, 예상값이 숫자인 경우 서프라이즈 계산
                    if pd.notna(event_row['actual_raw']) and event_row['actual_raw'] not in ['-', ''] and \
                       pd.notna(event_row['forecast_raw']) and event_row['forecast_raw'] not in ['-', ''] :
                        try:
                            actual_val = float(str(event_row['actual_raw']).replace('%','').replace('K','000').replace('M','000000').replace('B','000000000')) # 간단 변환
                            forecast_val = float(str(event_row['forecast_raw']).replace('%','').replace('K','000').replace('M','000000').replace('B','000000000'))
                            if forecast_val != 0 : # 0으로 나누기 방지
                                surprise_factor = (actual_val - forecast_val) / abs(forecast_val)
                                surprise_factor = np.clip(surprise_factor, -2, 2) # 극단값 제한 (-200% ~ +200% 범위로)
                        except ValueError:
                            logger.debug(f"경제지표 값 숫자 변환 실패: actual='{event_row['actual_raw']}', forecast='{event_row['forecast_raw']}'")


                    # 이벤트명 기반 영향 방향 결정 (단순 예시)
                    # TODO: event_category_impact_direction 와 surprise_factor를 사용하여 event_impact_direction 결정
                    # 예: CPI가 예상보다 높게 나오면 (positive_surprise), 주식에는 부정적 (-1)
                    matched_category = None
                    for cat_key in self.event_category_impact_direction.keys():
                        if cat_key in event_name_lower:
                            matched_category = cat_key
                            break
                    
                    if matched_category:
                        rules = self.event_category_impact_direction[matched_category]
                        if surprise_factor > 0.1 and rules.get('positive_surprise') == 'positive': event_impact_direction = 1
                        elif surprise_factor > 0.1 and rules.get('positive_surprise') == 'negative_stock': event_impact_direction = -1
                        elif surprise_factor < -0.1 and rules.get('negative_surprise') == 'positive_stock': event_impact_direction = 1
                        elif surprise_factor < -0.1 and rules.get('negative_surprise') == 'negative': event_impact_direction = -1
                        elif 'hike' in event_name_lower and rules.get('hike') == 'negative_stock': event_impact_direction = -1 # 금리 인상
                        elif 'cut' in event_name_lower and rules.get('cut') == 'positive_stock': event_impact_direction = 1 # 금리 인하
                        # ... 기타 규칙들 ...

                    # 개별 이벤트의 영향 점수 (중요도 * 국가 가중치 * 방향성 * 서프라이즈 강도)
                    # 서프라이즈 강도는 절대값으로 하여 영향의 크기를 반영하고, 방향은 event_impact_direction으로
                    event_score = (event_row['importance'] / 3.0) * country_weight * event_impact_direction * (1 + abs(surprise_factor)/2) # 서프라이즈 강도 반영
                    economic_impact_score += event_score
                    economic_confidence_total += (event_row['importance'] / 3.0) * country_weight # 신뢰도는 중요도와 국가 가중치 기반
                    economic_event_count += 1

                    event_info = {
                        'datetime': event_row['datetime'].strftime('%Y-%m-%d %H:%M'),
                        'country': country,
                        'event_name': event_row['event'],
                        'importance': event_row['importance'],
                        'actual': event_row['actual_raw'],
                        'forecast': event_row['forecast_raw'],
                        'impact_direction_assumed': event_impact_direction,
                        'calculated_event_score': round(event_score, 2)
                    }
                    critical_events_info.append(event_info)

                    if event_score > 0.3: # 예: 개별 이벤트 영향이 클 경우
                        analysis_result['key_positive_factors'].append(f"경제지표 호조: {country} {event_row['event']} (실제: {event_row['actual_raw']}, 예상: {event_row['forecast_raw']})")
                    elif event_score < -0.3:
                        analysis_result['key_risk_factors'].append(f"경제지표 부진: {country} {event_row['event']} (실제: {event_row['actual_raw']}, 예상: {event_row['forecast_raw']})")

            analysis_result['economic_event_analysis']['upcoming_critical_events'] = sorted(critical_events_info, key=lambda x: (x['datetime'], -x['importance']))
            if economic_event_count > 0:
                # economic_impact_score는 누적된 점수이므로, 평균을 내거나 정규화 필요
                # 여기서는 단순 합산된 값을 정규화 (tanh)
                normalized_economic_impact = np.tanh(economic_impact_score / (economic_event_count * 0.5)) # 분모는 경험적 조정
                analysis_result['economic_event_analysis']['impact_score'] = round(normalized_economic_impact, 3)
                analysis_result['economic_event_analysis']['summary'] = f"{economic_event_count}개 주요 경제 이벤트 분석됨. 예상 영향 점수: {normalized_economic_impact:.2f}"
                logger.debug(f"경제 이벤트 분석: 누적 점수 {economic_impact_score:.2f}, 정규화된 영향 {normalized_economic_impact:.2f}, 이벤트 수 {economic_event_count}.")
            else:
                analysis_result['economic_event_analysis']['summary'] = "분석 대상 주요 경제 이벤트 없음."
                logger.debug("분석 대상 주요 경제 이벤트 없음.")
        else:
            logger.debug("입력된 경제 이벤트 데이터 없음.")


        # 3. (향후 추가) 기술적 지표 분석 요약
        # technical_analysis_summary = "기술적 분석 데이터 부족"
        # if historical_data_df is not None and not historical_data_df.empty:
        #     # 예: 주요 지표 (RSI, MACD) 상태, 지지/저항선 등 분석
        #     # technical_analysis_summary = self._analyze_technical_state(historical_data_df)
        #     pass
        # analysis_result['technical_analysis'] = {'summary': technical_analysis_summary}

        # 4. 종합 점수 및 신뢰도 계산
        # 가중치: 뉴스 40%, 경제지표 60% (예시)
        # 각 분석의 신뢰도(count 기반)를 가중치로 사용할 수도 있음
        total_confidence_factor = news_confidence_total + economic_confidence_total
        if total_confidence_factor > 0:
            weighted_news_score = (news_sentiment_score * news_confidence_total) / total_confidence_factor
            weighted_economic_score = (analysis_result['economic_event_analysis']['impact_score'] * economic_confidence_total) / total_confidence_factor
            analysis_result['overall_sentiment_score'] = round(weighted_news_score + weighted_economic_score, 3) # 단순 합산 후 범위 조정 필요 시 np.tanh
            analysis_result['confidence'] = round(total_confidence_factor / (news_count + economic_event_count + 1e-6), 3) # 평균적인 신뢰도
        elif news_count > 0 : # 뉴스만 있을 경우
            analysis_result['overall_sentiment_score'] = round(news_sentiment_score, 3)
            analysis_result['confidence'] = round(news_confidence_total / (news_count + 1e-6), 3)
        elif economic_event_count > 0: # 경제지표만 있을 경우
            analysis_result['overall_sentiment_score'] = round(analysis_result['economic_event_analysis']['impact_score'], 3)
            analysis_result['confidence'] = round(economic_confidence_total / (economic_event_count + 1e-6), 3)


        analysis_result['overall_sentiment_label'] = self._get_sentiment_label_from_score(analysis_result['overall_sentiment_score'])

        # 5. 단기 전망 라벨 결정
        score = analysis_result['overall_sentiment_score']
        confidence = analysis_result['confidence']
        if score > 0.5 and confidence > 0.6:
            analysis_result['short_term_outlook_label'] = '매우 긍정적'
        elif score > 0.15 and confidence > 0.5:
            analysis_result['short_term_outlook_label'] = '긍정적'
        elif score < -0.5 and confidence > 0.6:
            analysis_result['short_term_outlook_label'] = '매우 부정적'
        elif score < -0.15 and confidence > 0.5:
            analysis_result['short_term_outlook_label'] = '부정적'
        elif confidence < 0.3: # 신뢰도가 너무 낮으면
            analysis_result['short_term_outlook_label'] = '판단 보류 (신뢰도 낮음)'
        else:
            analysis_result['short_term_outlook_label'] = '중립적'

        # 주요 요인 요약
        if news_sentiment_score > 0.2:
             analysis_result['key_positive_factors'].append(f"긍정적 뉴스 감성 ({news_sentiment_score:.2f})")
        elif news_sentiment_score < -0.2:
             analysis_result['key_risk_factors'].append(f"부정적 뉴스 감성 ({news_sentiment_score:.2f})")

        logger.info(f"'{symbol}' 통합 분석 완료: 전망 '{analysis_result['short_term_outlook_label']}', 종합점수 {analysis_result['overall_sentiment_score']:.2f} (신뢰도 {analysis_result['confidence']:.2f})")
        return analysis_result

    def _get_sentiment_label_from_score(self, score: float, positive_threshold=0.15, negative_threshold=-0.15) -> str:
        """점수로부터 감성 라벨을 반환합니다."""
        if score > positive_threshold:
            return 'positive'
        elif score < negative_threshold:
            return 'negative'
        else:
            return 'neutral'

    # --- 예시: 특정 주요 경제 지표 발표가 심볼 뉴스 감성에 미치는 영향 분석 ---
    def analyze_event_impact_on_news_sentiment(self,
                                               symbol: str,
                                               event_datetime: datetime,
                                               event_name: str,
                                               news_df: pd.DataFrame, # NewsSentimentAnalyzer 결과 DF
                                               time_window_hours: int = 24) -> Optional[Dict]:
        """
        특정 경제 지표 발표 전후의 뉴스 감성 변화를 분석합니다.

        Args:
            symbol: 분석 대상 심볼.
            event_datetime: 경제 지표 발표 시각.
            event_name: 경제 지표명.
            news_df: 분석할 뉴스 데이터 (sentiment_score, published_at 포함).
            time_window_hours: 이벤트 전후로 분석할 시간 윈도우 (시간 단위).

        Returns:
            분석 결과 딕셔너리 또는 None.
        """
        if news_df is None or news_df.empty or 'published_at' not in news_df.columns or 'sentiment_score' not in news_df.columns:
            logger.warning(f"뉴스 데이터 부족으로 '{event_name}' 영향 분석 불가 ({symbol})")
            return None

        logger.info(f"'{symbol}' 대상, '{event_name}'(@{event_datetime.strftime('%Y-%m-%d %H:%M')}) 발표 전후 뉴스 감성 분석 시작 (윈도우: {time_window_hours}시간)")

        # 시간 변환 및 필터링
        news_df['published_at'] = pd.to_datetime(news_df['published_at'])
        event_dt_utc = pd.to_datetime(event_datetime, utc=True) # 이벤트 시간을 UTC로 가정 또는 변환

        before_event_start = event_dt_utc - timedelta(hours=time_window_hours)
        after_event_end = event_dt_utc + timedelta(hours=time_window_hours)

        news_before_event = news_df[
            (news_df['published_at'] >= before_event_start) &
            (news_df['published_at'] < event_dt_utc) &
            (news_df.get('symbol', symbol) == symbol) # news_df에 symbol 컬럼이 있다면 사용
        ]

        news_after_event = news_df[
            (news_df['published_at'] > event_dt_utc) &
            (news_df['published_at'] <= after_event_end) &
            (news_df.get('symbol', symbol) == symbol)
        ]

        avg_sentiment_before = news_before_event['sentiment_score'].mean() if not news_before_event.empty else None
        avg_sentiment_after = news_after_event['sentiment_score'].mean() if not news_after_event.empty else None

        result = {
            'symbol': symbol,
            'event_name': event_name,
            'event_datetime': event_datetime.strftime('%Y-%m-%d %H:%M'),
            'avg_sentiment_before': round(avg_sentiment_before, 3) if pd.notna(avg_sentiment_before) else None,
            'avg_sentiment_after': round(avg_sentiment_after, 3) if pd.notna(avg_sentiment_after) else None,
            'news_count_before': len(news_before_event),
            'news_count_after': len(news_after_event),
            'sentiment_change': None
        }

        if pd.notna(avg_sentiment_before) and pd.notna(avg_sentiment_after):
            result['sentiment_change'] = round(avg_sentiment_after - avg_sentiment_before, 3)
            logger.info(f"분석 결과: 이전 감성 {result['avg_sentiment_before']}, 이후 감성 {result['avg_sentiment_after']}, 변화 {result['sentiment_change']}")
        else:
            logger.info("이벤트 전 또는 후의 뉴스 데이터가 부족하여 감성 변화를 계산할 수 없습니다.")

        return result


if __name__ == '__main__':
    # 로깅 설정 (테스트용)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # --- 테스트 데이터 생성 ---
    # 1. 뉴스 데이터 (NewsSentimentAnalyzer 결과와 유사한 형태)
    sample_news_data = [
        {'symbol': 'AAPL', 'published_at': datetime.now() - timedelta(hours=5), 'sentiment_score': 0.6, 'sentiment_label': 'positive', 'confidence': 0.8, 'title': 'Apple reports record profits', 'url': 'http://example.com/news1', 'source_name': 'TechNews'},
        {'symbol': 'AAPL', 'published_at': datetime.now() - timedelta(hours=2), 'sentiment_score': -0.3, 'sentiment_label': 'negative', 'confidence': 0.7, 'title': 'Supply chain issues for Apple', 'url': 'http://example.com/news2', 'source_name': 'BizTimes'},
        {'symbol': 'MSFT', 'published_at': datetime.now() - timedelta(hours=10), 'sentiment_score': 0.8, 'sentiment_label': 'positive', 'confidence': 0.9, 'title': 'Microsoft Azure growth accelerates', 'url': 'http://example.com/news3', 'source_name': 'CloudWeekly'},
    ]
    test_news_df = pd.DataFrame(sample_news_data)
    test_news_df['published_at'] = pd.to_datetime(test_news_df['published_at'], utc=True)


    # 2. 경제 지표 데이터 (EconomicCalendarCollector 결과와 유사한 형태)
    sample_economic_events = [
        {'datetime': datetime.now() + timedelta(hours=3), 'country_code': 'US', 'event': 'Non-Farm Payrolls', 'importance': 3, 'actual_raw': '', 'forecast_raw': '250K', 'previous_raw': '180K'},
        {'datetime': datetime.now() + timedelta(days=1), 'country_code': 'EU', 'event': 'ECB Interest Rate Decision', 'importance': 3, 'actual_raw': '', 'forecast_raw': '0.50%', 'previous_raw': '0.50%'},
        {'datetime': datetime.now() + timedelta(hours=10), 'country_code': 'US', 'event': 'CPI m/m', 'importance': 2, 'actual_raw': '0.4%', 'forecast_raw': '0.3%', 'previous_raw': '0.2%'}, # 이미 발표된 것으로 가정
    ]
    test_econ_df = pd.DataFrame(sample_economic_events)
    test_econ_df['datetime'] = pd.to_datetime(test_econ_df['datetime'], utc=True)


    # --- IntegratedAnalyzer 인스턴스 생성 및 테스트 ---
    # NewsSentimentAnalyzer는 실제 모델 로딩 때문에 시간이 걸릴 수 있으므로, None으로 두고 테스트하거나 mock 객체 사용 가능
    # 여기서는 실제 NewsSentimentAnalyzer를 사용 (API키 설정이 되어 있어야 NewsAPI 사용 가능)
    try:
        news_analyzer = NewsSentimentAnalyzer() # 실제 사용 시 APIConfig 객체 전달 필요
        analyzer = IntegratedAnalyzer(news_sentiment_analyzer=news_analyzer)
    except Exception as e:
        logger.error(f"NewsSentimentAnalyzer 초기화 중 오류 (모델 다운로드 등). 테스트를 위해 None으로 설정: {e}")
        analyzer = IntegratedAnalyzer(news_sentiment_analyzer=None)


    # 1. 특정 심볼 통합 분석 테스트
    aapl_analysis = analyzer.analyze_symbol_impact(
        symbol='AAPL',
        recent_news_df=test_news_df,
        upcoming_economic_events_df=test_econ_df
    )
    print("\n--- AAPL 통합 분석 결과 ---")
    if aapl_analysis:
        for key, value in aapl_analysis.items():
            if isinstance(value, list) and value:
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")

    msft_analysis = analyzer.analyze_symbol_impact(
        symbol='MSFT',
        recent_news_df=test_news_df, # MSFT 관련 뉴스도 포함됨
        upcoming_economic_events_df=test_econ_df
    )
    print("\n--- MSFT 통합 분석 결과 ---")
    if msft_analysis:
        for key, value in msft_analysis.items():
            if isinstance(value, list) and value:
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")

    # 2. 특정 경제 지표 발표가 뉴스 감성에 미치는 영향 분석 테스트
    # 예시: 'CPI m/m' 발표 전후 AAPL 뉴스 감성 분석
    # 실제로는 DB에서 해당 기간의 AAPL 뉴스 데이터를 충분히 가져와야 함
    cpi_event_time = datetime.now() + timedelta(hours=10) # 위 테스트 데이터의 CPI 발표 시간
    # 이 테스트를 위해서는 test_news_df에 CPI 발표 전후의 AAPL 뉴스가 더 많이 필요함
    # 아래는 개념 증명용.
    extended_news_data_for_event_analysis = sample_news_data + [
        {'symbol': 'AAPL', 'published_at': cpi_event_time - timedelta(hours=12), 'sentiment_score': 0.1, 'sentiment_label': 'neutral', 'confidence': 0.6, 'title': 'Apple pre-CPI outlook', 'url': 'http://example.com/news_pre', 'source_name': 'MarketWatch'},
        {'symbol': 'AAPL', 'published_at': cpi_event_time + timedelta(hours=2), 'sentiment_score': -0.5, 'sentiment_label': 'negative', 'confidence': 0.85, 'title': 'Apple stock dips after CPI data', 'url': 'http://example.com/news_post_cpi', 'source_name': 'Reuters'},
    ]
    test_news_df_for_event = pd.DataFrame(extended_news_data_for_event_analysis)
    test_news_df_for_event['published_at'] = pd.to_datetime(test_news_df_for_event['published_at'], utc=True)


    cpi_impact_on_aapl_news = analyzer.analyze_event_impact_on_news_sentiment(
        symbol='AAPL',
        event_datetime=cpi_event_time,
        event_name='CPI m/m',
        news_df=test_news_df_for_event, # 충분한 뉴스 데이터 필요
        time_window_hours=24
    )
    print("\n--- CPI 발표가 AAPL 뉴스 감성에 미치는 영향 분석 결과 ---")
    print(cpi_impact_on_aapl_news)