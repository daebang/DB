# data/collectors/news_collector.py

import requests
from bs4 import BeautifulSoup # BeautifulSoup 추가
import trafilatura # trafilatura 추가
from trafilatura.settings import use_config # trafilatura 설정용
from typing import Optional, List, Dict
import logging
import pandas as pd # pandas 임포트 추가 (to_datetime 사용)


logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self, api_config): # APIConfig 객체 주입
        self.api_key = api_config.news_api_key if api_config else None
        if not self.api_key:
            logger.error("News API 키가 설정되지 않았습니다. 뉴스 수집이 제한됩니다.")
        self.base_url = "https://newsapi.org/v2/everything"

        # trafilatura 설정 (기본 설정 사용 또는 필요시 커스터마이징)
        self.trafilatura_config = use_config()
        # self.trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "7") # 타임아웃 설정 (초)
        # self.trafilatura_config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "150") # 최소 추출 텍스트 크기

    def get_latest_news_by_keywords(self, keywords: str, language: str = "en", page_size: int = 10, sources: Optional[str] = None) -> Optional[List[Dict]]:
        """
        NewsAPI를 사용하여 특정 키워드에 대한 최신 뉴스 기사 수집.
        keywords: 검색어 (e.g., "Apple OR AAPL stock")
        language: 'en', 'ko' 등
        page_size: 가져올 기사 수
        sources: 특정 뉴스 출처 지정 (e.g., 'bloomberg,reuters')
        """
        if not self.api_key:
            logger.warning("News API 키가 없어 뉴스 조회 불가")
            return None

        params = {
            "q": keywords,
            "language": language,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "apiKey": self.api_key
        }
        if sources:
            params["sources"] = sources

        logger.debug(f"News API 호출: 키워드='{keywords}', 언어='{language}', 출처='{sources}'")
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                logger.error(f"News API 오류: {data.get('code')} - {data.get('message')}")
                return None

            raw_articles = data.get("articles", [])
            processed_articles = []

            for article_data in raw_articles:
                title = article_data.get('title')
                url = article_data.get('url')
                content_snippet = article_data.get('content')
                description = article_data.get('description')

                text_for_analysis = content_snippet if content_snippet and len(content_snippet) > 50 else description

                # 더 나은 본문 추출을 위해 trafilatura 사용 (선택적)
                full_text = self.fetch_article_full_text_with_trafilatura(url)
                if full_text and len(full_text) > len(text_for_analysis or ""): # trafilatura 결과가 더 좋으면 사용
                    text_for_analysis = full_text
                elif not text_for_analysis and full_text: # 기존 분석용 텍스트가 아예 없으면 trafilatura 결과 사용
                    text_for_analysis = full_text


                processed_articles.append({
                    'title': title,
                    'description': description,
                    'url': url,
                    'source_name': article_data.get('source', {}).get('name'),
                    'published_at': pd.to_datetime(article_data.get('publishedAt'), errors='coerce', utc=True),
                    'content_snippet': content_snippet,
                    'text_for_analysis': text_for_analysis, # 감성 분석 등에 사용할 최종 텍스트
                    'keywords_used': keywords
                })
            logger.info(f"'{keywords}' 키워드로 {len(processed_articles)}개 뉴스 수집 완료")
            return processed_articles

        except requests.exceptions.RequestException as e:
            logger.error(f"News API 요청 중 네트워크 오류 발생 ({keywords}): {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"뉴스 처리 중 예상치 못한 오류 발생 ({keywords}): {e}", exc_info=True)
            return None

    def fetch_article_full_text_with_trafilatura(self, url: str) -> Optional[str]:
        """
        requests로 HTML을 가져오고 trafilatura를 사용하여 기사 본문을 추출합니다.
        """
        if not url:
            return None
        logger.debug(f"Trafilatura로 기사 본문 수집 시도: {url}")
        try:
            # requests로 웹 페이지 다운로드
            headers = { # 일부 웹사이트는 User-Agent를 확인
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10) # 타임아웃 설정
            response.raise_for_status() # HTTP 오류 발생 시 예외 처리
            html_content = response.text

            # trafilatura로 본문 추출
            # include_comments=False, include_tables=False 등으로 원치 않는 내용 제외 가능
            # output_format='txt' (기본값) 또는 'xml', 'json' 등
            extracted_text = trafilatura.extract(
                html_content,
                config=self.trafilatura_config,
                include_comments=False,
                include_tables=True, # 테이블 내용 포함 여부
                # favor_precision=True # 정확도 우선 (더 짧은 텍스트가 나올 수 있음)
            )

            if extracted_text:
                # BeautifulSoup을 사용하여 HTML 태그 한 번 더 정리 (선택 사항)
                # soup = BeautifulSoup(extracted_text, "html.parser")
                # cleaned_text = soup.get_text(separator="\n", strip=True)
                # return cleaned_text
                return extracted_text # trafilatura가 이미 텍스트 위주로 반환
            else:
                logger.debug(f"Trafilatura가 {url} 에서 본문을 추출하지 못했습니다.")
                return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Trafilatura 위한 HTML 다운로드 실패 ({url}): {e}")
            return None
        except Exception as e:
            logger.warning(f"Trafilatura로 기사 본문 추출 실패 ({url}): {e}")
            return None

# 사용 예시 (테스트용)
if __name__ == '__main__':
    class DummyAPIConfig: # settings.py의 APIConfig 사용 권장
        news_api_key = "YOUR_NEWS_API_KEY" # 실제 NewsAPI 키로 교체

    if DummyAPIConfig.news_api_key == "YOUR_NEWS_API_KEY":
        print("테스트를 위해 DummyAPIConfig.news_api_key에 실제 NewsAPI 키를 입력하세요.")
    else:
        news_collector = NewsCollector(DummyAPIConfig())

        # 특정 키워드로 뉴스 검색 (영어)
        apple_news = news_collector.get_latest_news_by_keywords(keywords="NVIDIA earnings OR NVDA stock", language="en", page_size=2)
        if apple_news:
            print("\n--- NVIDIA 관련 최신 뉴스 (영어, trafilatura 사용) ---")
            for idx, news_item in enumerate(apple_news):
                print(f"\n--- 뉴스 #{idx+1} ---")
                print(f"제목: {news_item['title']}")
                print(f"출처: {news_item['source_name']} ({news_item['published_at']})")
                print(f"URL: {news_item['url']}")
                print(f"NewsAPI 제공 snippet: {news_item['content_snippet']}")
                print(f"분석용 텍스트 (trafilatura 시도 후):\n{news_item['text_for_analysis'][:400]}...") # 400자까지만 출력
                print("-" * 30)

        # 특정 URL에서 본문 추출 테스트
        # test_url_yna = "https://www.yna.co.kr/view/AKR20231026090400003" # 연합뉴스
        # test_url_reuters = "https://www.reuters.com/technology/nvidia-quarterly-revenue-results-beat-estimates-2023-11-21/" # 로이터
        # for url_to_test in [test_url_yna, test_url_reuters]:
        #     print(f"\n--- URL 본문 추출 테스트: {url_to_test} ---")
        #     full_text = news_collector.fetch_article_full_text_with_trafilatura(url_to_test)
        #     if full_text:
        #         print(full_text[:500] + "...")
        #     else:
        #         print("본문 추출 실패 또는 내용 없음.")