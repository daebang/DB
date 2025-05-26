# data/collectors/economic_calendar_collector.py
import requests # requests는 현재 코드에서 직접 사용되진 않지만, 향후 XHR 방식으로 변경 시 필요할 수 있어 유지합니다.
from bs4 import BeautifulSoup # BeautifulSoup도 현재 Selenium 위주 코드에서는 직접 사용되지 않지만, 향후 부분적 HTML 처리 등에 사용될 수 있어 유지합니다.
import pandas as pd
from datetime import datetime, timedelta
import json # json도 현재 코드에서 직접 사용되진 않지만, 향후 XHR 방식의 JSON 응답 처리에 필요할 수 있어 유지합니다.
import time # 명시적 대기로 최대한 대체했으나, 불가피한 경우 최소한으로 사용될 수 있습니다.
from typing import List, Dict, Optional, Tuple
import logging
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService # ChromeDriver 경로 관리를 위해 추가
from webdriver_manager.chrome import ChromeDriverManager # ChromeDriver 자동 관리를 위해 추가
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

logger = logging.getLogger(__name__)

class EconomicCalendarCollector:
    """경제 지표 캘린더 데이터 수집기"""

    def __init__(self, api_config=None): # api_config는 현재 사용되지 않지만, 향후를 위해 유지
        self.api_config = api_config
        self.base_url = "https://kr.investing.com/economic-calendar/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7', # 보다 일반적인 Accept-Language
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }

        # HTML의 title 속성(한글 국가명)을 키로 사용하도록 country_codes 수정/확장 필요
        self.country_codes = {
            '가나': 'GH', '그리스': 'GR', '나미비아': 'NA', '나이지리아': 'NG',
            '남아프리카 공화국': 'ZA', '네덜란드': 'NL', '노르웨이': 'NO', '뉴질랜드': 'NZ',
            '대만': 'TW', '덴마크': 'DK', '독일': 'DE', '라트비아': 'LV',
            '러시아': 'RU', '레바논': 'LB', '루마니아': 'RO', '룩셈부르크': 'LU',
            '르완다': 'RW', '리투아니아': 'LT', '말라위': 'MW', '말레이시아': 'MY',
            '멕시코': 'MX', '모로코': 'MA', '모리셔스': 'MU', '모잠비크': 'MZ',
            '몬테네그로': 'ME', '몰타': 'MT', '몽골': 'MN', '미국': 'US',
            '바레인': 'BH', '방글라데시': 'BD', '버뮤다': 'BM', '베네수엘라': 'VE',
            '베트남': 'VN', '벨기에': 'BE', '보스니아': 'BA', '보츠와나': 'BW',
            '불가리아': 'BG', '브라질': 'BR', '사우디 아라비아': 'SA', '세르비아': 'RS',
            '스리랑카': 'LK', '스웨덴': 'SE', '스위스': 'CH', '스페인': 'ES',
            '슬로바키아': 'SK', '슬로베니아': 'SI', '싱가폴': 'SG', '아랍 에미리트': 'AE',
            '아르헨티나': 'AR', '아이슬란드': 'IS', '아일랜드': 'IE', '아제르바이잔': 'AZ',
            '알바니아': 'AL', '앙골라': 'AO', '에스토니아': 'EE', '에콰도르': 'EC',
            '영국': 'GB', '오만': 'OM', '오스트리아': 'AT', '요르단': 'JO',
            '우간다': 'UG', '우루과이': 'UY', '우즈베키스탄': 'UZ', '우크라이나': 'UA',
            '유로 지역': 'EU', '이라크': 'IQ', '이스라엘': 'IL', '이집트': 'EG',
            '이탈리아': 'IT', '인도': 'IN', '인도네시아': 'ID', '일본': 'JP',
            '자메이카': 'JM', '잠비아': 'ZM', '중국': 'CN', '짐바브웨': 'ZW',
            '체코': 'CZ', '칠레': 'CL', '카자흐스탄': 'KZ', '카타르': 'QA',
            '캐나다': 'CA', '케냐': 'KE', '케이먼 제도': 'KY', '코스타리카': 'CR',
            '코트 디부 아르': 'CI', '콜롬비아': 'CO', '쿠웨이트': 'KW', '크로아티아': 'HR',
            '키르기스스탄': 'KG', '키프로스': 'CY', '탄자니아': 'TZ', '태국': 'TH',
            '튀니지': 'TN', '튀르키예': 'TR', '파라과이': 'PY', '파키스탄': 'PK',
            '팔레스타인 자치 정부': 'PS', '페루': 'PE', '포르투갈': 'PT', '폴란드': 'PL',
            '프랑스': 'FR', '핀란드': 'FI', '필리핀': 'PH', '한국': 'KR',
            '헝가리': 'HU', '호주': 'AU', '홍콩': 'HK'
            # 필요한 경우 더 추가
        }
        # 중요도 아이콘 클래스 (bull1, bull2, bull3 대신 아이콘 개수로 파악)
        self.importance_icon_class = "grayFullBullishIcon"


    def _setup_driver(self) -> webdriver.Chrome:
        """Selenium WebDriver 설정 및 반환"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 백그라운드 실행
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument(f'user-agent={self.headers["User-Agent"]}')
        chrome_options.add_argument(f'accept-language={self.headers["Accept-Language"]}')

        try:
            # webdriver-manager를 사용하여 ChromeDriver 자동 관리
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("ChromeDriver (headless) 설정 완료 (webdriver-manager 사용)")
            return driver
        except Exception as e:
            logger.error(f"ChromeDriver 설정 중 오류 발생: {e}", exc_info=True)
            raise  # 오류 발생 시 재호출하여 프로그램이 인지하도록 함

    def fetch_calendar_data(self, start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            countries: Optional[List[str]] = None, # 여기서 countries는 KR, US 등 표준 코드로 기대
                            importance_min: int = 1) -> pd.DataFrame: # 기본 최소 중요도를 1 (낮음)로 변경
        """
        경제 지표 캘린더 데이터 수집

        Args:
            start_date: 시작 날짜 (기본: 오늘)
            end_date: 종료 날짜 (기본: 오늘) # 기본값을 오늘로 변경, 필요시 조정
            countries: 필터링할 국가 코드 리스트 (예: ['US', 'EU', 'CN', 'KR']). 기본: 모든 주요 국가 포함
            importance_min: 최소 중요도 (1-낮음, 2-중간, 3-높음)

        Returns:
            경제 지표 데이터프레임
        """
        if start_date is None:
            start_date = datetime.now()
        if end_date is None:
            # end_date = start_date + timedelta(days=7) # 이전 기본값
            end_date = start_date # 기본적으로 당일 데이터만 가져오도록 변경. 여러 날짜는 명시적으로 지정.
        
        # 기본 국가 필터 (주요 국가 위주로 설정하거나, None으로 두어 모든 국가를 가져온 후 필터링)
        default_countries = ['US', 'EU', 'CN', 'JP', 'GB', 'KR', 'DE', 'CA', 'AU']
        target_countries = countries if countries is not None else default_countries

        logger.info(f"경제 캘린더 수집 시작: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"대상 국가: {target_countries}, 최소 중요도: {importance_min}")

        all_events = self._collect_with_selenium(start_date, end_date)

        if not all_events:
            logger.warning("수집된 경제 지표 데이터가 없습니다.")
            return pd.DataFrame()

        df = pd.DataFrame(all_events)
        
        # 데이터 타입 변환 및 정렬 (오류 방지)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df.dropna(subset=['datetime'], inplace=True) # datetime 변환 실패한 행 제거
            df = df.sort_values('datetime').reset_index(drop=True)
        else:
            logger.warning("'datetime' 컬럼이 없어 정렬을 수행할 수 없습니다.")
            return pd.DataFrame()

        # 필터링
        if not df.empty:
            original_count = len(df)
            if target_countries: # countries 리스트가 제공된 경우에만 필터링
                df = df[df['country_code'].isin(target_countries)]
            df = df[df['importance'] >= importance_min]
            logger.info(f"필터링 후 {len(df)}개 이벤트 (원본: {original_count}개)")
        
        logger.info(f"최종적으로 경제 지표 {len(df)}개 수집 완료")
        return df

    def _collect_with_selenium(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Selenium을 사용한 데이터 수집 (날짜별 순회)"""
        events = []
        driver = None
        max_retries_per_date = 2 # 날짜별 최대 재시도 횟수

        try:
            driver = self._setup_driver()
            current_date = start_date
            while current_date <= end_date:
                date_str_url = current_date.strftime('%Y-%m-%d') # URL용 날짜 포맷
                # Investing.com은 URL에 날짜를 yyyy-mm-dd 형식으로 사용합니다.
                # 예: https://kr.investing.com/economic-calendar/?dateFrom=2024-01-01&dateTo=2024-01-01
                # 정확한 URL 파라미터는 사이트마다 다를 수 있으므로, `ec.php`와 같은 스크립트 대신
                # 일반적으로 사용하는 캘린더 메인 URL에 date 파라미터를 붙이는 방식 시도
                
                # 실제 Investing.com 한국 경제 캘린더는 날짜를 URL 경로에 포함하는 것으로 보임.
                # 예: https://kr.investing.com/economic-calendar/
                # 그리고 JavaScript로 날짜를 선택하면 내부적으로 데이터를 로드하거나 페이지를 갱신.
                # 여기서는 Selenium으로 매일 URL을 새로 접속하여 해당 날짜 데이터를 가져오는 것으로 가정.
                # Investing.com은 필터링 시 POST 요청을 사용할 수 있으나,
                # XHR이 없다고 하셨으므로, 각 날짜 페이지를 직접 로드하는 방식으로 진행.
                # URL 파라미터 방식이 작동하지 않으면, Selenium으로 날짜 필터 UI를 직접 조작해야 함.
                # 현재는 URL 파라미터를 사용한 방식을 유지 (만약 작동한다면 이 방식이 더 간단)
                # url = f"{self.base_url}?dateFrom={date_str_url}&dateTo={date_str_url}"
                # Investing.com의 경우, dateFrom, dateTo 파라미터가 아니라, 페이지 내에서 날짜를 선택하고 JS로 갱신.
                # Selenium으로는 이 UI 조작을 흉내내야 함.
                # 하지만 여기서는 "각 날짜별로 URL을 새로 접속"하는 시나리오를 가정. Investing.com은 이 방식이 아닐 수 있음.
                # 현재로서는 XHR이 없으므로, Selenium이 로드한 기본 페이지(오늘 날짜)에서 데이터를 읽고,
                # 다른 날짜는 Selenium으로 날짜 변경 UI를 클릭해야 합니다.
                # 지금은 simplicity를 위해 해당 날짜의 페이지를 직접 로드한다고 가정하고,
                # 만약 이것이 작동하지 않으면, 메인 페이지 로드 후 날짜 UI를 조작하는 코드로 변경해야 합니다.
                # 아래는 특정 날짜로 직접 가는 URL이 없다고 가정하고, 매번 메인 페이지를 연 뒤 JS로 필터링하는 방식의 예시 (복잡)
                # 여기서는 원본 코드의 "날짜별 URL 순회" 아이디어를 살려서
                # URL 파라미터가 아니라 Investing.com이 사용하는 다른 방식 (JS 필터링 후 파싱)으로 가정하고
                # 일단은 매번 기본 URL을 로드 후 해당 날짜 데이터를 파싱하는 코드로 유지합니다. (실제로는 더 복잡한 UI 조작 필요)

                date_str_log = current_date.strftime('%Y-%m-%d') # 로그용
                logger.info(f"날짜 수집 시도: {date_str_log}")
                
                # Investing.com은 필터를 적용할 때 페이지 URL이 직접 변경되지 않고 AJAX로 데이터를 가져올 가능성이 높지만
                # XHR이 없다고 하셨으니, Selenium이 현재 보고 있는 페이지에서 데이터를 긁어오는 것으로 가정합니다.
                # 이 경우, Selenium으로 날짜 필터를 변경하는 UI 조작이 필요합니다.
                # 해당 UI 조작 로직은 복잡하므로, 여기서는 일단 현재 날짜 데이터만 가져온다고 단순화하거나,
                # 또는 URL이 date 파라미터를 받는다고 가정합니다 (이전 코드처럼).
                # XHR이 없다면, Selenium으로 "날짜 변경 UI"를 클릭하고, 새 데이터가 로드될 때까지 기다려야 합니다.

                # 현재는 각 날짜별 URL 구성이 아닌, "필터" UI를 Selenium으로 조작해야 정확한 데이터를 얻을 수 있습니다.
                # 하지만 XHR이 없다는 전제 하에, 우선은 주어진 날짜에 대한 페이지를 Selenium으로 열고
                # 그 페이지의 테이블을 파싱한다고 가정하고 코드를 작성합니다.
                # Investing.com의 경우, UI를 통해 날짜를 설정하고 필터링해야 합니다.
                # 이 코드는 해당 날짜에 대한 필터링 UI 조작이 이미 완료되었다고 가정하고 테이블을 읽습니다.
                # 실제로는 driver.get(self.base_url) 후 날짜 변경 UI 조작 코드가 들어가야 합니다.
                # 여기서는 원본 코드의 "날짜별로 URL을 만들어 접속" 하는 아이디어를 최대한 살리되, Investing.com 특성을 고려합니다.
                
                # === Investing.com 날짜 필터링 UI 조작 (예시) ===
                # 1. 기본 경제 캘린더 페이지로 이동
                if current_date == start_date: # 첫 날짜에만 기본 페이지 로드
                    driver.get(self.base_url)
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.ID, "economicCalendarData"))
                    )
                    logger.debug(f"기본 페이지 로드 완료: {self.base_url}")
                
                # 2. 날짜 필터 버튼 클릭 (달력 아이콘)
                try:
                    date_picker_toggle = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.ID, "datePickerToggleBtn"))
                    )
                    date_picker_toggle.click()
                    logger.debug("날짜 선택기 토글 버튼 클릭됨")
                    
                    # 3. 달력에서 시작 날짜와 종료 날짜 선택 (이 부분은 실제 달력 UI 구조에 따라 매우 달라짐)
                    # 여기서는 간단히 input 필드에 직접 날짜를 입력하는 방식을 시도 (만약 있다면)
                    # Investing.com은 실제로는 달력을 클릭해야 함.
                    date_from_input_id = "dateFrom" # 실제 ID 확인 필요 (HTML에는 picker가 있음)
                    date_to_input_id = "dateTo"     # 실제 ID 확인 필요
                    
                    date_str_for_input = current_date.strftime('%Y/%m/%d') # 사이트가 요구하는 형식
                    
                    # input 필드에 날짜 설정 (예시 - 실제로는 달력 클릭 필요)
                    # WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "actualDateFrom"))).clear()
                    # driver.find_element(By.ID, "actualDateFrom").send_keys(date_str_for_input)
                    # WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "actualDateTo"))).clear()
                    # driver.find_element(By.ID, "actualDateTo").send_keys(date_str_for_input)
                    
                    # 대신, Investing.com에서 사용하는 calendarFilters.datePickerFilter JS 함수 호출 시도
                    # 주의: 이 방식은 사이트 내부 JS 함수에 의존하므로 변경에 매우 취약
                    date_js_str = current_date.strftime('%Y-%m-%d')
                    js_script = f"calendarFilters.datePickerFilter('{date_js_str}', '{date_js_str}');"
                    driver.execute_script(js_script)
                    logger.info(f"JavaScript로 날짜 필터 적용 시도: {date_js_str}")

                    # 필터 적용 후 데이터 로딩 대기 (예: 테이블 내용 변경 또는 로딩 스피너 사라짐)
                    # Investing.com은 필터 적용 시 economicCalendarLoading ID를 가진 로딩 스피너를 사용함.
                    WebDriverWait(driver, 10).until(
                        EC.invisibility_of_element_located((By.ID, "economicCalendarLoading"))
                    )
                    logger.debug("날짜 필터 적용 후 로딩 완료")
                    
                except TimeoutException:
                    logger.warning(f"{date_str_log} 날짜 필터 UI 조작 중 타임아웃 발생. 현재 페이지 데이터로 진행.")
                except Exception as e_filter:
                    logger.error(f"{date_str_log} 날짜 필터 UI 조작 중 오류: {e_filter}", exc_info=True)
                # === UI 조작 끝 ===
                
                retry_count = 0
                success_on_date = False
                while retry_count < max_retries_per_date and not success_on_date:
                    try:
                        # 테이블 데이터가 실제로 로드될 때까지 대기
                        # economicCalendarData 테이블 내의 첫 번째 tr.js-event-item을 기다림
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "#economicCalendarData tr.js-event-item"))
                        )
                        logger.debug(f"{date_str_log}: 이벤트 테이블 데이터 확인됨.")

                        table = driver.find_element(By.ID, "economicCalendarData")
                        rows = table.find_elements(By.CSS_SELECTOR, "tr.js-event-item") # 이벤트 행만 선택

                        if not rows:
                            logger.info(f"{date_str_log}: js-event-item 클래스를 가진 이벤트 행이 없습니다.")
                            # "theDay" 클래스를 가진 행(날짜 표시 행)만 있고 이벤트가 없는 경우도 고려
                            day_header_row = table.find_elements(By.CSS_SELECTOR, "tr > td.theDay")
                            if day_header_row:
                                logger.info(f"{date_str_log}: 이벤트는 없으나 날짜 헤더는 존재합니다. 다음 날짜로 진행.")
                                success_on_date = True # 이벤트가 없는 것도 성공으로 간주
                                break 
                            else: # 날짜 헤더도 없으면 페이지가 이상한 것
                                logger.warning(f"{date_str_log}: 이벤트 행도, 날짜 헤더도 없습니다. 페이지 로드 실패 가능성.")
                                raise NoSuchElementException("테이블 데이터 없음")


                        for row_element in rows:
                            event_data = self._parse_row(row_element, current_date)
                            if event_data:
                                events.append(event_data)
                        
                        success_on_date = True
                        logger.info(f"{date_str_log}: {len(rows)}개 행 파싱 시도, {len(events)}개 유효 이벤트 누적.")

                    except TimeoutException:
                        retry_count += 1
                        logger.warning(f"{date_str_log}: 데이터 로딩 타임아웃 (시도 {retry_count}/{max_retries_per_date}).")
                        if retry_count >= max_retries_per_date:
                            logger.error(f"{date_str_log}: 최종 타임아웃. 이 날짜 데이터 수집 실패.")
                        else:
                            time.sleep(3) # 재시도 전 잠시 대기
                            driver.refresh() # 페이지 새로고침 후 재시도
                            logger.info(f"{date_str_log}: 페이지 새로고침 후 재시도...")
                    except NoSuchElementException as e_nse:
                        retry_count += 1
                        logger.warning(f"{date_str_log}: 필수 요소 못찾음 (시도 {retry_count}/{max_retries_per_date}): {e_nse}")
                        if retry_count >= max_retries_per_date:
                            logger.error(f"{date_str_log}: 최종적으로 요소 못찾음. 이 날짜 데이터 수집 실패.")
                        else:
                            time.sleep(3)
                            driver.refresh()
                            logger.info(f"{date_str_log}: 페이지 새로고침 후 재시도...")
                    except Exception as e_date:
                        retry_count += 1 # 일반 오류도 재시도 카운트
                        logger.error(f"{date_str_log} 날짜 처리 중 예외 발생 (시도 {retry_count}/{max_retries_per_date}): {e_date}", exc_info=True)
                        if retry_count >= max_retries_per_date:
                             logger.error(f"{date_str_log}: 최종 예외 발생. 이 날짜 데이터 수집 실패.")
                        else:
                            time.sleep(3)
                            # driver.refresh() # 심각한 오류일 수 있으므로 새로고침은 선택적

                current_date += timedelta(days=1)
                if current_date <= end_date : # 마지막 날짜가 아니라면 다음 날짜 수집 전 약간의 대기
                    time.sleep(1) # 과도한 요청 방지를 위한 최소한의 간격

        except WebDriverException as e_wd:
            logger.critical(f"WebDriver 오류 발생: {e_wd}", exc_info=True)
        except Exception as e_main:
            logger.error(f"Selenium 전체 수집 과정 중 오류 발생: {e_main}", exc_info=True)
        finally:
            if driver:
                driver.quit()
                logger.info("WebDriver 종료됨.")
        return events

    def _parse_row(self, row_element, event_date_obj: datetime) -> Optional[Dict]:
        """테이블의 단일 행(tr)에서 이벤트 데이터 파싱"""
        try:
            cells = row_element.find_elements(By.TAG_NAME, "td")
            if len(cells) < 7: # 최소 7개 셀 (시간, 국기, 중요도, 이벤트, 실제, 예상, 이전) + 알림용 마지막 셀은 선택적
                logger.debug(f"행 분석 중단: 셀 개수 부족 ({len(cells)}개)")
                return None

            # 시간 추출 (첫 번째 셀)
            time_str = cells[0].text.strip()
            if not time_str or ':' not in time_str: # '하루 종일' 또는 빈 시간 제외
                logger.debug(f"유효하지 않은 시간 정보: '{time_str}'")
                return None
            
            # 국가 코드 추출 (두 번째 셀)
            country_span = cells[1].find_element(By.TAG_NAME, "span") # 국가명은 span title에 있음
            country_name_kr = country_span.get_attribute("title").strip()
            country_code = self.country_codes.get(country_name_kr, 'OTHER')
            if country_code == 'OTHER':
                 logger.warning(f"알 수 없는 국가명: '{country_name_kr}', 코드를 OTHER로 설정.")


            # 중요도 추출 (세 번째 셀)
            # 옵션 1: data-img_key 속성 사용 (HTML 구조에 따라 이 방법이 더 안정적일 수 있음)
            # importance_attr = cells[2].get_attribute("data-img_key") # 예: "bull1", "bull2", "bull3"
            # importance = 0
            # if importance_attr:
            #     if "bull3" in importance_attr: importance = 3
            #     elif "bull2" in importance_attr: importance = 2
            #     elif "bull1" in importance_attr: importance = 1
            # else:
            #     logger.debug(f"중요도 속성(data-img_key)을 찾을 수 없음. 기본값 0 사용.")
            # 옵션 2: 황소 아이콘 개수 세기 (HTML 구조에 따라 이 방법이 더 안정적일 수 있음)
            importance_icons = cells[2].find_elements(By.CSS_SELECTOR, f"i.{self.importance_icon_class}")
            importance = len(importance_icons)
            if importance == 0 and "sentiment" in cells[2].get_attribute("class"): # 휴일 등 텍스트로 된 경우
                if "높은 변동성이 예상됨" in cells[2].get_attribute("title"): importance = 3
                elif "보통 정도의 변동성이 예상됨" in cells[2].get_attribute("title"): importance = 2
                elif "낮은 변동성이 예상됨" in cells[2].get_attribute("title"): importance = 1
                else: importance = 0 # 중요도 알 수 없음
            
            # 이벤트명 추출 (네 번째 셀)
            event_name_tag = cells[3].find_element(By.TAG_NAME, "a") # 이벤트명은 a 태그 안에 있음
            event_name = event_name_tag.text.strip()
            event_link = event_name_tag.get_attribute('href')

            # 실제, 예상, 이전 값 추출 (다섯 번째, 여섯 번째, 일곱 번째 셀)
            actual_raw = cells[4].text.strip()
            forecast_raw = cells[5].text.strip()
            previous_raw = cells[6].text.strip()

            actual = self._parse_value(actual_raw)
            forecast = self._parse_value(forecast_raw)
            previous = self._parse_value(previous_raw)

            # 날짜와 시간 조합
            try:
                hour, minute = map(int, time_str.split(':'))
                event_datetime = event_date_obj.replace(hour=hour, minute=minute, second=0, microsecond=0)
            except ValueError:
                logger.warning(f"시간 문자열 파싱 오류: '{time_str}'. 이 이벤트 건너뜀.")
                return None

            parsed_event = {
                'datetime': event_datetime,
                'country_code': country_code,
                'country_name_kr': country_name_kr, # 디버깅 또는 매핑 확장을 위해 추가
                'importance': importance,
                'event': event_name,
                'actual': actual,
                'forecast': forecast,
                'previous': previous,
                'actual_raw': actual_raw,         # 원본 문자열도 저장
                'forecast_raw': forecast_raw,
                'previous_raw': previous_raw,
                'event_link': event_link          # 이벤트 상세 링크 추가
            }
            # logger.debug(f"성공적으로 파싱된 이벤트: {parsed_event}")
            return parsed_event

        except NoSuchElementException as e_nse:
            logger.debug(f"행 파싱 중 요소 못찾음: {e_nse} - 해당 행 건너뜀.")
            # logger.debug(f"문제의 행 HTML: {row_element.get_attribute('outerHTML')}") # 디버깅용
            return None
        except Exception as e:
            logger.error(f"행 파싱 중 일반 오류 발생: {e} - 해당 행 건너뜀.", exc_info=True)
            # logger.error(f"문제의 행 HTML: {row_element.get_attribute('outerHTML')}") # 디버깅용
            return None

    def _parse_value(self, value_str: str) -> Optional[float]:
        """값 문자열(%, K, M, B 등 포함)을 float으로 파싱"""
        if not value_str or value_str == '' or value_str == '\xa0': # Non-breaking space
            return None
        
        text = value_str.strip().replace(',', '').replace('$', '').replace('€', '').replace('¥', '')
        
        try:
            if text.endswith('%'):
                return float(text.replace('%', '')) / 100.0 # 백분율은 0-1 사이 값으로 변환 (선택적)
                                                            # 만약 5.0%를 5.0으로 저장하고 싶으면 / 100.0 제거
            
            multiplier = 1
            if text.endswith('K'):
                multiplier = 1e3
                text = text[:-1]
            elif text.endswith('M'):
                multiplier = 1e6
                text = text[:-1]
            elif text.endswith('B'):
                multiplier = 1e9
                text = text[:-1]
            elif text.endswith('T'): # 조 단위 (Trillion)
                multiplier = 1e12
                text = text[:-1]

            return float(text) * multiplier
        except ValueError:
            logger.debug(f"값 파싱 오류: '{value_str}' -> float 변환 실패")
            return None # 변환 실패 시 None 반환

    # --- 아래는 보조적인 메서드 또는 향후 확장용으로 유지 ---
    def get_upcoming_high_impact_events(self, hours_ahead: int = 24, 
                                        target_countries: Optional[List[str]] = None,
                                        importance_min: int = 3) -> pd.DataFrame:
        """향후 N시간 내 고영향 이벤트 조회"""
        now = datetime.now()
        start_date = now
        end_date = now + timedelta(hours=hours_ahead)
        
        logger.info(f"{hours_ahead}시간 내 고영향(중요도 {importance_min} 이상) 이벤트 조회 시작.")
        
        df = self.fetch_calendar_data(
            start_date=start_date,
            end_date=end_date,
            countries=target_countries, # 여기서 countries는 표준 코드로 전달
            importance_min=importance_min
        )
        # fetch_calendar_data에서 이미 시간 기준으로 필터링 및 정렬이 되었을 것임.
        # 추가적으로 현재 시간 이후의 이벤트만 필터링
        if not df.empty and 'datetime' in df.columns:
            df = df[df['datetime'] > now].reset_index(drop=True)
            logger.info(f"조회된 향후 고영향 이벤트 수: {len(df)}")
        return df

    def get_events_by_indicator(self, indicator_names: List[str],
                                days_back: int = 30,
                                target_countries: Optional[List[str]] = None) -> pd.DataFrame:
        """특정 지표명의 과거 데이터 조회 (부분 문자열 매칭)"""
        now = datetime.now()
        start_date = now - timedelta(days=days_back)
        end_date = now
        
        logger.info(f"과거 {days_back}일간 '{', '.join(indicator_names)}' 지표 조회 시작.")
        
        df = self.fetch_calendar_data(
            start_date=start_date,
            end_date=end_date,
            countries=target_countries, # 여기서 countries는 표준 코드로 전달
            importance_min=1 # 모든 중요도 포함
        )
        
        if df.empty:
            logger.warning("지표별 과거 데이터 조회 결과 없음.")
            return df
            
        # 지표명 필터링 (대소문자 구분 없이 부분 매칭)
        # 정규 표현식의 각 이름을 이스케이프하여 특수문자가 문제 일으키지 않도록 함
        safe_indicator_names = [re.escape(name) for name in indicator_names]
        pattern = '|'.join(safe_indicator_names)
        mask = df['event'].str.contains(pattern, case=False, na=False, regex=True)
        
        filtered_df = df[mask].reset_index(drop=True)
        logger.info(f"조회된 지표 관련 과거 이벤트 수: {len(filtered_df)}")
        return filtered_df

    def analyze_forecast_accuracy(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """예측 정확도 기본 분석 (예시)"""
        if df.empty or 'actual' not in df.columns or 'forecast' not in df.columns or 'previous' not in df.columns or 'event' not in df.columns:
            logger.warning("예측 정확도 분석을 위한 데이터 부족 또는 컬럼 누락.")
            return {}

        valid_df = df.dropna(subset=['actual', 'forecast', 'previous'])
        
        if valid_df.empty:
            logger.info("실제, 예측, 이전 값이 모두 있는 데이터가 없어 정확도 분석 불가.")
            return {}
            
        accuracy_summary = {}
        
        for event_name, group in valid_df.groupby('event'):
            if len(group) < 3:  # 통계적 의미를 위해 최소 샘플 수 설정
                continue
            
            # 방향성 정확도: (실제 - 이전)과 (예상 - 이전)의 부호가 같은 경우
            actual_change_direction = (group['actual'] - group['previous']) > 0
            forecast_change_direction = (group['forecast'] - group['previous']) > 0
            direction_correct_count = (actual_change_direction == forecast_change_direction).sum()
            direction_accuracy = direction_correct_count / len(group) if len(group) > 0 else 0.0
            
            # 평균 절대 오차율 (MAPE와 유사하나, 0으로 나누는 문제 회피 위해 변형)
            # 실제 값이 0인 경우를 대비하여 분모에 작은 값 추가 또는 다른 지표 사용 고려
            abs_error = abs(group['actual'] - group['forecast'])
            # 0으로 나누는 것을 방지하기 위해, 실제 값이 0에 가까우면 오차율 계산에서 제외하거나,
            # 분모에 매우 작은 값을 더할 수 있습니다. 여기서는 간단히 평균 절대 오차를 사용.
            mean_abs_error = abs_error.mean()
            
            # 실제 값이 0이 아닌 경우에 대한 평균 절대 백분율 오차 (MAPE)
            mape_series = (abs(group['actual'] - group['forecast']) / abs(group['actual'])) * 100
            mape = mape_series[group['actual'] != 0].mean() # 실제 값이 0이 아닌 경우만 계산
            if pd.isna(mape): mape = None # 계산 불가 시 None

            accuracy_summary[event_name] = {
                'direction_accuracy': round(direction_accuracy, 4),
                'mean_absolute_error': round(mean_abs_error, 4) if pd.notna(mean_abs_error) else None,
                'mean_absolute_percentage_error': round(mape, 2) if mape is not None else None,
                'sample_size': len(group)
            }
        
        logger.info(f"{len(accuracy_summary)}개 지표에 대한 예측 정확도 분석 완료.")
        return accuracy_summary

if __name__ == '__main__':
    # 로거 기본 설정 (테스트용)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    collector = EconomicCalendarCollector()

    # 오늘 날짜의 한국, 미국, 유로존, 중국, 일본의 중요도 2 이상 이벤트 수집
    today = datetime.now()
    # df_today_high_importance = collector.fetch_calendar_data(
    #     start_date=today,
    #     end_date=today,
    #     countries=['KR', 'US', 'EU', 'CN', 'JP'],
    #     importance_min=2
    # )
    # print("\n=== 오늘 주요 국가, 중요도 2 이상 이벤트 ===")
    # if not df_today_high_importance.empty:
    #     print(df_today_high_importance.to_string())
    # else:
    #     print("데이터 없음")

    # 지난 7일간 미국의 모든 중요도 이벤트 수집
    # start_past = today - timedelta(days=7)
    # df_us_past_week = collector.fetch_calendar_data(
    #     start_date=start_past,
    #     end_date=today, # 오늘까지
    #     countries=['US'],
    #     importance_min=1
    # )
    # print(f"\n=== 지난 7일간 미국 모든 이벤트 (오늘: {today.strftime('%Y-%m-%d')}) ===")
    # if not df_us_past_week.empty:
    #     print(df_us_past_week.to_string())
    # else:
    #     print("데이터 없음")


    # 특정 지표 (예: '실업률', 'GDP', 'CPI') 과거 30일 데이터 조회 (미국, 한국)
    # indicators_to_search = ['실업률', '국내총생산', '소비자물가지수'] # HTML title 기준 검색어
    # df_specific_indicators = collector.get_events_by_indicator(
    #     indicator_names=indicators_to_search,
    #     days_back=90, # 예시로 90일
    #     target_countries=['US', 'KR']
    # )
    # print(f"\n=== 지난 90일간 '{', '.join(indicators_to_search)}' 관련 이벤트 (미국, 한국) ===")
    # if not df_specific_indicators.empty:
    #     print(df_specific_indicators.to_string())
    #     # 예측 정확도 분석
    #     accuracy = collector.analyze_forecast_accuracy(df_specific_indicators)
    #     print("\n--- 예측 정확도 분석 (예시) ---")
    #     for event, acc_data in accuracy.items():
    #         print(f"지표: {event}, 방향 정확도: {acc_data['direction_accuracy']:.2%}, 평균절대오차율: {acc_data['mean_absolute_percentage_error'] if acc_data['mean_absolute_percentage_error'] else 'N/A'} (샘플 수: {acc_data['sample_size']})")
    # else:
    #     print("데이터 없음")

    # 향후 24시간 내 고영향 이벤트 (중요도 3)
    df_upcoming = collector.get_upcoming_high_impact_events(hours_ahead=24*7, importance_min=3, target_countries=['US','KR','JP','EU','CN','GB'])
    print(f"\n=== 향후 7일간 주요국 고영향(3) 이벤트 ===")
    if not df_upcoming.empty:
        print(df_upcoming.to_string())
    else:
        print("데이터 없음")        