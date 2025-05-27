# 통합 주식 분석 및 트레이딩 시스템 (Integrated Stock Analysis and Trading System)

# 통합 주식 분석 및 트레이딩 시스템 (Integrated Stock Analysis and Trading System)

## 🌟 프로젝트 목표 (Project Goal)
본 프로젝트는 실시간 주식 데이터 수집, AI 기반 가격 예측, 뉴스 감성 분석, 기술적 지표 및 경제 지표 분석 등을 통합적으로 제공하여 사용자가 정보에 입각한 투자 결정을 내리고 자동화된 트레이딩 전략을 실행할 수 있도록 지원하는 시스템을 구축하는 것을 목표로 합니다.
특히, Investing.com과 같은 외부 소스에서 경제 지표 캘린더 데이터를 수집하여 분석에 활용하고, 이를 다른 분석 결과(ML, 뉴스 감성, 기술적 지표)와 종합하여 기간별 시장 상승/하락 예측을 제공하는 것을 주요 기능으로 포함합니다.

## 🚀 빠른 시작 (Quick Start)

1.  **환경 설정 (Environment Setup)**
    ```bash
    pip install -r requirements.txt
    # requirements.txt에 webdriver-manager, trafilatura, psycopg2-binary (PostgreSQL 사용 시) 등이 포함되어 있는지 확인하세요.
    ```

2.  **설정 파일 수정 (Modify Configuration File)**
    `user_config/config.json` 파일에서 API 키 (Alpha Vantage, NewsAPI 등) 및 데이터베이스 설정을 확인하고 필요시 수정하세요.
    기본 설정 파일은 `config/settings.py` 에 정의되어 있으며, 최초 실행 시 `user_config` 디렉토리에 `config.json`이 생성될 수 있습니다.

3.  **실행 (Execution)**
    ```bash
    python main.py
    ```

4.  **📁 프로젝트 구조 (Project Structure)**
    ```
    core/                        핵심 엔진 (컨트롤러, 이벤트 매니저, 워크플로우 엔진)
    data/                        데이터 수집 및 저장
      ├─ database_manager.py: SQLAlchemy 기반 DB 연동 관리
      ├─ models.py: SQLAlchemy 데이터 모델 정의
      ├─collectors/                 외부 데이터 수집기
         ├─ economic_calendar_collector.py  : Investing.com 경제 지표 캘린더 수집 (Selenium 기반)
         ├─ news_collector.py               : 뉴스 데이터 수집 (NewsAPI, trafilatura)
         └─ stock_data_collector.py         : 주식 데이터 수집 (yfinance)
    analysis/                    AI/ML 모델 및 분석
      ├─ sentiment/               뉴스 감성 분석 모듈 (news_sentiment.py)
      ├─ technical/
      │  └─ indicators.py       기술적 지표 계산 함수 (SMA, EMA, RSI, MACD, 볼린저 밴드 등)
      ├─ models/                  (Phase 2) ML/DL 모델 구현부 (예: ml_models.py, dl_models.py)
      ├─ feature_engineering.py   (Phase 2) 피처 엔지니어링
      └─ model_manager.py         (Phase 2) 모델 학습/관리/예측 담당
    trading/                     거래 전략 및 실행
      ├─ execution/               주문 실행 모듈 (order_manager.py)
      ├─ portfolio/               포트폴리오 관리 모듈 (portfolio_manager.py)
      └─ strategy/                트레이딩 전략 (base_strategy.py, ml_strategy.py)
    ui/                          사용자 인터페이스 (PyQt5)
      ├─ dialogs/                 설정 대화상자 (settings_dialog.py)
      ├─ widgets/                 차트/데이터 표시 (chart_widget.py, data_widget.py, trading_widget.py)
      └─ styles/                  QSS 스타일 시트 (dark_style.qss)
    utils/                       유틸리티 함수 (logger.py, scheduler.py)
    config/                      애플리케이션 설정 관리 (settings.py)
    user_config/                 사용자 설정 파일 저장 (config.json)
    logs/                        로그 저장
    main.py                      애플리케이션 진입점
    requirements.txt             의존성 목록
    ```

---

## ✨ 현재까지 수행한 작업 (Completed Tasks)
- **코어 시스템 구축**: MainController, EventManager, WorkflowEngine, Scheduler 등 핵심 로직 구현 완료.
- **데이터 관리 및 기본 분석 모듈 (Phase 1 완료)**:
  - 데이터 수집 기능: 주식 데이터(yfinance), 뉴스 데이터(NewsAPI, trafilatura), Investing.com 경제 지표 캘린더 수집 기능 구현 및 UI 연동 완료.
  - SQLAlchemy 기반 DB 모델 정의 및 `DatabaseManager`를 통한 DB CRUD 기능 구현 완료.
  - 데이터 수집 태스크에 DB 연동 및 캐시 + DB 조회 로직 구현 완료.
- **기본 분석 기능**:
  - `NewsSentimentAnalyzer` (NLTK VADER, FinBERT) 구현 완료.
  - 기술적 지표(SMA, EMA, RSI, MACD, 볼린저 밴드 등) 계산 기능 완료 및 UI 연동.
- **트레이딩 시스템 기초**: `OrderManager`, `PortfolioManager`, 기본 `MLTradingStrategy` 구조 정의.
- **사용자 인터페이스 (PyQt5)**: 주요 UI 컴포넌트 구성 및 데이터 연동, 스타일 개선.
- **설정 및 유틸리티**: `Settings` 클래스, 로깅 시스템 구축.

---

## 📝 향후 할 일 (To-Do List)

### Phase 2: 핵심 분석 엔진 및 ML 모델 구축 (진행 중)
-   **뉴스 감성 + 경제 지표 통합 영향 분석 모듈 개발**:
    -   수집된 뉴스 감성 데이터와 경제 지표 데이터를 종합적으로 분석.
    -   시장 또는 개별 종목에 대한 단기적 영향 평가 로직 구현.
    -   결과를 UI의 "AI 분석" 탭 또는 별도 탭에 시각화.
-   **ML/DL 예측 모델 구현 및 학습 파이프라인 설계**:
    -   `analysis/models/` (또는 `models/`) 디렉토리에 실제 ML/DL 모델 (LSTM, RandomForest 등) 파일 생성.
    -   기술적 지표, 뉴스 감성, 경제 지표 등을 활용한 피처 엔지니어링 모듈 개발.
    -   데이터 전처리, 모델 학습, 평가, 하이퍼파라미터 튜닝을 포함하는 학습 파이프라인 설계 및 `WorkflowEngine` 연동.
-   **`ModelManager` 개선**:
    -   `TempModelManager`를 실제 모델 학습/저장/로드/예측 기능을 갖춘 `ModelManager`로 대체.
    -   모델 버전 관리 및 성능 추적 기능 기초 설계.

### Phase 3: 통합 분석 및 UI 연동
-   종합 예측 점수 계산 및 UI 표시 ("AI 분석" 탭).
-   `TradingWidget` 등 UI 기능 통합 완성.

### Phase 4: 트레이딩, 백테스팅 및 고도화
-   실거래 기능 연동 및 리스크 관리.
-   백테스팅 엔진 설계 및 구현.
-   모델 성능 개선 및 지속적인 시스템 고도화.

---

## ⚠️ 주의사항 (Important Notes)
- 이 시스템은 **교육 및 연구 목적**으로 제작되었습니다.
- 실제 거래에 사용하기 전에 **충분한 테스트와 검증**이 필요합니다.
- 투자 결정에 따른 모든 책임은 사용자 본인에게 있습니다.
