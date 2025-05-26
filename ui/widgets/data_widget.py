# ui/widgets/data_widget.py

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pandas as pd # 데이터 처리를 위해 추가
import logging

logger = logging.getLogger(__name__)

class DataWidget(QWidget):
    """데이터 표시 위젯 (실시간 호가, 기술적 지표, 뉴스, AI 예측 등)"""

    def __init__(self, controller=None, parent=None): # controller 인자 추가 및 parent 인자 명시
        super().__init__(parent) # 부모 위젯 전달
        self.controller = controller
        self.symbol = "AAPL" # 기본 심볼
        self.setup_ui()
        logger.info("DataWidget 초기화 완료")

    def setup_ui(self):
        """UI 설정"""
        main_layout = QVBoxLayout(self) # self를 부모로 전달하여 레이아웃 설정
        main_layout.setContentsMargins(0, 0, 0, 0) # 여백 최소화

        # 탭 위젯
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 실시간 데이터 탭
        self.realtime_tab = self._create_realtime_tab()
        self.tab_widget.addTab(self.realtime_tab, "📊 실시간")

        # 기술적 지표 탭
        self.technical_tab = self._create_technical_tab()
        self.tab_widget.addTab(self.technical_tab, "📈 기술적 지표")

        # 뉴스 감성 탭
        self.sentiment_tab = self._create_sentiment_tab()
        self.tab_widget.addTab(self.sentiment_tab, "📰 뉴스") # 탭 이름 변경

        # 예측 탭
        self.prediction_tab = self._create_prediction_tab()
        self.tab_widget.addTab(self.prediction_tab, "🤖 AI 분석") # 탭 이름 변경

        # 경제 지표 탭 (신규 추가 제안)
        self.economic_indicators_tab = self._create_economic_indicators_tab()
        self.tab_widget.addTab(self.economic_indicators_tab, "🌍 경제 지표")


    def _create_realtime_tab(self):
        """실시간 데이터 탭 생성"""
        widget = QWidget()
        layout = QFormLayout(widget) # 위젯에 레이아웃 설정
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 실시간 데이터 라벨들
        self.price_label = QLabel("-")
        self.change_label = QLabel("-")
        self.volume_label = QLabel("-")
        self.high_label = QLabel("-")
        self.low_label = QLabel("-")
        self.prev_close_label = QLabel("-") # 전일 종가
        self.market_cap_label = QLabel("-") # 시가총액 (예시)

        # 스타일 적용
        self.price_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #E0E0E0;")
        self.change_label.setStyleSheet("font-size: 18px;") # 색상은 데이터에 따라 변경
        label_font = QFont()
        label_font.setPointSize(10)

        for lbl in [self.volume_label, self.high_label, self.low_label, self.prev_close_label, self.market_cap_label]:
            lbl.setFont(label_font)

        layout.addRow("현재가:", self.price_label)
        layout.addRow("등락:", self.change_label)
        layout.addRow("거래량:", self.volume_label)
        layout.addRow("당일 고가:", self.high_label)
        layout.addRow("당일 저가:", self.low_label)
        layout.addRow("전일 종가:", self.prev_close_label)
        layout.addRow("시가총액:", self.market_cap_label)

        # 데이터 업데이트 시간 표시 (예시)
        self.last_updated_label = QLabel("최종 업데이트: -")
        self.last_updated_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.last_updated_label.setStyleSheet("font-size: 9px; color: #AAAAAA;")
        layout.addRow(self.last_updated_label)

        return widget

    def _create_technical_tab(self):
        """기술적 지표 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5,5,5,5)

        self.technical_table = QTableWidget(0, 2) # 행은 동적으로 추가, 컬럼: 지표명, 값
        self.technical_table.setHorizontalHeaderLabels(["지표명", "현재 값"]) # 헤더 레이블 수정
        self.technical_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch) # 지표명 컬럼 확장
        self.technical_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents) # 값 컬럼 내용에 맞게
        self.technical_table.setEditTriggers(QAbstractItemView.NoEditTriggers) # 편집 불가
        self.technical_table.setAlternatingRowColors(True) # 행 색상 교차
        self.technical_table.setSortingEnabled(True) # 정렬 기능 추가

        # 초기에는 비어있음, 데이터 수신 시 채워짐
        layout.addWidget(self.technical_table)
        return widget

    def _create_sentiment_tab(self):
        """뉴스 감성 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5,5,5,5)

        # 종목별 종합 감성 정보 (예시)
        top_info_layout = QFormLayout()
        self.overall_sentiment_score_label = QLabel("-")
        self.overall_sentiment_trend_label = QLabel("-")
        self.news_count_label = QLabel("-")

        top_info_layout.addRow("최근 24시간 평균 감성 점수:", self.overall_sentiment_score_label)
        top_info_layout.addRow("감성 트렌드:", self.overall_sentiment_trend_label)
        top_info_layout.addRow("관련 뉴스 수 (24h):", self.news_count_label)
        layout.addLayout(top_info_layout)

        # 뉴스 목록
        self.news_list_widget = QListWidget()
        self.news_list_widget.setAlternatingRowColors(True)
        self.news_list_widget.itemDoubleClicked.connect(self._on_news_item_double_clicked) # 더블클릭 시 브라우저로 링크 열기

        layout.addWidget(QLabel("최근 주요 뉴스:"))
        layout.addWidget(self.news_list_widget)
        return widget

    def _create_prediction_tab(self):
        """AI 예측 및 종합 판단 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10,10,10,10)

        group_box = QGroupBox("AI 기반 종합 분석 결과")
        form_layout = QFormLayout()

        self.ml_direction_label = QLabel("-") # 예: 상승, 하락, 보합
        self.ml_confidence_label = QLabel("-") # 예: 75%
        self.target_price_label = QLabel("-") # 예: $155.50 (단기 목표가)
        self.overall_signal_label = QLabel("-") # 예: 매수 추천, 관망, 매도 고려
        self.analysis_summary_text = QTextEdit() # 여러 줄 분석 요약
        self.analysis_summary_text.setReadOnly(True)
        self.analysis_summary_text.setMinimumHeight(80)


        label_font = QFont()
        label_font.setPointSize(11)
        self.ml_direction_label.setFont(label_font)
        self.ml_confidence_label.setFont(label_font)
        self.target_price_label.setFont(label_font)

        signal_font = QFont()
        signal_font.setPointSize(14)
        signal_font.setBold(True)
        self.overall_signal_label.setFont(signal_font)
        self.overall_signal_label.setAlignment(Qt.AlignCenter)


        form_layout.addRow("예측 방향 (ML):", self.ml_direction_label)
        form_layout.addRow("신뢰도:", self.ml_confidence_label)
        form_layout.addRow("AI 목표가 (참고):", self.target_price_label)
        form_layout.addRow(QLabel("종합 판단:")) # 빈 라벨로 공간 확보
        form_layout.addRow(self.overall_signal_label) # 종합 판단은 크게 표시
        form_layout.addRow(QLabel("분석 요약:"))
        form_layout.addRow(self.analysis_summary_text)


        group_box.setLayout(form_layout)
        layout.addWidget(group_box)
        layout.addStretch()
        return widget

    def _create_economic_indicators_tab(self):
        """ 주요 경제 지표 표시 탭 """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5,5,5,5)

        # 필터링 옵션 (예시: 국가, 중요도 - 추후 구현)
        # filter_layout = QHBoxLayout()
        # self.eco_country_filter_combo = QComboBox()
        # # ... 국가 목록 채우기 ...
        # self.eco_importance_filter_combo = QComboBox()
        # self.eco_importance_filter_combo.addItems(["모든 중요도", "높음(★★★)", "중간(★★☆)", "낮음(★☆☆)"])
        # filter_layout.addWidget(QLabel("국가:"))
        # filter_layout.addWidget(self.eco_country_filter_combo)
        # filter_layout.addWidget(QLabel("중요도:"))
        # filter_layout.addWidget(self.eco_importance_filter_combo)
        # filter_layout.addStretch()
        # refresh_button = QPushButton("새로고침")
        # refresh_button.clicked.connect(self._request_economic_data_refresh) # 컨트롤러에 요청하는 메서드 필요
        # filter_layout.addWidget(refresh_button)
        # layout.addLayout(filter_layout)

        self.economic_indicators_table = QTableWidget(0, 7) # 컬럼 추가: 시간, 국가, 중요도, 이벤트, 실제, 예상, 이전
        self.economic_indicators_table.setHorizontalHeaderLabels([
            "시간", "국가", "중요도", "이벤트", "실제", "예상", "이전"
        ])

        header = self.economic_indicators_table.horizontalHeader()
        # 시간, 국가, 중요도, 실제, 예상, 이전 컬럼은 내용에 맞게 자동 조절
        for i in [0, 1, 2, 4, 5, 6]:
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        # 이벤트 컬럼은 남은 공간을 모두 차지하도록 설정 (Stretch)
        # header.setSectionResizeMode(3, QHeaderView.Stretch)
        # 사용자가 직접 컬럼 너비를 조절할 수 있도록 Interactive 모드도 고려할 수 있습니다.
        header.setSectionResizeMode(QHeaderView.Interactive) # 전체 컬럼에 적용

        self.economic_indicators_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.economic_indicators_table.setAlternatingRowColors(True) # 행 번갈아가며 색상
        self.economic_indicators_table.setSortingEnabled(True)
        self.economic_indicators_table.setWordWrap(True) # 셀 내용 자동 줄 바꿈 활성화

        layout.addWidget(self.economic_indicators_table)
        return widget
    def update_economic_indicators_display(self, economic_data_df: pd.DataFrame): # 메서드명 변경 및 인자 타입 명시
        """ 경제 지표 데이터를 받아 테이블에 표시 """
        if economic_data_df is None:
            logger.warning("경제 지표 데이터가 None입니다. 테이블을 비웁니다.")
            self.economic_indicators_table.setRowCount(0)
            return

        self.economic_indicators_table.setSortingEnabled(False) # 업데이트 중 정렬 비활성화
        self.economic_indicators_table.setRowCount(0) # 기존 내용 삭제

        if economic_data_df.empty:
            logger.info("표시할 경제 지표 데이터가 없습니다.")
            self.economic_indicators_table.setSortingEnabled(True)
            return
        
        # 기본 글씨 색상 (다크 테마에 어울리도록)
        default_text_color = QColor("#E0E0E0") # 밝은 회색 (스타일시트에서 설정하는 것이 더 좋음)

        for index, row in economic_data_df.iterrows():
            row_count = self.economic_indicators_table.rowCount()
            self.economic_indicators_table.insertRow(row_count)

            dt_str = row['datetime'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['datetime']) else "-"
            dt_item = QTableWidgetItem(dt_str)
            
            country_str = str(row.get('country_code', '-'))
            country_item = QTableWidgetItem(country_str)
            
            importance_val = row.get('importance', 0)
            # 중요도에 따라 색상이나 텍스트를 다르게 표시
            if importance_val == 3:
                importance_str = "★★★"
                importance_color = QColor("red")
            elif importance_val == 2:
                importance_str = "★★☆"
                importance_color = QColor("orange")
            elif importance_val == 1:
                importance_str = "★☆☆"
                importance_color = QColor("yellow")
            else:
                importance_str = "-"
                importance_color = default_text_color # 중요도 없는 경우 기본색
            
            importance_item = QTableWidgetItem(importance_str)
            importance_item.setTextAlignment(Qt.AlignCenter)
            importance_item.setForeground(importance_color) # 중요도 색상 적용

            event_str = str(row.get('event', '-'))
            event_item = QTableWidgetItem(event_str)
            # 이벤트 셀은 자동으로 줄바꿈 되므로, setWordWrap은 테이블 전체에 적용
            
            actual_str = str(row.get('actual_raw', '-'))
            actual_item = QTableWidgetItem(actual_str)
            
            forecast_str = str(row.get('forecast_raw', '-'))
            forecast_item = QTableWidgetItem(forecast_str)
            
            previous_str = str(row.get('previous_raw', '-'))
            previous_item = QTableWidgetItem(previous_str)
            
            # 값에 따른 색상 강조 (실제 vs 예상)
            actual_val_numeric = row.get('actual') # 파싱된 숫자 값
            forecast_val_numeric = row.get('forecast')

            surprise_color = None
            if pd.notnull(actual_val_numeric) and pd.notnull(forecast_val_numeric):
                if actual_val_numeric > forecast_val_numeric:
                    surprise_color = QColor("lightgreen") # 예상보다 좋음 (긍정적 서프라이즈)
                elif actual_val_numeric < forecast_val_numeric:
                    surprise_color = QColor("#FF7F7F") # 예상보다 나쁨 (부정적 서프라이즈 - 연한 빨강)
            
            if surprise_color:
                actual_item.setBackground(surprise_color)
                # 글씨가 잘 보이도록 배경색에 따라 글씨색 조정 (선택적)
                # actual_item.setForeground(QColor("black"))


            # 모든 아이템에 기본 글씨 색상 적용 (스타일시트에서 하는 것이 더 좋음)
            for item in [dt_item, country_item, event_item, actual_item, forecast_item, previous_item]:
                item.setForeground(default_text_color)
            # 중요도 아이템은 위에서 개별 색상 적용됨

            self.economic_indicators_table.setItem(row_count, 0, dt_item)
            self.economic_indicators_table.setItem(row_count, 1, country_item)
            self.economic_indicators_table.setItem(row_count, 2, importance_item)
            self.economic_indicators_table.setItem(row_count, 3, event_item)
            self.economic_indicators_table.setItem(row_count, 4, actual_item)
            self.economic_indicators_table.setItem(row_count, 5, forecast_item)
            self.economic_indicators_table.setItem(row_count, 6, previous_item)
            
            event_link = row.get('event_link')
            if event_link:
                event_item.setToolTip(f"상세 정보 링크: {event_link}\n클릭하여 열기 (미구현)")
                # 더블클릭 시 링크 열기 기능은 QTableWidget의 itemDoubleClicked 시그널을 연결하여 구현 가능

        self.economic_indicators_table.setSortingEnabled(True)
        # 테이블의 행 높이를 내용에 맞게 조절 (줄바꿈된 텍스트가 다 보이도록)
        self.economic_indicators_table.resizeRowsToContents()
        logger.info(f"경제 지표 테이블 업데이트 완료: {len(economic_data_df)}개 항목.")
    # add_economic_indicator_row는 이제 update_economic_indicators_display로 대체됨
    # def add_economic_indicator_row(self, name: str, value: str, next_release: str):
    #     pass

    def set_symbol(self, symbol: str):
        """종목 변경 시 호출되어 UI 초기화 또는 데이터 요청"""
        if self.symbol != symbol:
            self.symbol = symbol
            logger.info(f"DataWidget: 심볼 변경됨 - {self.symbol}")
            # 각 탭의 내용 초기화
            self.clear_all_tabs_data()
            # 컨트롤러에 새 심볼에 대한 데이터 요청 (MainWindow에서 이미 수행)
            # if self.controller:
            #     self.controller.request_realtime_quote(self.symbol)
            #     self.controller.request_news_data(self.symbol) # 종목명 또는 관련 키워드
            #     # 기술적 지표, AI 분석 등도 요청
            self.tab_widget.setTabText(0, f"📊 실시간 ({self.symbol})") # 탭 이름에 심볼 표시


    def clear_all_tabs_data(self):
        """모든 탭의 데이터 표시를 초기화합니다."""
        # 실시간 탭
        self.price_label.setText("-")
        self.change_label.setText("-")
        self.change_label.setStyleSheet("font-size: 18px; color: #E0E0E0;") # 기본색
        self.volume_label.setText("-")
        self.high_label.setText("-")
        self.low_label.setText("-")
        self.prev_close_label.setText("-")
        self.market_cap_label.setText("-")
        self.last_updated_label.setText("최종 업데이트: -")

        # 기술적 지표 탭
        self.technical_table.setRowCount(0)

        # 뉴스 탭
        self.news_list_widget.clear()
        self.overall_sentiment_score_label.setText("-")
        self.overall_sentiment_trend_label.setText("-")
        self.news_count_label.setText("-")


        # AI 예측 탭
        self.ml_direction_label.setText("-")
        self.ml_confidence_label.setText("-")
        self.target_price_label.setText("-")
        self.overall_signal_label.setText("-")
        self.overall_signal_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #E0E0E0;") # 기본색
        self.analysis_summary_text.clear()


        # 경제 지표 탭은 심볼 변경과 무관하므로 여기서는 초기화하지 않음.
        # if hasattr(self, 'economic_indicators_table'):
        #     self.economic_indicators_table.setRowCount(0) # 이 줄을 주석 처리하거나 삭제
       
        logger.debug("심볼 관련 데이터 위젯 탭 내용 초기화됨 (경제 지표 탭 제외).")
    # --- MainWindow의 핸들러 함수들로부터 호출될 업데이트 메서드들 ---
    def update_realtime_quote_display(self, quote_data: dict):
        """ 실시간 호가 데이터를 받아 UI에 표시 (MainWindow에서 호출) """
        if not quote_data or quote_data.get('symbol') != self.symbol:
            return

        self.price_label.setText(f"${quote_data.get('price', 0):.2f}")
        change = quote_data.get('change', 0)
        change_percent = quote_data.get('change_percent', 0) * 100 # 0.05 -> 5%

        change_text = f"{change:+.2f} ({change_percent:+.2f}%)"
        if change > 0:
            self.change_label.setStyleSheet("font-size: 18px; color: #4CAF50;") # 초록색
        elif change < 0:
            self.change_label.setStyleSheet("font-size: 18px; color: #F44336;") # 빨간색
        else:
            self.change_label.setStyleSheet("font-size: 18px; color: #E0E0E0;") # 기본색
        self.change_label.setText(change_text)

        self.volume_label.setText(f"{quote_data.get('volume', 0):,}")
        self.high_label.setText(f"${quote_data.get('high', 0):.2f}")
        self.low_label.setText(f"${quote_data.get('low', 0):.2f}")
        self.prev_close_label.setText(f"${quote_data.get('previous_close', 0):.2f}")

        # 시가총액 등 추가 정보 (API 응답에 따라)
        # market_cap = quote_data.get('market_cap')
        # self.market_cap_label.setText(f"${market_cap:,}" if market_cap else "-")

        retrieved_at = quote_data.get('retrieved_at')
        if retrieved_at and isinstance(retrieved_at, pd.Timestamp):
            self.last_updated_label.setText(f"최종 업데이트: {retrieved_at.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
             self.last_updated_label.setText("최종 업데이트: -")


    def update_technical_indicators_display(self, symbol: str, timeframe: str, data_with_indicators: pd.DataFrame):
        """기술적 지표 데이터를 받아 테이블에 표시 (가장 최근 값)"""
        if data_with_indicators is None or data_with_indicators.empty:
            logger.warning(f"{symbol} ({timeframe})에 대한 기술적 지표 데이터가 없습니다.")
            self.technical_table.setRowCount(0)
            return

        self.technical_table.setSortingEnabled(False)
        self.technical_table.setRowCount(0)
        
        indicator_cols = [col for col in data_with_indicators.columns if any(kw in col for kw in ['SMA_', 'EMA_', 'RSI_', 'MACD', 'BB_'])] # 예시 키워드
        
        if not indicator_cols:
            logger.info(f"{symbol} ({timeframe}): 표시할 기술적 지표 컬럼이 데이터에 없습니다.")
            self.technical_table.setSortingEnabled(True)
            return

        if data_with_indicators.empty: # 추가적인 빈 DataFrame 체크
            logger.info(f"{symbol} ({timeframe}): 기술적 지표 DataFrame이 비어있습니다.")
            self.technical_table.setSortingEnabled(True)
            return
            
        latest_indicators = data_with_indicators.iloc[-1] 

        default_text_color = QColor("#E0E0E0")

        for col_name in indicator_cols:
            if col_name in latest_indicators:
                value = latest_indicators[col_name]
                if pd.isna(value):
                    value_str = "-"
                elif isinstance(value, float):
                    value_str = f"{value:.2f}" 
                else:
                    value_str = str(value)
                
                row_count = self.technical_table.rowCount()
                self.technical_table.insertRow(row_count)
                
                item_name = QTableWidgetItem(col_name)
                item_name.setForeground(default_text_color)
                
                item_value = QTableWidgetItem(value_str)
                item_value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item_value.setForeground(default_text_color)
                
                self.technical_table.setItem(row_count, 0, item_name)
                self.technical_table.setItem(row_count, 1, item_value)

        self.technical_table.setSortingEnabled(True)
        logger.info(f"{symbol} ({timeframe}) 기술적 지표 테이블 업데이트 완료 ({len(indicator_cols)}개 지표).")
    def update_news_display(self, news_items: list):
        """ 뉴스 목록 및 감성 요약 정보를 받아 UI에 표시 """
        self.news_list_widget.clear()
        if not news_items:
            no_news_item = QListWidgetItem("최근 뉴스가 없습니다.")
            no_news_item.setForeground(QColor("#AAAAAA"))
            self.news_list_widget.addItem(no_news_item)
            return

        # TODO: 뉴스 아이템에서 감성 점수 추출 및 아이콘/색상으로 표시
        # news_items는 [{'title': ..., 'url': ..., 'sentiment_score': 0.5 (예시)} ...] 형태라고 가정
        for news in news_items:
            title = news.get('title', '제목 없음')
            url = news.get('url')
            sentiment_score = news.get('sentiment_score') # 감성 분석 결과가 포함되어 있다고 가정

            display_text = title
            icon = None
            tooltip = f"URL: {url}\n"

            if sentiment_score is not None:
                if sentiment_score > 0.2:
                    icon = QIcon.fromTheme("face-smile") # 또는 사용자 정의 아이콘
                    display_text = f"🟢 {title}"
                    tooltip += f"감성점수: {sentiment_score:.2f} (긍정적)"
                elif sentiment_score < -0.2:
                    icon = QIcon.fromTheme("face-sad")
                    display_text = f"🔴 {title}"
                    tooltip += f"감성점수: {sentiment_score:.2f} (부정적)"
                else:
                    icon = QIcon.fromTheme("face-plain")
                    display_text = f"⚪ {title}"
                    tooltip += f"감성점수: {sentiment_score:.2f} (중립적)"


            list_item = QListWidgetItem(display_text)
            if icon:
                list_item.setIcon(icon)
            list_item.setData(Qt.UserRole, url) # URL 정보 저장
            list_item.setToolTip(tooltip)
            self.news_list_widget.addItem(list_item)

        # TODO: 전체 뉴스 감성 요약 정보 업데이트 (별도 데이터 필요)
        # self.overall_sentiment_score_label.setText(...)
        # self.overall_sentiment_trend_label.setText(...)
        # self.news_count_label.setText(f"{len(news_items)}개")


    def _on_news_item_double_clicked(self, item: QListWidgetItem):
        """ 뉴스 항목 더블클릭 시 URL 열기 """
        url = item.data(Qt.UserRole)
        if url:
            QDesktopServices.openUrl(QUrl(url))


    def update_analysis_display(self, analysis_data: dict):
        """ AI 분석 및 종합 판단 결과를 받아 UI에 표시 """
        ml_pred = analysis_data.get('ml_prediction', {})
        direction = ml_pred.get('action', '-') # BUY, SELL, HOLD 등
        confidence = ml_pred.get('confidence', 0.0)
        target_price = ml_pred.get('target_price') # 모델이 목표가를 제공한다면

        self.ml_direction_label.setText(str(direction))
        self.ml_confidence_label.setText(f"{confidence:.0%}" if confidence else "-")

        if target_price:
            self.target_price_label.setText(f"${target_price:.2f}")
        else:
            self.target_price_label.setText("-")

        # 종합 판단 로직 (Controller에서 처리 후 전달된 값 사용)
        overall_signal = analysis_data.get('overall_signal', '-') # 예: "강력 매수", "관망"
        signal_details = analysis_data.get('signal_reason', 'N/A')

        self.overall_signal_label.setText(overall_signal)
        if "매수" in overall_signal:
            self.overall_signal_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        elif "매도" in overall_signal:
            self.overall_signal_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #F44336;")
        else:
            self.overall_signal_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFC107;") # 관망 등은 주황색

        # 기술적 지표, 뉴스 감성 등 다른 분석 결과도 함께 표시
        summary_lines = [f"ML 예측: {direction} (신뢰도: {confidence:.0%})"]
        if target_price:
            summary_lines.append(f"AI 목표가: ${target_price:.2f}")

        tech_summary = []
        if 'sma_20' in analysis_data: tech_summary.append(f"SMA(20): {analysis_data['sma_20']:.2f}")
        if 'rsi_14' in analysis_data: tech_summary.append(f"RSI(14): {analysis_data['rsi_14']:.2f}")
        if tech_summary: summary_lines.append(f"기술적 지표: {', '.join(tech_summary)}")

        if 'news_sentiment' in analysis_data:
            summary_lines.append(f"뉴스 감성: {analysis_data['news_sentiment']:.2f}")

        summary_lines.append(f"\n종합 의견 근거: {signal_details}")
        self.analysis_summary_text.setPlainText("\n".join(summary_lines))

    # update_economic_indicators는 update_economic_indicators_display로 대체됨
    # def update_economic_indicators(self, indicators: list):
    #     pass
