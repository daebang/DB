# ui/widgets/data_widget.py

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pandas as pd # 데이터 처리를 위해 추가
import logging
from typing import Dict, Any, Optional, List # List 추가

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

        # AI 분석 탭 (이전 '예측 탭')
        self.ai_analysis_tab = self._create_ai_analysis_tab() # 메서드명 변경
        self.tab_widget.addTab(self.ai_analysis_tab, "🤖 AI 분석")

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
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10) # 여백 약간 추가

        # 종목별 종합 감성 정보 (QGroupBox으로 묶기)
        sentiment_summary_group = QGroupBox("뉴스 감성 요약 (최근 24시간)")
        summary_layout = QFormLayout()
        summary_layout.setSpacing(8)

        self.overall_sentiment_score_label = QLabel("-")
        self.overall_sentiment_trend_label = QLabel("-") # 예: "개선 중", "악화 중", "안정적"
        self.news_count_label = QLabel("-")
        self.positive_news_ratio_label = QLabel("-") # 긍정 뉴스 비율
        self.negative_news_ratio_label = QLabel("-") # 부정 뉴스 비율

        # 라벨 스타일링
        for lbl in [self.overall_sentiment_score_label, self.overall_sentiment_trend_label,
                    self.news_count_label, self.positive_news_ratio_label, self.negative_news_ratio_label]:
            lbl.setFont(QFont("Arial", 10))

        summary_layout.addRow("평균 감성 점수:", self.overall_sentiment_score_label)
        summary_layout.addRow("감성 트렌드:", self.overall_sentiment_trend_label)
        summary_layout.addRow("관련 뉴스 수:", self.news_count_label)
        summary_layout.addRow("긍정 뉴스 비율:", self.positive_news_ratio_label)
        summary_layout.addRow("부정 뉴스 비율:", self.negative_news_ratio_label)
        sentiment_summary_group.setLayout(summary_layout)
        layout.addWidget(sentiment_summary_group)

        # 뉴스 목록
        self.news_list_widget = QListWidget()
        self.news_list_widget.setAlternatingRowColors(True)
        self.news_list_widget.itemDoubleClicked.connect(self._on_news_item_double_clicked)
        self.news_list_widget.setStyleSheet("""
            QListWidget::item { padding: 5px; }
            QListWidget::item:hover { background-color: #4a4a4a; }
        """) # 아이템 패딩 및 호버 효과

        news_list_label = QLabel("최근 주요 뉴스")
        news_list_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(news_list_label)
        layout.addWidget(self.news_list_widget)
        return widget

    def _create_ai_analysis_tab(self): # 이전에 _create_prediction_tab 이었던 메서드
        """AI 기반 종합 분석 결과를 표시하는 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10) # 그룹 간 간격 추가

        # 기본 폰트 설정
        default_font_size = "10pt" # QLabel 기본 폰트 크기
        group_title_font_size = "11pt" # QGroupBox 제목 폰트 크기
        recommendation_action_font_size = "16pt" # 추천 행동 레이블 폰트 크기
        
        # 종합 추천 섹션
        recommendation_group = QGroupBox("AI 종합 투자 추천")
        recommendation_group.setStyleSheet(f"QGroupBox {{ font-size: {group_title_font_size}; font-weight: bold; }}")
        rec_layout = QFormLayout()
        rec_layout.setSpacing(10) # 항목 간 간격 증가
        rec_layout.setLabelAlignment(Qt.AlignLeft) # 레이블 왼쪽 정렬

        self.current_price_for_recommendation_label = QLabel("현재가(분석시점): -")
        self.current_price_for_recommendation_label.setStyleSheet(f"font-size: {default_font_size}; color: #B0B0B0;")

        # "추천 행동:" 텍스트를 가진 QLabel을 별도로 생성하여 스타일 적용
        recommendation_action_title_label = QLabel("추천 행동:")
        recommendation_action_title_label.setStyleSheet(f"font-size: {default_font_size}; font-weight: bold;")

        self.recommendation_action_label = QLabel("-") 
        self.recommendation_action_label.setStyleSheet(f"font-size: {recommendation_action_font_size}; font-weight: bold;") # 기본 색상은 update_analysis_display에서 설정

        self.overall_confidence_label = QLabel("종합 신뢰도: -")
        self.overall_confidence_label.setStyleSheet(f"font-size: {default_font_size};")

        self.risk_level_label = QLabel("리스크 수준: -")
        self.risk_level_label.setStyleSheet(f"font-size: {default_font_size};")
        
        rec_layout.addRow(self.current_price_for_recommendation_label)
        rec_layout.addRow(recommendation_action_title_label, self.recommendation_action_label) # 제목 레이블과 값 레이블 분리
        rec_layout.addRow(self.overall_confidence_label)
        rec_layout.addRow(self.risk_level_label)
        
        recommendation_group.setLayout(rec_layout)
        layout.addWidget(recommendation_group)

        # 상세 분석 내용 섹션 (QTextEdit 사용)
        analysis_details_group = QGroupBox("상세 분석 및 기간별 예측")
        analysis_details_group.setStyleSheet(f"QGroupBox {{ font-size: {group_title_font_size}; font-weight: bold; }}")
        details_layout = QVBoxLayout()
        
        self.analysis_summary_text = QTextEdit()
        self.analysis_summary_text.setReadOnly(True)
        self.analysis_summary_text.setMinimumHeight(300) # 상세 내용 표시 위해 높이 증가
        self.analysis_summary_text.setStyleSheet(f"font-size: {default_font_size}; border: 1px solid #4A4A4A;") # 테두리 추가

        details_layout.addWidget(self.analysis_summary_text)
        analysis_details_group.setLayout(details_layout)
        layout.addWidget(analysis_details_group)
        
        layout.addStretch()
        widget.setLayout(layout) # 최종 레이아웃 설정 추가
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
        header.setSectionResizeMode(3, QHeaderView.Stretch) # 이벤트 컬럼 확장
        
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
        self.positive_news_ratio_label.setText("-")
        self.negative_news_ratio_label.setText("-")
        self.overall_sentiment_score_label.setStyleSheet("")
        self.overall_sentiment_trend_label.setStyleSheet("")

        # AI 분석 탭 초기화
        self.recommendation_action_label.setText("-")
        self.recommendation_action_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.current_price_for_recommendation_label.setText("현재가: -")
        self.overall_confidence_label.setText("종합 신뢰도: -")
        self.risk_level_label.setText("리스크 수준: -")
        self.analysis_summary_text.setHtml("") # HTML 내용도 클리어


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
        
        market_cap = quote_data.get('market_cap')
        self.market_cap_label.setText(f"${market_cap:,}" if market_cap else "-")

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
        
    # update_news_display는 Controller로부터 news_items와 news_summary_data를 함께 받도록 수정
    # 또는 news_items만 받고, DataWidget 내부에서 controller.get_news_summary(symbol) 호출
    def update_news_display(self, news_items: list, news_summary: Optional[Dict] = None): # news_summary 인자 추가
        self.news_list_widget.clear() # 기존 목록 지우기

        if news_summary:
            self.overall_sentiment_score_label.setText(f"{news_summary.get('avg_sentiment', 0.0):.2f}")
            self.overall_sentiment_trend_label.setText(str(news_summary.get('sentiment_trend', '-')))
            self.news_count_label.setText(str(news_summary.get('news_count', 0)))
            self.positive_news_ratio_label.setText(f"{news_summary.get('positive_news_ratio', 0.0):.1%}")
            self.negative_news_ratio_label.setText(f"{news_summary.get('negative_news_ratio', 0.0):.1%}")

            # 점수/트렌드에 따른 색상 변경 (예시)
            avg_sent = news_summary.get('avg_sentiment', 0.0)
            if avg_sent > 0.1: self.overall_sentiment_score_label.setStyleSheet("color: #4CAF50;") # 초록
            elif avg_sent < -0.1: self.overall_sentiment_score_label.setStyleSheet("color: #F44336;") # 빨강
            else: self.overall_sentiment_score_label.setStyleSheet("") # 기본색

            trend = news_summary.get('sentiment_trend', '-')
            if trend == 'improving': self.overall_sentiment_trend_label.setStyleSheet("color: #4CAF50;")
            elif trend == 'deteriorating': self.overall_sentiment_trend_label.setStyleSheet("color: #F44336;")
            else: self.overall_sentiment_trend_label.setStyleSheet("")

        else: # 요약 정보가 없으면 기본값으로
            self.overall_sentiment_score_label.setText("-")
            self.overall_sentiment_trend_label.setText("-")
            self.news_count_label.setText("-")
            self.positive_news_ratio_label.setText("-")
            self.negative_news_ratio_label.setText("-")
            self.overall_sentiment_score_label.setStyleSheet("")
            self.overall_sentiment_trend_label.setStyleSheet("")


        if not news_items:
            no_news_item = QListWidgetItem("최근 뉴스가 없습니다.")
            no_news_item.setForeground(QColor("#AAAAAA"))
            self.news_list_widget.addItem(no_news_item)
            return

        for news in news_items[:30]: # 너무 많은 뉴스는 UI 성능 저하, 최근 30개만 표시
            title = news.get('title', '제목 없음')
            url = news.get('url')
            source_name = news.get('source_name', news.get('source', {}).get('name', 'N/A'))
            published_at_dt = news.get('published_at')
            if isinstance(published_at_dt, str): # 문자열이면 datetime으로 변환
                published_at_dt = pd.to_datetime(published_at_dt, errors='coerce', utc=True)

            published_at_str = published_at_dt.strftime('%Y-%m-%d %H:%M') if pd.notna(published_at_dt) else 'N/A'

            # sentiment_score는 news_items에 이미 포함되어 있다고 가정 (NewsCollector 또는 Controller에서 처리)
            sentiment_score = news.get('sentiment_score') # IntegratedAnalyzer 결과가 아닌 개별 뉴스 감성
            sentiment_label = news.get('sentiment_label', self._get_label_from_score(sentiment_score) if sentiment_score is not None else "neutral")


            # HTML로 아이템 텍스트 구성
            item_html = f"""
            <div style='margin-bottom: 3px;'>
                <strong style='font-size: 10pt;'>{title}</strong><br/>
                <small style='color: #AAAAAA;'>{source_name} - {published_at_str}</small>
            </div>
            """
            list_item = QListWidgetItem()
            # list_item.setText(title) # 직접 텍스트 설정 대신 라벨 사용

            # QLabel을 사용하여 HTML 렌더링 (더 유연함)
            item_label = QLabel(item_html)
            item_label.setWordWrap(True)
            item_label.setOpenExternalLinks(False) # 링크는 더블클릭으로 처리

            # 아이콘 설정 (기존 로직 활용)
            icon = QIcon()
            text_color = QColor(self.palette().color(QPalette.Text)) # 현재 테마의 기본 텍스트 색상

            if sentiment_label == 'positive':
                icon = QIcon.fromTheme("face-smile-symbolic", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-apply-16.png")) # 대체 아이콘
                text_color = QColor("#4CAF50")
            elif sentiment_label == 'negative':
                icon = QIcon.fromTheme("face-sad-symbolic", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-cancel-16.png"))
                text_color = QColor("#F44336")
            else: # neutral
                icon = QIcon.fromTheme("face-plain-symbolic", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-help-16.png"))

            list_item.setIcon(icon)
            list_item.setData(Qt.UserRole, url)
            list_item.setToolTip(f"{title}\nURL: {url}\nSentiment: {sentiment_label} ({sentiment_score:.2f} if sentiment_score is not None else 'N/A')\n더블클릭하여 원본 기사 열기")
            # list_item.setForeground(text_color) # QListWidgetItem 자체의 색상보다 내부 라벨 스타일 활용

            # QListWidgetItem에 QLabel을 직접 설정할 수 없으므로,
            # QListWidget에 setItemWidget을 사용하거나, list_item.setText(item_html) 후 view()에서 HTML delegate 사용.
            # 여기서는 QListWidgetItem의 기본 텍스트 표시 기능을 활용하고, 스타일은 아이콘과 툴팁으로 강화.
            # 더 복잡한 레이아웃을 원하면 setItemWidget 사용.
            # 간단히는, 타이틀만 QListWidgetItem 텍스트로, 나머지는 툴팁으로.
            list_item.setText(f"{title} ({source_name})") # 아이템에 표시될 주 텍스트
            if sentiment_label == 'positive': list_item.setForeground(QColor("#C8E6C9")) # 연한 초록
            elif sentiment_label == 'negative': list_item.setForeground(QColor("#FFCDD2")) # 연한 빨강
            
            self.news_list_widget.addItem(list_item)


    def _get_label_from_score(self, score: float, positive_threshold=0.1, negative_threshold=-0.1) -> str:
        """점수로부터 감성 라벨을 반환합니다. (DataWidget 내 로컬 헬퍼)"""
        if score is None: return 'neutral'
        if score > positive_threshold: return 'positive'
        if score < negative_threshold: return 'negative'
        return 'neutral'
    
    def _on_news_item_double_clicked(self, item: QListWidgetItem):
        url = item.data(Qt.UserRole)
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _on_news_item_double_clicked(self, item: QListWidgetItem):
        """ 뉴스 항목 더블클릭 시 URL 열기 """
        url = item.data(Qt.UserRole)
        if url:
            QDesktopServices.openUrl(QUrl(url))


    def update_analysis_display(self, predictions_data: Dict[str, Any]):
        """
        TimeframePredictor로부터 받은 종합 예측 데이터를 UI에 표시합니다.
        predictions_data는 timeframe_predictor.predict_all_timeframes()의 반환값입니다.
        """
        if not predictions_data or predictions_data.get('symbol') != self.symbol:
            logger.warning(f"AI 분석 데이터가 없거나 심볼({predictions_data.get('symbol')}) 불일치 ({self.symbol}). UI 업데이트 건너뜀.")
            # UI 초기화 또는 "데이터 없음" 메시지 표시
            self.recommendation_action_label.setText("데이터 없음")
            self.recommendation_action_label.setStyleSheet(f"font-size: {getattr(self, 'recommendation_action_font_size', '16pt')}; font-weight: bold; color: #AAAAAA;")
            self.current_price_for_recommendation_label.setText("현재가(분석시점): -")
            self.overall_confidence_label.setText("종합 신뢰도: -")
            self.risk_level_label.setText("리스크 수준: -")
            self.analysis_summary_text.setHtml("<p style='color:#AAAAAA; font-style:italic; font-size:10pt; padding:10px;'>AI 분석 데이터가 없거나 현재 종목과 일치하지 않습니다. 데이터를 로드 중이거나 분석을 기다려주세요.</p>")
            return

        # 1. 종합 추천 섹션 업데이트
        recommendation = predictions_data.get('recommendation', {})
        action = recommendation.get('action', '데이터 분석중') # 기본값 변경
        strength = recommendation.get('strength', '') # 기본값 변경
        action_strength_text = f"{action}"
        if strength and action != '데이터 분석중': # '데이터 분석중'일 때는 강도 표시 안함
            action_strength_text += f" ({strength.capitalize()})"
        
        self.recommendation_action_label.setText(action_strength_text)

        current_price = predictions_data.get('current_price')
        self.current_price_for_recommendation_label.setText(f"현재가(분석시점): ${current_price:.2f}" if current_price is not None else "현재가: -")

        overall_conf = predictions_data.get('overall_confidence', 0.0)
        self.overall_confidence_label.setText(f"종합 신뢰도: {overall_conf:.1%}") # 소수점 한 자리 %
        
        risk_level = recommendation.get('risk_level', 'N/A').capitalize()
        self.risk_level_label.setText(f"리스크 수준: {risk_level}")

        # 추천 행동에 따른 색상
        # QLabel의 스타일시트에서 font-size를 가져오기 위해 getattr 사용 및 기본값 설정
        base_style = f"font-size: {getattr(self, 'recommendation_action_font_size', '16pt')}; font-weight: bold;"
        if action == 'BUY':
            self.recommendation_action_label.setStyleSheet(f"{base_style} color: #4CAF50;") # 초록
        elif action == 'SELL':
            self.recommendation_action_label.setStyleSheet(f"{base_style} color: #F44336;") # 빨강
        elif action == 'HOLD':
            self.recommendation_action_label.setStyleSheet(f"{base_style} color: #FFC107;") # 주황
        else: # 데이터 분석중 또는 기타
            self.recommendation_action_label.setStyleSheet(f"{base_style} color: #E0E0E0;")

        # 2. 상세 분석 내용 (HTML로 구성)
        summary_html_lines = []
        # 스타일 정의 (HTML 상단에 한 번만 정의하여 재사용)
        styles = {
            "h2": "font-size:14pt; color:#E0E0E0; margin-bottom:10px;",
            "h3": "font-size:12pt; color:#D0D0D0; margin-top:15px; margin-bottom:8px;",
            "h4": "font-size:11pt; color:#C0C0C0; margin-top:10px; margin-bottom:5px;",
            "p_base": f"font-size:{getattr(self, 'default_font_size', '10pt')}; line-height:1.6;",
            "p_factor": f"font-size:9pt; color:#B0B0B0; line-height:1.5;",
            "ul_base": f"font-size:{getattr(self, 'default_font_size', '10pt')}; margin-left:0px; padding-left:20px; list-style-type: disclosure-closed;", # 목록 스타일 변경
            "ul_factor": "font-size:9pt; margin-left:0px; padding-left:20px; color:#A0A0A0; list-style-type: circle;",
            "positive_ret": "color:#4CAF50; font-weight:bold;",
            "negative_ret": "color:#F44336; font-weight:bold;",
            "neutral_ret": "color:#E0E0E0; font-weight:bold;",
        }

        summary_html_lines.append(f"<h2 style='{styles['h2']}'>종합 투자 추천 ({self.symbol})</h2>")
        action_color = self.recommendation_action_label.styleSheet().split('color: ')[-1].split(';')[0] # 현재 레이블 색상 가져오기
        summary_html_lines.append(f"<p style='{styles['p_base']}'><b>추천 행동:</b> <span style='color:{action_color}; font-weight:bold;'>{action_strength_text}</span></p>")
        summary_html_lines.append(f"<p style='{styles['p_base']}'><b>현재가 (분석 시점):</b> ${current_price:.2f}</p>")
        summary_html_lines.append(f"<p style='{styles['p_base']}'><b>종합 신뢰도:</b> {overall_conf:.1%}</p>")
        summary_html_lines.append(f"<p style='{styles['p_base']}'><b>리스크 수준:</b> {risk_level}</p>")
        
        reasoning = recommendation.get('reasoning', [])
        if reasoning:
            summary_html_lines.append(f"<b style='font-size:{getattr(self, 'default_font_size', '10pt')};'>주요 근거:</b><ul style='{styles['ul_base']}'>")
            for reason in reasoning:
                summary_html_lines.append(f"<li>{reason}</li>")
            summary_html_lines.append("</ul>")

        suggested_strategy = recommendation.get('suggested_strategy')
        if suggested_strategy:
            summary_html_lines.append(f"<p style='{styles['p_base']}'><b>제안 전략:</b> {suggested_strategy}</p>")
        
        summary_html_lines.append(f"<hr><h3 style='{styles['h3']}'>기간별 상세 예측:</h3>")

        timeframes_data = predictions_data.get('timeframes', {})
        
        def format_period_prediction(period_name: str, pred_detail: Dict, period_group_style: str) -> List[str]:
            lines = []
            pred_return = pred_detail.get('predicted_return', 0)
            pred_price = pred_detail.get('predicted_price')
            price_range_low, price_range_high = pred_detail.get('price_range', (None, None))
            conf = pred_detail.get('confidence', 0)
            factors = pred_detail.get('factors', []) 
            if not factors: factors = pred_detail.get('key_factors',[])

            news_impact_info = pred_detail.get('news_impact', {}) 
            econ_impact_info = pred_detail.get('economic_impact', {}) 
            
            if pred_return > 0.01 : ret_style = styles['positive_ret'] # 미미한 변동은 중립으로
            elif pred_return < -0.01: ret_style = styles['negative_ret']
            else: ret_style = styles['neutral_ret']
            
            lines.append(f"<p style='{period_group_style}'><b>{period_name}:</b> <span style='{ret_style}'>{pred_return:+.2f}%</span> ")
            lines.append(f"(예상가: ${pred_price:.2f}" if pred_price is not None else "(예상가: -)")
            if price_range_low is not None and price_range_high is not None:
                lines.append(f", 범위: ${price_range_low:.2f} ~ ${price_range_high:.2f}")
            lines.append(f", 신뢰도: {conf:.1%})</p>")

            if factors:
                lines.append(f"<ul style='{styles['ul_factor']}'>")
                for factor in factors: lines.append(f"<li>{factor}</li>")
                lines.append("</ul>")
            if news_impact_info and news_impact_info.get('impact') != 'neutral' and news_impact_info.get('news_count', 0) > 0 :
                 lines.append(f"<p style='{styles['p_factor']}'>&nbsp;&nbsp;└ 뉴스 영향: {news_impact_info.get('impact', 'N/A')} (점수: {news_impact_info.get('score', 0):.2f}, 뉴스 {news_impact_info.get('news_count',0)}개)</p>")
            if econ_impact_info and econ_impact_info.get('impact') != 'neutral' and econ_impact_info.get('key_indicators'):
                lines.append(f"<p style='{styles['p_factor']}'>&nbsp;&nbsp;└ 경제지표 영향: {econ_impact_info.get('impact','N/A')}</p>")
            return lines

        # 기간별 예측 표시
        term_map = {'short_term': "단기 (1일 ~ 3일)", 'medium_term': "중기 (1주일 ~ 2개월)", 'long_term': "장기 (3개월 이상)"}
        for term_key, term_title in term_map.items():
            term_preds = timeframes_data.get(term_key, {})
            if term_preds:
                summary_html_lines.append(f"<h4 style='{styles['h4']}'>{term_title}:</h4>")
                # 기간(예: '1일', '1주일') 순서대로 정렬하기 위한 키 생성 함수
                def sort_key_period(item_tuple):
                    period_str = item_tuple[0] # 예: "1일", "2주일", "3개월"
                    num_part = ''.join(filter(str.isdigit, period_str))
                    unit_part = ''.join(filter(str.isalpha, period_str)).lower()
                    num = int(num_part) if num_part else 0
                    
                    if '일' in unit_part: multiplier = 1
                    elif '주' in unit_part: multiplier = 7
                    elif '개월' in unit_part: multiplier = 30
                    else: multiplier = 0
                    return num * multiplier

                sorted_periods = sorted(term_preds.items(), key=sort_key_period)

                for period, detail in sorted_periods:
                    summary_html_lines.extend(format_period_prediction(period, detail, styles['p_base']))
                summary_html_lines.append("<br>")

        if not summary_html_lines: # 모든 예측이 비어있을 경우
             summary_html_lines.append(f"<p style='{styles['p_base']} color:#AAAAAA; font-style:italic;'>세부 예측 정보가 없습니다.</p>")

        self.analysis_summary_text.setHtml("<div style='padding:5px;'>" + "".join(summary_html_lines) + "</div>")
        logger.info(f"AI 분석 결과 UI 업데이트 완료 ({self.symbol}). 추천: {action_strength_text}")