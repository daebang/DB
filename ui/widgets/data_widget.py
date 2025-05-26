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

        self.technical_table = QTableWidget(0, 2) # 행은 동적으로 추가
        self.technical_table.setHorizontalHeaderLabels(["지표명", "값"])
        self.technical_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.technical_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.technical_table.setEditTriggers(QAbstractItemView.NoEditTriggers) # 편집 불가
        self.technical_table.setAlternatingRowColors(True) # 행 색상 교차

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

        self.economic_indicators_table = QTableWidget(0, 3) # 지표명, 현재값, 다음 발표일
        self.economic_indicators_table.setHorizontalHeaderLabels(["주요 경제 지표", "최근 값", "다음 발표 (예상)"])
        self.economic_indicators_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.economic_indicators_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.economic_indicators_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.economic_indicators_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.economic_indicators_table.setAlternatingRowColors(True)

        # TODO: 컨트롤러를 통해 주요 경제 지표 데이터 로드 및 업데이트 로직 필요
        # 예시 데이터
        # self.add_economic_indicator_row("미국 CPI (YoY)", "3.5%", "2025-06-15")
        # self.add_economic_indicator_row("미 연준 기준금리", "5.50%", "2025-07-01")

        layout.addWidget(self.economic_indicators_table)
        return widget

    def add_economic_indicator_row(self, name: str, value: str, next_release: str):
        row_count = self.economic_indicators_table.rowCount()
        self.economic_indicators_table.insertRow(row_count)
        self.economic_indicators_table.setItem(row_count, 0, QTableWidgetItem(name))
        self.economic_indicators_table.setItem(row_count, 1, QTableWidgetItem(value))
        self.economic_indicators_table.setItem(row_count, 2, QTableWidgetItem(next_release))


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

        # 경제 지표 탭 (심볼 변경과 무관할 수 있으나, 필요시 초기화)
        # self.economic_indicators_table.setRowCount(0)


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


    def update_technical_indicators_display(self, indicators: dict):
        """ 기술적 지표 데이터를 받아 테이블에 표시 """
        self.technical_table.setRowCount(0) # 기존 내용 삭제
        for name, value in indicators.items():
            row_count = self.technical_table.rowCount()
            self.technical_table.insertRow(row_count)
            self.technical_table.setItem(row_count, 0, QTableWidgetItem(str(name)))
            item_value = QTableWidgetItem(f"{value:.2f}" if isinstance(value, (float, int)) else str(value))
            item_value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.technical_table.setItem(row_count, 1, item_value)

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

    def update_economic_indicators(self, indicators: list):
        """ 경제 지표 데이터를 받아 테이블에 표시 """
        self.economic_indicators_table.setRowCount(0)
        for indicator in indicators: # indicator는 {'name': ..., 'value': ..., 'next_release': ...} 형태의 dict라고 가정
            self.add_economic_indicator_row(
                indicator.get('name','-'),
                indicator.get('value','-'),
                indicator.get('next_release','-')
            )