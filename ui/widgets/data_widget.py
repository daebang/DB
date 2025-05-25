from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class DataWidget(QWidget):
    """데이터 표시 위젯"""
    
    def __init__(self):
        super().__init__()
        self.symbol = "AAPL"
        self.setup_ui()
    
    def setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout()
        
        # 탭 위젯
        self.tab_widget = QTabWidget()
        
        # 실시간 데이터 탭
        self.realtime_tab = self.create_realtime_tab()
        self.tab_widget.addTab(self.realtime_tab, "실시간")
        
        # 기술적 지표 탭
        self.technical_tab = self.create_technical_tab()
        self.tab_widget.addTab(self.technical_tab, "기술적 지표")
        
        # 뉴스 감성 탭
        self.sentiment_tab = self.create_sentiment_tab()
        self.tab_widget.addTab(self.sentiment_tab, "뉴스 감성")
        
        # 예측 탭
        self.prediction_tab = self.create_prediction_tab()
        self.tab_widget.addTab(self.prediction_tab, "AI 예측")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def create_realtime_tab(self):
        """실시간 데이터 탭 생성"""
        widget = QWidget()
        layout = QFormLayout()
        
        # 실시간 데이터 라벨들
        self.price_label = QLabel("$150.00")
        self.change_label = QLabel("+$2.50 (+1.69%)")
        self.volume_label = QLabel("1,234,567")
        self.high_label = QLabel("$152.30")
        self.low_label = QLabel("$148.90")
        
        # 스타일 적용
        self.price_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff00;")
        self.change_label.setStyleSheet("font-size: 16px; color: #00ff00;")
        
        layout.addRow("현재가:", self.price_label)
        layout.addRow("변동:", self.change_label)
        layout.addRow("거래량:", self.volume_label)
        layout.addRow("고가:", self.high_label)
        layout.addRow("저가:", self.low_label)
        
        widget.setLayout(layout)
        return widget
    
    def create_technical_tab(self):
        """기술적 지표 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 테이블 위젯
        self.technical_table = QTableWidget(10, 2)
        self.technical_table.setHorizontalHeaderLabels(["지표", "값"])
        
        # 더미 데이터
        indicators = [
            ("RSI (14)", "65.4"),
            ("MACD", "1.23"),
            ("SMA (20)", "$149.50"),
            ("EMA (12)", "$150.20"),
            ("볼린저 상단", "$155.00"),
            ("볼린저 하단", "$145.00"),
            ("스토캐스틱 %K", "72.1"),
            ("ATR", "2.45"),
            ("거래량 비율", "1.34"),
            ("변동성", "18.5%")
        ]
        
        for i, (indicator, value) in enumerate(indicators):
            self.technical_table.setItem(i, 0, QTableWidgetItem(indicator))
            self.technical_table.setItem(i, 1, QTableWidgetItem(value))
        
        layout.addWidget(self.technical_table)
        widget.setLayout(layout)
        return widget
    
    def create_sentiment_tab(self):
        """뉴스 감성 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 감성 점수 표시
        sentiment_layout = QHBoxLayout()
        
        self.sentiment_score_label = QLabel("감성 점수: +0.65")
        self.sentiment_score_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff00;")
        
        self.sentiment_trend_label = QLabel("트렌드: 상승")
        
        sentiment_layout.addWidget(self.sentiment_score_label)
        sentiment_layout.addWidget(self.sentiment_trend_label)
        
        # 뉴스 목록
        self.news_list = QListWidget()
        news_items = [
            "🟢 Apple 분기 실적 예상치 상회",
            "🟡 iPhone 15 판매량 안정적 성장",
            "🟢 AI 칩 개발 관련 긍정적 전망",
            "🔴 중국 시장 규제 우려 확산",
            "🟢 서비스 부문 매출 신기록 달성"
        ]
        
        for item in news_items:
            self.news_list.addItem(item)
        
        layout.addLayout(sentiment_layout)
        layout.addWidget(QLabel("최근 뉴스:"))
        layout.addWidget(self.news_list)
        
        widget.setLayout(layout)
        return widget
    
    def create_prediction_tab(self):
        """AI 예측 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 예측 결과 표시
        prediction_layout = QFormLayout()
        
        self.ml_prediction_label = QLabel("상승 (신뢰도: 78%)")
        self.ml_prediction_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        
        self.lstm_prediction_label = QLabel("$152.30 (+1.5%)")
        self.lstm_prediction_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        
        self.ensemble_prediction_label = QLabel("매수 추천")
        self.ensemble_prediction_label.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 16px;")
        
        prediction_layout.addRow("ML 분류:", self.ml_prediction_label)
        prediction_layout.addRow("LSTM 가격 예측:", self.lstm_prediction_label)
        prediction_layout.addRow("종합 판단:", self.ensemble_prediction_label)
        
        # 모델 성능 정보
        performance_group = QGroupBox("모델 성능")
        performance_layout = QFormLayout()
        
        performance_layout.addRow("ML 정확도:", QLabel("82.5%"))
        performance_layout.addRow("LSTM MAE:", QLabel("$1.23"))
        performance_layout.addRow("최근 수익률:", QLabel("+15.7%"))
        
        performance_group.setLayout(performance_layout)
        
        layout.addLayout(prediction_layout)
        layout.addWidget(performance_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def set_symbol(self, symbol):
        """종목 설정"""
        self.symbol = symbol
    
    def update_data(self, data):
        """데이터 업데이트"""
        if 'price' in data:
            self.price_label.setText(f"${data['price']:.2f}")
        if 'change' in data:
            change_color = "#00ff00" if data['change'] >= 0 else "#ff0000"
            self.change_label.setText(f"{data['change']:+.2f} ({data.get('change_percent', 0):+.2f}%)")
            self.change_label.setStyleSheet(f"font-size: 16px; color: {change_color};")