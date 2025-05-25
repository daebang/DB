from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class TradingWidget(QWidget):
    """트레이딩 위젯"""
    
    def __init__(self):
        super().__init__()
        self.symbol = "AAPL"
        self.setup_ui()
    
    def setup_ui(self):
        """UI 설정"""
        layout = QHBoxLayout()
        
        # 주문 섹션
        order_group = self.create_order_section()
        layout.addWidget(order_group)
        
        # 포지션 섹션
        position_group = self.create_position_section()
        layout.addWidget(position_group)
        
        # 전략 섹션
        strategy_group = self.create_strategy_section()
        layout.addWidget(strategy_group)
        
        self.setLayout(layout)
    
    def create_order_section(self):
        """주문 섹션 생성"""
        group = QGroupBox("주문")
        layout = QVBoxLayout()
        
        # 주문 타입
        order_type_layout = QHBoxLayout()
        self.market_radio = QRadioButton("시장가")
        self.limit_radio = QRadioButton("지정가")
        self.market_radio.setChecked(True)
        
        order_type_layout.addWidget(self.market_radio)
        order_type_layout.addWidget(self.limit_radio)
        
        # 주문 입력
        form_layout = QFormLayout()
        
        self.quantity_spin = QSpinBox()
        self.quantity_spin.setRange(1, 10000)
        self.quantity_spin.setValue(100)
        
        self.price_spin = QDoubleSpinBox()
        self.price_spin.setRange(0.01, 10000.00)
        self.price_spin.setValue(150.00)
        self.price_spin.setEnabled(False)
        
        # 라디오 버튼 연결
        self.market_radio.toggled.connect(lambda checked: self.price_spin.setEnabled(not checked))
        
        form_layout.addRow("수량:", self.quantity_spin)
        form_layout.addRow("가격:", self.price_spin)
        
        # 주문 버튼
        button_layout = QHBoxLayout()
        
        self.buy_button = QPushButton("매수")
        self.buy_button.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        self.buy_button.clicked.connect(self.place_buy_order)
        
        self.sell_button = QPushButton("매도")
        self.sell_button.setStyleSheet("background-color: #f44336; font-weight: bold;")
        self.sell_button.clicked.connect(self.place_sell_order)
        
        button_layout.addWidget(self.buy_button)
        button_layout.addWidget(self.sell_button)
        
        layout.addLayout(order_type_layout)
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)
        
        group.setLayout(layout)
        return group
    
    def create_position_section(self):
        """포지션 섹션 생성"""
        group = QGroupBox("포지션 현황")
        layout = QVBoxLayout()
        
        # 포지션 테이블
        self.position_table = QTableWidget(0, 6)
        self.position_table.setHorizontalHeaderLabels([
            "종목", "수량", "평균단가", "현재가", "손익", "손익률"
        ])
        
        # 더미 데이터
        self.add_position_row("AAPL", 100, 148.50, 150.00, 150.00, 1.01)
        self.add_position_row("GOOGL", 50, 2800.00, 2850.00, 2500.00, 1.79)
        
        # 총 손익 표시
        total_layout = QHBoxLayout()
        self.total_pnl_label = QLabel("총 손익: +$2,650.00 (+1.25%)")
        self.total_pnl_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ff00;")
        total_layout.addWidget(self.total_pnl_label)
        total_layout.addStretch()
        
        layout.addWidget(self.position_table)
        layout.addLayout(total_layout)
        
        group.setLayout(layout)
        return group
    
    def create_strategy_section(self):
        """전략 섹션 생성"""
        group = QGroupBox("자동 트레이딩")
        layout = QVBoxLayout()
        
        # 전략 상태
        status_layout = QHBoxLayout()
        self.strategy_status_label = QLabel("상태: 중단")
        self.strategy_status_label.setStyleSheet("font-weight: bold; color: #ff9800;")
        status_layout.addWidget(self.strategy_status_label)
        status_layout.addStretch()
        
        # 전략 설정
        form_layout = QFormLayout()
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["ML 기반 전략", "기술적 분석 전략", "감성 기반 전략"])
        
        self.auto_trade_check = QCheckBox("자동 주문 실행")
        self.risk_limit_spin = QDoubleSpinBox()
        self.risk_limit_spin.setRange(0.01, 0.50)
        self.risk_limit_spin.setValue(0.05)
        self.risk_limit_spin.setSuffix("%")
        
        form_layout.addRow("전략:", self.strategy_combo)
        form_layout.addRow("", self.auto_trade_check)
        form_layout.addRow("리스크 한도:", self.risk_limit_spin)
        
        # 제어 버튼
        control_layout = QHBoxLayout()
        
        self.start_strategy_button = QPushButton("전략 시작")
        self.start_strategy_button.clicked.connect(self.start_strategy)
        
        self.stop_strategy_button = QPushButton("전략 중단")
        self.stop_strategy_button.clicked.connect(self.stop_strategy)
        self.stop_strategy_button.setEnabled(False)
        
        control_layout.addWidget(self.start_strategy_button)
        control_layout.addWidget(self.stop_strategy_button)
        
        layout.addLayout(status_layout)
        layout.addLayout(form_layout)
        layout.addLayout(control_layout)
        
        group.setLayout(layout)
        return group
    
    def add_position_row(self, symbol, quantity, avg_price, current_price, pnl, pnl_percent):
        """포지션 행 추가"""
        row = self.position_table.rowCount()
        self.position_table.insertRow(row)
        
        self.position_table.setItem(row, 0, QTableWidgetItem(symbol))
        self.position_table.setItem(row, 1, QTableWidgetItem(str(quantity)))
        self.position_table.setItem(row, 2, QTableWidgetItem(f"${avg_price:.2f}"))
        self.position_table.setItem(row, 3, QTableWidgetItem(f"${current_price:.2f}"))
        
        # 손익 색상 설정
        pnl_item = QTableWidgetItem(f"${pnl:+.2f}")
        pnl_percent_item = QTableWidgetItem(f"{pnl_percent:+.2f}%")
        
        color = "#00ff00" if pnl >= 0 else "#ff0000"
        pnl_item.setForeground(QColor(color))
        pnl_percent_item.setForeground(QColor(color))
        
        self.position_table.setItem(row, 4, pnl_item)
        self.position_table.setItem(row, 5, pnl_percent_item)
    
    def place_buy_order(self):
        """매수 주문"""
        quantity = self.quantity_spin.value()
        if self.limit_radio.isChecked():
            price = self.price_spin.value()
            QMessageBox.information(self, "주문 접수", f"{self.symbol} {quantity}주 지정가 매수 주문 (${price:.2f})")
        else:
            QMessageBox.information(self, "주문 접수", f"{self.symbol} {quantity}주 시장가 매수 주문")
    
    def place_sell_order(self):
        """매도 주문"""
        quantity = self.quantity_spin.value()
        if self.limit_radio.isChecked():
            price = self.price_spin.value()
            QMessageBox.information(self, "주문 접수", f"{self.symbol} {quantity}주 지정가 매도 주문 (${price:.2f})")
        else:
            QMessageBox.information(self, "주문 접수", f"{self.symbol} {quantity}주 시장가 매도 주문")
    
    def start_strategy(self):
        """전략 시작"""
        self.strategy_status_label.setText("상태: 실행중")
        self.strategy_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        self.start_strategy_button.setEnabled(False)
        self.stop_strategy_button.setEnabled(True)
    
    def stop_strategy(self):
        """전략 중단"""
        self.strategy_status_label.setText("상태: 중단")
        self.strategy_status_label.setStyleSheet("font-weight: bold; color: #ff9800;")
        self.start_strategy_button.setEnabled(True)
        self.stop_strategy_button.setEnabled(False)
    
    def set_symbol(self, symbol):
        """종목 설정"""
        self.symbol = symbol