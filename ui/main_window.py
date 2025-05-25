import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from ui.widgets.chart_widget import ChartWidget
from ui.widgets.data_widget import DataWidget
from ui.widgets.trading_widget import TradingWidget
from ui.dialogs.settings_dialog import SettingsDialog

class MainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("통합 주식 분석 및 트레이딩 시스템")
        self.setGeometry(100, 100, 1400, 900)
        
        # 중앙 위젯 설정
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # UI 구성
        self._create_menu_bar()
        self._create_toolbar()
        self._create_layout()
        self._create_status_bar()
        
        # 스타일 설정
        self._set_style()
        
        # 타이머 설정 (실시간 업데이트)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(1000)  # 1초마다 업데이트
    
    def _create_menu_bar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu('파일')
        
        # 설정 불러오기
        load_action = QAction('설정 불러오기', self)
        load_action.triggered.connect(self._load_settings)
        file_menu.addAction(load_action)
        
        # 설정 저장
        save_action = QAction('설정 저장', self)
        save_action.triggered.connect(self._save_settings)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # 종료
        exit_action = QAction('종료', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 데이터 메뉴
        data_menu = menubar.addMenu('데이터')
        
        refresh_action = QAction('데이터 새로고침', self)
        refresh_action.setShortcut('F5')
        refresh_action.triggered.connect(self._refresh_data)
        data_menu.addAction(refresh_action)
        
        # 전략 메뉴
        strategy_menu = menubar.addMenu('전략')
        
        strategy_settings_action = QAction('전략 설정', self)
        strategy_settings_action.triggered.connect(self._open_strategy_dialog)
        strategy_menu.addAction(strategy_settings_action)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu('도움말')
        
        about_action = QAction('정보', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """툴바 생성"""
        toolbar = self.addToolBar('메인 툴바')
        
        # 시작/중단 버튼
        self.start_button = QPushButton('시작')
        self.start_button.clicked.connect(self._start_trading)
        toolbar.addWidget(self.start_button)
        
        self.stop_button = QPushButton('중단')
        self.stop_button.clicked.connect(self._stop_trading)
        self.stop_button.setEnabled(False)
        toolbar.addWidget(self.stop_button)
        
        toolbar.addSeparator()
        
        # 종목 선택
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
        self.symbol_combo.currentTextChanged.connect(self._on_symbol_changed)
        toolbar.addWidget(QLabel('종목:'))
        toolbar.addWidget(self.symbol_combo)
        
        toolbar.addSeparator()
        
        # 설정 버튼
        settings_button = QPushButton('설정')
        settings_button.clicked.connect(self._open_settings)
        toolbar.addWidget(settings_button)
    
    def _create_layout(self):
        """레이아웃 생성"""
        # 메인 스플리터 (세로 분할)
        main_splitter = QSplitter(Qt.Vertical)
        
        # 상단 스플리터 (가로 분할)
        top_splitter = QSplitter(Qt.Horizontal)
        
        # 차트 위젯
        self.chart_widget = ChartWidget()
        top_splitter.addWidget(self.chart_widget)
        
        # 데이터 위젯
        self.data_widget = DataWidget()
        top_splitter.addWidget(self.data_widget)
        
        # 스플리터 비율 설정
        top_splitter.setSizes([800, 400])
        
        # 하단 트레이딩 위젯
        self.trading_widget = TradingWidget()
        
        # 메인 스플리터에 추가
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.trading_widget)
        main_splitter.setSizes([600, 300])
        
        # 중앙 위젯에 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(main_splitter)
        self.central_widget.setLayout(layout)
    
    def _create_status_bar(self):
        """상태바 생성"""
        self.status_bar = self.statusBar()
        
        # 상태 라벨들
        self.connection_status = QLabel("연결: 대기중")
        self.data_status = QLabel("데이터: 대기중")
        self.strategy_status = QLabel("전략: 중단")
        
        self.status_bar.addPermanentWidget(self.connection_status)
        self.status_bar.addPermanentWidget(self.data_status)
        self.status_bar.addPermanentWidget(self.strategy_status)
        
        self.status_bar.showMessage("시스템 준비 완료")
    
    def _set_style(self):
        """스타일 설정"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QMenuBar {
                background-color: #3c3c3c;
                color: #ffffff;
                border: none;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #4CAF50;
            }
            QToolBar {
                background-color: #3c3c3c;
                border: none;
                spacing: 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
            QComboBox {
                background-color: #4c4c4c;
                color: white;
                border: 1px solid #666666;
                padding: 5px;
                border-radius: 3px;
            }
            QStatusBar {
                background-color: #3c3c3c;
                color: #ffffff;
            }
        """)
    
    def _start_trading(self):
        """트레이딩 시작"""
        self.controller.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.strategy_status.setText("전략: 실행중")
        self.status_bar.showMessage("트레이딩 시작됨")
    
    def _stop_trading(self):
        """트레이딩 중단"""
        self.controller.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.strategy_status.setText("전략: 중단")
        self.status_bar.showMessage("트레이딩 중단됨")
    
    def _on_symbol_changed(self, symbol):
        """종목 변경"""
        self.chart_widget.set_symbol(symbol)
        self.data_widget.set_symbol(symbol)
        self.trading_widget.set_symbol(symbol)
    
    def _refresh_data(self):
        """데이터 새로고침"""
        current_symbol = self.symbol_combo.currentText()
        self.data_status.setText("데이터: 업데이트중...")
        # 실제 데이터 새로고침 로직
        QTimer.singleShot(2000, lambda: self.data_status.setText("데이터: 최신"))
    
    def _update_ui(self):
        """UI 실시간 업데이트"""
        # 실시간 데이터 업데이트
        current_symbol = self.symbol_combo.currentText()
        if hasattr(self.controller, 'get_realtime_data'):
            data = self.controller.get_realtime_data(current_symbol)
            if data:
                self.chart_widget.update_data(data)
                self.data_widget.update_data(data)
    
    def _load_settings(self):
        """설정 불러오기"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "설정 파일 열기", "", "JSON Files (*.json)"
        )
        if filename:
            self.controller.settings.config_file = filename
            self.controller.settings.load_config()
            self.status_bar.showMessage("설정 불러오기 완료")
    
    def _save_settings(self):
        """설정 저장"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "설정 파일 저장", "", "JSON Files (*.json)"
        )
        if filename:
            self.controller.settings.config_file = filename
            self.controller.settings.save_config()
            self.status_bar.showMessage("설정 저장 완료")
    
    def _open_settings(self):
        """설정 대화상자 열기"""
        dialog = SettingsDialog(self.controller.settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.status_bar.showMessage("설정 업데이트 완료")
    
    def _open_strategy_dialog(self):
        """전략 설정 대화상자 열기"""
        pass  # 구현 예정
    
    def _show_about(self):
        """정보 대화상자 표시"""
        QMessageBox.about(self, "정보", 
            "통합 주식 분석 및 트레이딩 시스템\n"
            "버전 1.0\n\n"
            "머신러닝과 기술적 분석을 결합한\n"
            "자동 트레이딩 시스템입니다.")
    
    def closeEvent(self, event):
        """종료 이벤트 처리"""
        reply = QMessageBox.question(self, '종료 확인', 
            '정말로 프로그램을 종료하시겠습니까?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.controller.stop()
            event.accept()
        else:
            event.ignore()