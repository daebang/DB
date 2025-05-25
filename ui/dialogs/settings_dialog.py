from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class SettingsDialog(QDialog):
    """설정 대화상자"""
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("설정")
        self.setModal(True)
        self.setup_ui()
    
    def setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout()
        
        # 탭 위젯
        tab_widget = QTabWidget()
        
        # API 설정 탭
        api_tab = self.create_api_tab()
        tab_widget.addTab(api_tab, "API 설정")
        
        # 데이터베이스 설정 탭
        db_tab = self.create_database_tab()
        tab_widget.addTab(db_tab, "데이터베이스")
        
        # 트레이딩 설정 탭
        trading_tab = self.create_trading_tab()
        tab_widget.addTab(trading_tab, "트레이딩")
        
        # 버튼
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("확인")
        ok_button.clicked.connect(self.accept)
        
        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        layout.addWidget(tab_widget)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.resize(500, 400)
    
    def create_api_tab(self):
        """API 설정 탭"""
        widget = QWidget()
        layout = QFormLayout()
        
        self.alpha_vantage_key_edit = QLineEdit(self.settings.api_config.alpha_vantage_key)
        self.news_api_key_edit = QLineEdit(self.settings.api_config.news_api_key)
        self.broker_api_key_edit = QLineEdit(self.settings.api_config.broker_api_key)
        self.broker_secret_edit = QLineEdit(self.settings.api_config.broker_secret)
        self.broker_secret_edit.setEchoMode(QLineEdit.Password)
        
        layout.addRow("Alpha Vantage API Key:", self.alpha_vantage_key_edit)
        layout.addRow("News API Key:", self.news_api_key_edit)
        layout.addRow("Broker API Key:", self.broker_api_key_edit)
        layout.addRow("Broker Secret:", self.broker_secret_edit)
        
        widget.setLayout(layout)
        return widget
    
    def create_database_tab(self):
        """데이터베이스 설정 탭"""
        widget = QWidget()
        layout = QFormLayout()
        
        self.db_host_edit = QLineEdit(self.settings.db_config.host)
        self.db_port_spin = QSpinBox()
        self.db_port_spin.setRange(1, 65535)
        self.db_port_spin.setValue(self.settings.db_config.port)
        self.db_name_edit = QLineEdit(self.settings.db_config.database)
        self.db_user_edit = QLineEdit(self.settings.db_config.username)
        self.db_password_edit = QLineEdit(self.settings.db_config.password)
        self.db_password_edit.setEchoMode(QLineEdit.Password)
        
        layout.addRow("Host:", self.db_host_edit)
        layout.addRow("Port:", self.db_port_spin)
        layout.addRow("Database:", self.db_name_edit)
        layout.addRow("Username:", self.db_user_edit)
        layout.addRow("Password:", self.db_password_edit)
        
        widget.setLayout(layout)
        return widget
    
    def create_trading_tab(self):
        """트레이딩 설정 탭"""
        widget = QWidget()
        layout = QFormLayout()
        
        self.max_position_spin = QDoubleSpinBox()
        self.max_position_spin.setRange(0.01, 1.0)
        self.max_position_spin.setSingleStep(0.01)
        self.max_position_spin.setValue(self.settings.trading_config.max_position_size)
        self.max_position_spin.setSuffix("%")
        
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(0.01, 0.5)
        self.stop_loss_spin.setSingleStep(0.01)
        self.stop_loss_spin.setValue(self.settings.trading_config.stop_loss_percent)
        self.stop_loss_spin.setSuffix("%")
        
        self.take_profit_spin = QDoubleSpinBox()
        self.take_profit_spin.setRange(0.01, 1.0)
        self.take_profit_spin.setSingleStep(0.01)
        self.take_profit_spin.setValue(self.settings.trading_config.take_profit_percent)
        self.take_profit_spin.setSuffix("%")
        
        self.risk_free_rate_spin = QDoubleSpinBox()
        self.risk_free_rate_spin.setRange(0.0, 0.1)
        self.risk_free_rate_spin.setSingleStep(0.001)
        self.risk_free_rate_spin.setValue(self.settings.trading_config.risk_free_rate)
        self.risk_free_rate_spin.setSuffix("%")
        
        layout.addRow("최대 포지션 크기:", self.max_position_spin)
        layout.addRow("손절 비율:", self.stop_loss_spin)
        layout.addRow("익절 비율:", self.take_profit_spin)
        layout.addRow("무위험 수익률:", self.risk_free_rate_spin)
        
        widget.setLayout(layout)
        return widget
    
    def accept(self):
        """설정 저장"""
        # API 설정 업데이트
        self.settings.api_config.alpha_vantage_key = self.alpha_vantage_key_edit.text()
        self.settings.api_config.news_api_key = self.news_api_key_edit.text()
        self.settings.api_config.broker_api_key = self.broker_api_key_edit.text()
        self.settings.api_config.broker_secret = self.broker_secret_edit.text()
        
        # 데이터베이스 설정 업데이트
        self.settings.db_config.host = self.db_host_edit.text()
        self.settings.db_config.port = self.db_port_spin.value()
        self.settings.db_config.database = self.db_name_edit.text()
        self.settings.db_config.username = self.db_user_edit.text()
        self.settings.db_config.password = self.db_password_edit.text()
        
        # 트레이딩 설정 업데이트
        self.settings.trading_config.max_position_size = self.max_position_spin.value()
        self.settings.trading_config.stop_loss_percent = self.stop_loss_spin.value()
        self.settings.trading_config.take_profit_percent = self.take_profit_spin.value()
        self.settings.trading_config.risk_free_rate = self.risk_free_rate_spin.value()
        
        # 설정 저장
        self.settings.save_config()
        
        super().accept()