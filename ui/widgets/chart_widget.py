import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
import numpy as np

class ChartWidget(QWidget):
    """차트 위젯"""
    
    def __init__(self):
        super().__init__()
        self.symbol = "AAPL"
        self.setup_ui()
        
    def setup_ui(self):
        """UI 설정"""
        layout = QVBoxLayout()
        
        # 차트 타입 선택
        controls_layout = QHBoxLayout()
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(['캔들스틱', '라인', '볼륨'])
        self.chart_type_combo.currentTextChanged.connect(self.update_chart_type)
        
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1분', '5분', '1시간', '1일'])
        
        controls_layout.addWidget(QLabel('차트 타입:'))
        controls_layout.addWidget(self.chart_type_combo)
        controls_layout.addWidget(QLabel('시간대:'))
        controls_layout.addWidget(self.timeframe_combo)
        controls_layout.addStretch()
        
        # 차트 위젯
        self.chart_widget = pg.GraphicsLayoutWidget()
        self.price_plot = self.chart_widget.addPlot(title="주가 차트")
        self.volume_plot = self.chart_widget.addPlot(title="거래량", row=1, col=0)
        
        # 스타일 설정
        self.price_plot.setLabel('left', '가격 ($)')
        self.price_plot.setLabel('bottom', '시간')
        self.volume_plot.setLabel('left', '거래량')
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.chart_widget)
        self.setLayout(layout)
        
        # 더미 데이터로 초기화
        self.init_dummy_data()
    
    def init_dummy_data(self):
        """더미 데이터로 초기화"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 150 + np.cumsum(np.random.randn(100) * 0.5)
        volumes = np.random.randint(1000000, 5000000, 100)
        
        self.data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })
        
        self.update_chart()
    
    def set_symbol(self, symbol):
        """종목 설정"""
        self.symbol = symbol
        self.price_plot.setTitle(f"{symbol} 주가 차트")
    
    def update_data(self, new_data):
        """데이터 업데이트"""
        # 실제 구현에서는 새로운 데이터를 받아서 차트 업데이트
        pass
    
    def update_chart(self):
        """차트 업데이트"""
        if self.data is not None and not self.data.empty:
            # 가격 차트
            self.price_plot.clear()
            x = np.arange(len(self.data))
            self.price_plot.plot(x, self.data['close'], pen='w', width=2)
            
            # 거래량 차트
            self.volume_plot.clear()
            self.volume_plot.plot(x, self.data['volume'], pen='b', fillLevel=0, brush=(0, 0, 255, 100))
    
    def update_chart_type(self, chart_type):
        """차트 타입 변경"""
        self.update_chart()