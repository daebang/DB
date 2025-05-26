# ui/widgets/chart_widget.py

import pyqtgraph as pg
from pyqtgraph import GraphicsObject, InfiniteLine, TextItem, Point # Point는 사용 안 함
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPicture, QPainter, QPen, QBrush, QColor, QFont # QFont 추가
import pandas as pd
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

# --- CandlestickItem (이전 답변의 개선된 버전 사용) ---
class CandlestickItem(GraphicsObject):
    def __init__(self, data):
        GraphicsObject.__init__(self)
        self.data = data  # data: list of (time_x, open, high, low, close)
        # self.picture = QPicture() # generatePicture에서 QPicture 생성하도록 변경
        self.generatePicture(data)

    def generatePicture(self, data):
        self.picture = QPicture() # 여기서 QPicture 객체 생성
        painter = QPainter(self.picture)
        # painter.setRenderHint(QPainter.Antialiasing) # 앤티앨리어싱 (선택적)

        if not data:
            painter.end()
            return

        # 막대 너비 계산
        bar_width_ratio = 0.7
        if len(data) > 1:
            # x축 데이터가 정렬되어 있고 숫자형(Unix timestamp)이라고 가정
            x_values = np.array([d[0] for d in data])
            # diff_x = np.diff(x_values)
            # min_dx = np.min(diff_x[diff_x > 0]) if len(diff_x[diff_x > 0]) > 0 else (24*3600) # 0보다 큰 최소 간격
            # bar_width = min_dx * bar_width_ratio
            # 더 간단하게는, 평균 간격 사용 또는 시간대에 따른 고정 비율 사용
            avg_interval = np.mean(np.diff(x_values)) if len(x_values) > 1 else (24 * 3600)
            bar_width = avg_interval * bar_width_ratio

        else: # 데이터 포인트가 하나일 경우
            bar_width = (24 * 3600) * bar_width_ratio * 0.1 # 매우 작게 표시 (임시)


        for x, o, h, l, c in data:
            # 심지
            pen_wick = pg.mkPen(color=(200, 200, 200), width=1) # 심지 펜 설정
            painter.setPen(pen_wick)
            painter.drawLine(QPointF(x, l), QPointF(x, h))

            # 몸통
            if o == c:
                pen_doji = pg.mkPen(color=(220, 220, 220), width=1)
                painter.setPen(pen_doji)
                painter.drawLine(QPointF(x - bar_width / 2, o), QPointF(x + bar_width / 2, o))
            else:
                if o > c:  # 음봉
                    color_body = QColor(0, 100, 255, 200) # 파란색 계열
                    pen_body = pg.mkPen(color=color_body.darker(120), width=1)
                else:  # 양봉
                    color_body = QColor(255, 80, 80, 200)  # 빨간색 계열
                    pen_body = pg.mkPen(color=color_body.darker(120), width=1)
                
                painter.setPen(pen_body)
                painter.setBrush(QBrush(color_body))
                painter.drawRect(QRectF(x - bar_width / 2, o, bar_width, c - o))
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if not self.data:
            return QRectF()
        min_x = min(d[0] for d in self.data)
        max_x = max(d[0] for d in self.data)
        min_y = min(d[3] for d in self.data)
        max_y = max(d[2] for d in self.data)
        
        # 약간의 여백 추가
        # x_padding = (max_x - min_x) * 0.05 if (max_x - min_x) > 0 else 1
        # y_padding = (max_y - min_y) * 0.05 if (max_y - min_y) > 0 else 1
        # return QRectF(min_x - x_padding, min_y - y_padding, (max_x - min_x) + 2*x_padding, (max_y - min_y) + 2*y_padding)
        # 정확한 바운딩 박스는 아이템의 실제 픽셀 크기를 고려해야 할 수 있음
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)


    def setData(self, data):
        self.data = data
        # self.picture = QPicture() # 여기서 초기화하면 안됨. generatePicture에서 해야 함.
        self.prepareGeometryChange()
        self.generatePicture(data)
        self.update()

class ChartWidget(QWidget):
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.symbol = "AAPL"
        self.current_data = None # 원본 OHLCV 데이터 (지표 미포함)
        self.current_data_with_ta = None # 기술적 지표 포함된 데이터
        self.candlestick_item = None
        self.price_line_item = None
        self.volume_bar_item = None

        self.sma_plot_items = {}
        self.ema_plot_items = {}
        self.bollinger_plot_items = {}

        self.v_line = None
        self.h_line = None
        self.price_text_item = None
        self.date_text_item = None

        self.setup_ui()
        self.price_plot.setTitle(f"{self.symbol} (데이터 로딩 중...)")

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)

        controls_layout = QHBoxLayout()
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(['캔들스틱', '라인'])
        self.chart_type_combo.currentTextChanged.connect(self.redraw_chart)

        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(['1일', '1주', '1개월', '1시간', '30분', '15분', '5분', '1분']) # 순서 변경
        self.timeframe_combo.currentTextChanged.connect(self.on_timeframe_or_symbol_changed)

        controls_layout.addWidget(QLabel('차트:'))
        controls_layout.addWidget(self.chart_type_combo)
        controls_layout.addWidget(QLabel('시간대:'))
        controls_layout.addWidget(self.timeframe_combo)
        controls_layout.addStretch()

        pg.setConfigOptions(antialias=True, background=pg.mkColor(30,30,30))
        self.chart_widget_layout = pg.GraphicsLayoutWidget()

        self.price_plot = self.chart_widget_layout.addPlot(row=0, col=0)
        self.price_plot.setLabel('left', '가격', units='') # 단위는 데이터에 따라 유동적일 수 있음
        self.price_plot.showGrid(x=True, y=True, alpha=0.2)
        self.price_plot.getAxis('bottom').setStyle(showValues=False)
        self.price_plot.setDownsampling(auto=True, mode='peak')
        self.price_plot.setClipToView(True)
        self.price_plot.setAutoVisible(y=True)
        self.price_plot.addLegend() # *** 범례 추가 ***

        self.chart_widget_layout.nextRow()
        self.volume_plot = self.chart_widget_layout.addPlot(row=1, col=0)
        self.volume_plot.setLabel('left', '거래량')
        self.volume_plot.showGrid(x=True, y=True, alpha=0.2)
        self.volume_plot.setMaximumHeight(120)
        self.volume_plot.setDownsampling(auto=True, mode='peak')
        self.volume_plot.setClipToView(True)

        self.price_plot.setXLink(self.volume_plot)

        self.date_axis = pg.DateAxisItem(orientation='bottom')
        self.volume_plot.setAxisItems({'bottom': self.date_axis})

        pen_crosshair = pg.mkPen(color=(180, 180, 200, 150), style=Qt.DashLine, width=1) # 투명도 추가
        self.v_line = InfiniteLine(angle=90, movable=False, pen=pen_crosshair)
        self.h_line = InfiniteLine(angle=0, movable=False, pen=pen_crosshair)
        
        self.price_text_item = TextItem(anchor=(0,1), fill=(40,40,40,200), border=pg.mkPen(color=(150,150,150,150),width=1))
        self.price_text_item.setZValue(10)
        
        # date_text_item 초기화 (AttributeError 방지)
        # DateAxisItem이 X축 레이블을 잘 보여주므로, 별도의 date_text_item은 크로스헤어 정보 표시에 불필요할 수 있음.
        # 만약 특정 위치의 거래량 등을 표시하고 싶다면 volume_plot에 추가하고 활용.
        self.date_text_item = TextItem(anchor=(0,0), color=(200,200,200)) # 예시: 볼륨 플롯용 정보
        # self.volume_plot.addItem(self.date_text_item) # 필요 시 추가
        # self.date_text_item.hide()

        self.price_plot.addItem(self.v_line, ignoreBounds=True)
        self.price_plot.addItem(self.h_line, ignoreBounds=True)
        self.price_plot.addItem(self.price_text_item)

        self.v_line.hide()
        self.h_line.hide()
        self.price_text_item.hide()

        self.proxy = pg.SignalProxy(self.price_plot.scene().sigMouseMoved, rateLimit=30, slot=self.on_mouse_moved_on_price_plot) # rateLimit 조정

        layout.addLayout(controls_layout)
        layout.addWidget(self.chart_widget_layout)
        self.setLayout(layout)

    # ... (init_dummy_data는 실제 데이터 로딩으로 대체) ...
    def on_timeframe_or_symbol_changed(self):
        current_symbol = self.symbol
        current_timeframe = self.timeframe_combo.currentText()
        logger.info(f"차트 데이터 요청: 심볼={current_symbol}, 시간대={current_timeframe}")
        self.price_plot.setTitle(f"{current_symbol} ({current_timeframe}) - 데이터 로딩 중...")
        self.clear_chart_items() # 요청 전에 기존 차트 클리어

        if self.controller and hasattr(self.controller, 'request_historical_data'):
            self.controller.request_historical_data(current_symbol, current_timeframe)
        else:
            logger.warning("컨트롤러가 없거나 데이터 요청 메서드가 없어 차트 데이터 로드 불가.")
            # self.init_dummy_data() # 더미 데이터 로드 (테스트용)


    def set_symbol(self, symbol: str):
        if self.symbol != symbol:
            self.symbol = symbol
            logger.info(f"차트 위젯: 심볼 변경됨 - {self.symbol}")
            self.on_timeframe_or_symbol_changed()

    def update_historical_data(self, data_df: pd.DataFrame):
        if data_df is None or data_df.empty:
            logger.warning(f"{self.symbol}에 대한 과거 데이터가 비어있습니다.")
            self.current_data = None # 원본 데이터 저장
            self.current_data_with_ta = None # TA 데이터도 클리어
            self.clear_chart_items(clear_indicators=True) # 지표도 함께 클리어
            self.price_plot.setTitle(f"{self.symbol} ({self.timeframe_combo.currentText()}) - 데이터 없음")
            self.hide_crosshair()
            return

        logger.debug(f"{self.symbol} 과거 데이터 수신 ({len(data_df)}개), 차트 업데이트 시작 (원본 데이터)")
        self.current_data = data_df.copy() # 원본 데이터 저장
        
        if 'timestamp' not in self.current_data.columns:
            logger.error("수신된 과거 데이터에 'timestamp' 컬럼이 없습니다.")
            self.current_data = None
            self.clear_chart_items()
            self.price_plot.setTitle(f"{self.symbol} ({self.timeframe_combo.currentText()}) - 시간 데이터 오류")
            return

        if not pd.api.types.is_numeric_dtype(self.current_data['timestamp']):
            try:
                self.current_data['timestamp'] = pd.to_datetime(self.current_data['timestamp']).astype(np.int64) // 10**9
            except Exception as e:
                logger.error(f"차트용 timestamp 변환 실패: {e}. current_data['timestamp']는 숫자형 Unix timestamp여야 합니다.")
                self.current_data = None
                self.clear_chart_items()
                self.price_plot.setTitle(f"{self.symbol} ({self.timeframe_combo.currentText()}) - 시간 데이터 변환 오류")
                return
        
        self.current_data = self.current_data.sort_values(by='timestamp')
        self._redraw_base_chart() # 기본 차트(캔들/라인, 거래량) 먼저 그림

        # current_data_with_ta가 이미 있다면, 그것으로 지표를 다시 그림
        if self.current_data_with_ta is not None and not self.current_data_with_ta.empty:
            # 현재 심볼과 시간대가 일치하는지 확인 후 TA 업데이트
            current_symbol_in_ta = self.current_data_with_ta.attrs.get('symbol')
            current_timeframe_in_ta = self.current_data_with_ta.attrs.get('timeframe')
            if current_symbol_in_ta == self.symbol and current_timeframe_in_ta == self.timeframe_combo.currentText():
                 self.update_technical_indicators_on_chart(self.current_data_with_ta)
            else: # 일치하지 않으면 TA 데이터도 클리어하고 새로 받아야 함
                self.current_data_with_ta = None
                self.clear_indicator_plots()


        # X, Y축 범위 설정
        if not self.current_data.empty:
            min_ts = self.current_data['timestamp'].min()
            max_ts = self.current_data['timestamp'].max()
            self.price_plot.setXRange(min_ts, max_ts, padding=0.02) 
            self.volume_plot.setXRange(min_ts, max_ts, padding=0.02)
            # Y축은 지표 추가 후 다시 autoRange 할 수 있음
            self.price_plot.enableAutoRange(axis='y') 
            self.volume_plot.enableAutoRange(axis='y')
        else:
            self.price_plot.autoRange()
            self.volume_plot.autoRange()


    def _redraw_base_chart(self):
        """캔들스틱/라인, 거래량 등 기본 차트 항목을 그립니다."""
        self.clear_chart_items(clear_indicators=False) # 기본 차트 아이템만 클리어, 지표는 유지할 수도 있음 (선택)
                                                    # 또는 clear_indicators=True로 하고 지표도 다시 그림
        if self.current_data is None or self.current_data.empty:
            self.price_plot.setTitle(f"{self.symbol} ({self.timeframe_combo.currentText()}) - 데이터 없음")
            return

        x_timestamps = self.current_data['timestamp'].values
        current_chart_type = self.chart_type_combo.currentText()
        timeframe_str = self.timeframe_combo.currentText()

        # 거래량 바 너비 계산
        bar_width_factor = 0.7 
        if len(x_timestamps) > 1:
            avg_interval = np.mean(np.diff(x_timestamps)) 
            bar_width = avg_interval * bar_width_factor
        elif len(x_timestamps) == 1: 
            if '분' in timeframe_str: bar_width = int(timeframe_str.replace('분','')) * 60 * bar_width_factor
            elif '시간' in timeframe_str: bar_width = int(timeframe_str.replace('시간','')) * 3600 * bar_width_factor
            else: bar_width = (24*3600) * bar_width_factor 
        else: 
            bar_width = 1 

        self.volume_bar_item = pg.BarGraphItem(x=x_timestamps, height=self.current_data['volume'].values, width=bar_width, brush=(80,80,150,180), pen=pg.mkPen(None))
        self.volume_plot.addItem(self.volume_bar_item)


        if current_chart_type == '라인':
            self.price_line_item = self.price_plot.plot(x_timestamps, self.current_data['close'].values, pen=pg.mkPen('c', width=2), name="종가")
        elif current_chart_type == '캔들스틱':
            candlestick_data_tuples = []
            # yfinance는 Open, High, Low, Close 컬럼명을 사용하므로, 이전 코드의 ohlc 컬럼명 변환 확인
            # self.current_data 에 이미 open, high, low, close (소문자) 컬럼이 있다고 가정
            for _idx, row in self.current_data.iterrows():
                if all(col in row and pd.notnull(row[col]) for col in ['timestamp', 'open', 'high', 'low', 'close']):
                    candlestick_data_tuples.append((row['timestamp'], row['open'], row['high'], row['low'], row['close']))
            
            if candlestick_data_tuples:
                if self.candlestick_item is None:
                    self.candlestick_item = CandlestickItem(candlestick_data_tuples) # CandlestickItem 정의 필요
                    self.price_plot.addItem(self.candlestick_item)
                else:
                    self.candlestick_item.setData(candlestick_data_tuples)
        
        self.price_plot.setTitle(f"{self.symbol} - {timeframe_str} ({current_chart_type})")

    def redraw_chart(self): # 사용자가 차트 타입 변경 시 호출
        self._redraw_base_chart()
        if self.current_data_with_ta is not None and not self.current_data_with_ta.empty:
            self.update_technical_indicators_on_chart(self.current_data_with_ta)
        # X축 범위 재설정
        if self.current_data is not None and not self.current_data.empty:
            min_ts = self.current_data['timestamp'].min()
            max_ts = self.current_data['timestamp'].max()
            self.price_plot.setXRange(min_ts, max_ts, padding=0.02)
            self.price_plot.enableAutoRange(axis='y') # Y축은 지표 포함하여 자동 조정


    def update_technical_indicators_on_chart(self, data_with_indicators: pd.DataFrame):
        self.current_data_with_ta = data_with_indicators.copy() # TA 데이터 캐싱
        # ... (이전 답변의 update_technical_indicators_on_chart 로직과 거의 동일)
        # 각 plot 아이템에 name 인자 확실히 전달 (범례용)
        if data_with_indicators is None or data_with_indicators.empty or 'timestamp' not in data_with_indicators.columns:
            logger.warning("차트에 표시할 기술적 지표 데이터가 부적절합니다.")
            self.clear_indicator_plots() # 기존 지표 플롯 제거
            return

        logger.debug(f"차트에 기술적 지표 업데이트 시작 (데이터 {len(data_with_indicators)}개)")
        self.clear_indicator_plots() # 기존 지표 플롯 먼저 제거

        x_timestamps = data_with_indicators['timestamp'].values

        # SMA 플롯
        sma_cols = [col for col in data_with_indicators.columns if col.startswith('SMA_')]
        sma_pens = [pg.mkPen(color, width=1) for color in ['#FFA500', '#FFC0CB', '#DA70D6']] # 주황, 핑크, 보라
        for i, col in enumerate(sma_cols):
            if col in data_with_indicators and pd.api.types.is_numeric_dtype(data_with_indicators[col]):
                self.sma_plot_items[col] = self.price_plot.plot(x_timestamps, data_with_indicators[col].values, pen=sma_pens[i % len(sma_pens)], name=col) # name 추가
                logger.debug(f"{col} 플롯 추가됨")

        # EMA 플롯
        ema_cols = [col for col in data_with_indicators.columns if col.startswith('EMA_')]
        ema_pens = [pg.mkPen(color, width=1) for color in ['#ADD8E6', '#90EE90', '#87CEFA']] # 연파랑, 연초록, 하늘색
        for i, col in enumerate(ema_cols):
            if col in data_with_indicators and pd.api.types.is_numeric_dtype(data_with_indicators[col]):
                self.ema_plot_items[col] = self.price_plot.plot(x_timestamps, data_with_indicators[col].values, pen=ema_pens[i % len(ema_pens)], name=col) # name 추가
                logger.debug(f"{col} 플롯 추가됨")

        # 볼린저 밴드 플롯
        bb_middle_col = next((col for col in data_with_indicators.columns if col.startswith('BB_Middle_')), None)
        if bb_middle_col:
            # window_str = bb_middle_col.split('_')[-1] # 컬럼명에서 window 추출
            # 더 안전한 방법: BB_Middle_20 -> 20 추출
            try:
                window_str = re.search(r'BB_Middle_(\d+)', bb_middle_col).group(1)
            except AttributeError:
                logger.warning(f"볼린저 밴드 기간 추출 실패: {bb_middle_col}")
                window_str = "Default" # 또는 오류 처리

            bb_upper_col = f'BB_Upper_{window_str}'
            bb_lower_col = f'BB_Lower_{window_str}'

            if all(col in data_with_indicators and pd.api.types.is_numeric_dtype(data_with_indicators[col]) for col in [bb_upper_col, bb_middle_col, bb_lower_col]):
                pen_bb_middle = pg.mkPen('#A9A9A9', width=1, style=Qt.DotLine) # 중간선: 회색 점선
                pen_bb_outer = pg.mkPen('#A9A9A9', width=1)      # 상하단선: 회색 실선
                
                self.bollinger_plot_items[bb_middle_col] = self.price_plot.plot(x_timestamps, data_with_indicators[bb_middle_col].values, pen=pen_bb_middle, name=bb_middle_col)
                self.bollinger_plot_items[bb_upper_col] = self.price_plot.plot(x_timestamps, data_with_indicators[bb_upper_col].values, pen=pen_bb_outer, name=bb_upper_col)
                self.bollinger_plot_items[bb_lower_col] = self.price_plot.plot(x_timestamps, data_with_indicators[bb_lower_col].values, pen=pen_bb_outer, name=bb_lower_col)
                
                logger.debug(f"볼린저 밴드 ({window_str}) 플롯 추가됨")
        
        # Y축 범위 재조정 (지표 포함)
        self.price_plot.enableAutoRange(axis='y', enable=True)


    def clear_chart_items(self, clear_indicators: bool = True): #
        if self.candlestick_item: self.price_plot.removeItem(self.candlestick_item); self.candlestick_item = None
        if self.price_line_item: self.price_plot.removeItem(self.price_line_item); self.price_line_item = None
        if self.volume_bar_item: self.volume_plot.removeItem(self.volume_bar_item); self.volume_bar_item = None
        
        if clear_indicators:
            self.clear_indicator_plots()

    def clear_indicator_plots(self):
        """차트에서 모든 기술적 지표 플롯 아이템을 제거합니다."""
        for item_dict in [self.sma_plot_items, self.ema_plot_items, self.bollinger_plot_items]:
            for key, plot_item in list(item_dict.items()): # list()로 복사하여 반복 중 삭제 오류 방지
                if plot_item:
                    try:
                        self.price_plot.removeItem(plot_item)
                    except Exception as e:
                        logger.warning(f"차트에서 {key} 지표 아이템 제거 중 오류: {e}")
                item_dict[key] = None # 또는 del item_dict[key]
            item_dict.clear()
        # logger.debug("모든 기술적 지표 플롯 제거됨")
    def on_mouse_moved_on_price_plot(self, pos_arg_from_signal): # 인자 이름 변경
        """가격 차트 위에서 마우스가 움직일 때 호출됩니다. pos_arg_from_signal은 (QPointF,) 형태의 튜플입니다."""
        if self.current_data is None or self.current_data.empty:
            self.hide_crosshair()
            return

        # pos_arg_from_signal이 튜플이고, 그 첫번째 요소가 QPointF인지 확인
        if not isinstance(pos_arg_from_signal, tuple) or not pos_arg_from_signal:
            logger.warning(f"on_mouse_moved_on_price_plot: 예상치 못한 인자 타입 또는 빈 튜플: {pos_arg_from_signal}")
            self.hide_crosshair()
            return
        
        scene_pos_qpointf = pos_arg_from_signal[0] # 튜플의 첫 번째 요소가 QPointF
        if not isinstance(scene_pos_qpointf, QPointF):
            logger.warning(f"on_mouse_moved_on_price_plot: 튜플의 요소가 QPointF가 아님: {scene_pos_qpointf}")
            self.hide_crosshair()
            return

        scene_rect = self.price_plot.sceneBoundingRect()
        if not scene_rect.contains(scene_pos_qpointf):
            self.hide_crosshair()
            return

        vb = self.price_plot.getViewBox()
        mouse_point_in_view = vb.mapSceneToView(scene_pos_qpointf)
        mouse_x, mouse_y = mouse_point_in_view.x(), mouse_point_in_view.y()

        self.v_line.setPos(mouse_x)
        self.h_line.setPos(mouse_y)
        self.v_line.show()
        self.h_line.show()

        if 'timestamp' not in self.current_data.columns:
            self.hide_crosshair()
            return
            
        x_data = self.current_data['timestamp'].values
        if len(x_data) == 0:
            self.hide_crosshair()
            return

        # X축 범위 벗어났는지 체크 (mouse_x는 view 좌표계의 x값)
        view_x_min, view_x_max = vb.viewRange()[0]
        if not (view_x_min <= mouse_x <= view_x_max): # mouse_x가 현재 보이는 x축 범위 내에 있는지
             self.price_text_item.hide()
             # self.hide_crosshair() # 크로스 라인은 유지하고 텍스트만 숨길 수도 있음
             return

        index = np.abs(x_data - mouse_x).argmin()

        if not (0 <= index < len(self.current_data)):
            self.hide_crosshair()
            return

        data_point = self.current_data.iloc[index]
        timeframe = self.timeframe_combo.currentText()
        
        dt_object = pd.to_datetime(data_point['timestamp'], unit='s')
        if '일' in timeframe or '주' in timeframe or '월' in timeframe:
            date_str = dt_object.strftime('%Y-%m-%d')
        else: # 분봉, 시간봉
            date_str = dt_object.strftime('%Y-%m-%d %H:%M')


        o = data_point.get('open', np.nan) # get으로 안전하게 접근
        h = data_point.get('high', np.nan)
        l = data_point.get('low', np.nan)
        c = data_point.get('close', np.nan)
        v = data_point.get('volume', np.nan)

        # HTML 텍스트 생성 (값이 NaN일 경우 '-' 표시)
        def format_val(val, precision=2, is_volume=False):
            if pd.isna(val): return "-"
            return f"{val:,.{precision}f}" if not is_volume else f"{val:,.0f}"

        html_text = f"""<div style='font-family: Consolas, "Courier New", monospace; font-size: 9pt; color: #E0E0E0; background-color:rgba(30,30,30,220); padding: 5px 8px; border-radius: 3px; border: 1px solid #606060;'>
                        <strong style='color:#FFFFFF;'>{date_str}</strong><br>
                        O: <span style='color:#FFD700;'>{format_val(o)}</span> H: <span style='color:#80FF80;'>{format_val(h)}</span><br>
                        L: <span style='color:#FF8080;'>{format_val(l)}</span> C: <span style='color:#80D0FF;'>{format_val(c)}</span><br>
                        Vol: <span style='color:#C0C0C0;'>{format_val(v, is_volume=True)}</span><br>
                        Y(가격): <span style='color:#C0C0C0;'>{mouse_y:.2f}</span>
                        </div>"""
        self.price_text_item.setHtml(html_text)
        
        # TextItem 위치 조정
        view_range = vb.viewRange()
        text_anchor_x = mouse_x
        text_anchor_y = mouse_y

        # TextItem의 예상 너비/높이 (대략적인 값, 실제로는 그려봐야 알 수 있음)
        est_text_width_pixels = 150 # 예시
        est_text_height_pixels = 80 # 예시
        # 픽셀을 뷰 좌표계 단위로 변환 (대략적)
        pixel_width_in_view = vb.pixelWidth()
        pixel_height_in_view = vb.pixelHeight()
        est_text_width_view = est_text_width_pixels * pixel_width_in_view
        # est_text_height_view = est_text_height_pixels * pixel_height_in_view


        # 마우스 오른쪽으로 일정 간격만큼 이동시켜 표시
        x_offset = (view_range[0][1] - view_range[0][0]) * 0.01 # 뷰 너비의 1%

        if (text_anchor_x + x_offset + est_text_width_view) > view_range[0][1]: # 오른쪽 경계 침범 시
            self.price_text_item.setAnchor((1,1)) # 오른쪽 상단 기준
            text_anchor_x = mouse_x - x_offset
        else:
            self.price_text_item.setAnchor((0,1)) # 왼쪽 상단 기준
            text_anchor_x = mouse_x + x_offset
        
        self.price_text_item.setPos(text_anchor_x, text_anchor_y)
        self.price_text_item.show()


    def hide_crosshair(self, event=None):
        if self.v_line: self.v_line.hide()
        if self.h_line: self.h_line.hide()
        if self.price_text_item: self.price_text_item.hide()
        # date_text_item은 현재 기본적으로 사용되지 않으므로, 초기화되었다면 숨김 처리
        if hasattr(self, 'date_text_item') and self.date_text_item:
             self.date_text_item.hide()

    # ... (나머지 코드)