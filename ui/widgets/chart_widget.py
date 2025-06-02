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

class CustomDateAxisItem(pg.AxisItem):
    """X축에 실제 날짜/시간을 표시하는 커스텀 축"""
    def __init__(self, timestamps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamps = timestamps
        
    def tickStrings(self, values, scale, spacing):
        strings = []
        if not hasattr(self, 'timestamps') or self.timestamps is None or len(self.timestamps) == 0:
            # logger.warning("CustomDateAxisItem: Timestamps for ticks are not available.") # 필요시 로깅
            return [str(int(v)) for v in values] # 타임스탬프 없으면 그냥 인덱스 반환 (임시)

        for v_float in values: # pyqtgraph는 float 값을 전달할 수 있음
            try:
                # 가장 가까운 정수 인덱스로 변환 (또는 바닥/올림 처리)
                # v_float는 플롯에 표시될 위치의 인덱스이므로, 정수여야 함
                idx = int(round(v_float)) # round 추가하여 가장 가까운 인덱스 사용

                if 0 <= idx < len(self.timestamps):
                    # self.timestamps[idx]가 유효한 Unix timestamp인지 확인
                    ts_value = self.timestamps[idx]
                    if pd.isna(ts_value):
                        # logger.debug(f"Tick formatting: Timestamp at index {idx} is NaN.") # 필요시 로깅
                        strings.append('') # NaN 타임스탬프는 빈 문자열
                        continue

                    dt = pd.to_datetime(ts_value, unit='s', errors='coerce')
                    if pd.NaT == dt: # 변환 실패 시
                        # logger.warning(f"Tick formatting: Failed to convert timestamp {ts_value} at index {idx} to datetime.") # 필요시 로깅
                        strings.append(str(idx)) # 변환 실패 시 인덱스 표시
                        continue

                    # 시간 간격(spacing)에 따라 포맷 결정
                    # spacing은 X축 값(여기서는 인덱스)의 간격임. 실제 시간 간격으로 변환하려면 추가 계산 필요.
                    # 여기서는 간단히 values의 범위를 보고 결정하거나, 고정된 규칙 사용
                    if spacing <= 1 and (self.timestamps.max() - self.timestamps.min()) <= 3*24*3600 : # 대략 3일 이내 데이터는 시간까지
                        strings.append(dt.strftime('%H:%M'))
                    elif spacing <= 7 and (self.timestamps.max() - self.timestamps.min()) <= 30*24*3600: # 대략 한달 이내 데이터는 날짜/시간
                        strings.append(dt.strftime('%m/%d %H:%M'))
                    elif spacing <= 30 : # 월 단위 데이터는 날짜만
                         strings.append(dt.strftime('%m-%d'))
                    else: # 그 외는 년-월
                        strings.append(dt.strftime('%Y-%m'))
                else:
                    # logger.debug(f"Tick formatting: Index {idx} (from value {v_float:.2f}) is out of bounds for timestamps array (len {len(self.timestamps)}).") # 필요시 로깅
                    strings.append('') # 범위 벗어나면 빈 문자열
            except ValueError: # int(v_float) 등에서 발생 가능
                # logger.error(f"Tick formatting: ValueError for value {v_float}.", exc_info=True) # 필요시 로깅
                strings.append('')
            except Exception as e: # 그 외 예외
                # logger.error(f"Tick formatting: Unexpected error for value {v_float}: {e}", exc_info=True) # 필요시 로깅
                strings.append('')
        return strings

        

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
        bar_width_ratio = 0.7 # 막대 너비 비율 (인덱스 간격 대비)

        if not data:
            painter.end()
            return

        # x_values는 이제 (0, 1, 2, ...) 형태의 인덱스임
        # 따라서 인덱스 간의 평균 간격은 1.0이 됨
        # 그러므로 bar_width는 bar_width_ratio와 거의 동일하게 됨 (예: 0.7)
        # 이는 인덱스 기반 X축에서 적절한 상대적 너비임
        bar_width = bar_width_ratio # 데이터가 여러 개일 때, 인덱스 간격은 1이므로

        if len(data) == 1: # 데이터 포인트가 하나일 경우에도 적절한 너비
            bar_width = bar_width_ratio * 0.5 # 예: 기본 너비의 절반으로 표시

        # 만약 x 좌표가 연속적이지 않은 인덱스일 가능성이 있다면 (예: 필터링된 데이터),
        # 실제 x 값 (d[0])의 차이를 보고 너비를 동적으로 계산해야 하지만,
        # 현재는 enumerate로 생성된 연속 idx를 사용하므로 위 방식이 유효
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
        self.timeframe_combo.addItems(['1일', '1주', '1개월']) # Tiingo는 일봉 데이터만 지원
        self.timeframe_combo.currentTextChanged.connect(self.on_timeframe_or_symbol_changed)

        controls_layout.addWidget(QLabel('차트:'))
        controls_layout.addWidget(self.chart_type_combo)
        controls_layout.addWidget(QLabel('시간대:'))
        controls_layout.addWidget(self.timeframe_combo)
        controls_layout.addStretch()

        pg.setConfigOptions(antialias=True, background=pg.mkColor(30,30,30))
        self.chart_widget_layout = pg.GraphicsLayoutWidget()

        self.price_plot = self.chart_widget_layout.addPlot(row=0, col=0)
        self.price_plot.setLabel('left', '가격', units='')
        self.price_plot.showGrid(x=True, y=True, alpha=0.2)
        self.price_plot.getAxis('bottom').setStyle(showValues=False)
        self.price_plot.setDownsampling(auto=True, mode='peak')
        self.price_plot.setClipToView(True)
        self.price_plot.setAutoVisible(y=True)
        
        # 범례 추가 및 설정
        self.legend = self.price_plot.addLegend(offset=(10, 10))  # 오프셋 조정
        self.legend.setParentItem(self.price_plot.vb)  # ViewBox에 부착
        self.legend.anchor((0, 0), (0, 0))  # 왼쪽 상단에 고정
        self.legend.setBrush(pg.mkBrush(40, 40, 40, 200))  # 반투명 배경
        self.legend.setPen(pg.mkPen(150, 150, 150, 150))   # 테두리
        self.legend.setLabelTextColor(pg.mkColor(224, 224, 224))  # 텍스트 색상

        self.chart_widget_layout.nextRow()
        self.volume_plot = self.chart_widget_layout.addPlot(row=1, col=0)
        self.volume_plot.setLabel('left', '거래량')
        self.volume_plot.showGrid(x=True, y=True, alpha=0.2)
        self.volume_plot.setMaximumHeight(120)
        self.volume_plot.setDownsampling(auto=True, mode='peak')
        self.volume_plot.setClipToView(True)

        self.price_plot.setXLink(self.volume_plot)


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

        if data_df is None or data_df.empty:
            logger.warning(f"{self.symbol}에 대한 과거 데이터가 비어있습니다. (ChartWidget)")
            # ... (existing empty data handling) ...
            return

        if data_df is None or data_df.empty:
            logger.warning(f"{self.symbol}에 대한 과거 데이터가 비어있습니다. (ChartWidget)")
            self.current_data = None 
            self.current_data_with_ta = None 
            self.clear_chart_items(clear_indicators=True) 
            self.price_plot.setTitle(f"{self.symbol} ({self.timeframe_combo.currentText()}) - 데이터 없음")
            self.hide_crosshair()
            return

        logger.debug(f"{self.symbol} 과거 데이터 수신 ({len(data_df)}개), 차트 업데이트 시작 (원본 데이터) (ChartWidget)")
        processed_df = data_df.copy()

        # Ensure processed_df has a DatetimeIndex
        if isinstance(processed_df.index, pd.DatetimeIndex):
            if processed_df.index.tz is not None:
                processed_df.index = processed_df.index.tz_convert(None)
        elif 'timestamp' in processed_df.columns:
            if pd.api.types.is_datetime64_any_dtype(processed_df['timestamp']):
                logger.debug(f"'{self.symbol}' 차트 데이터: 'timestamp' 컬럼이 이미 datetime64 타입입니다. 인덱스로 설정합니다.")
                processed_df.set_index(pd.to_datetime(processed_df['timestamp']), inplace=True)
            elif pd.api.types.is_numeric_dtype(processed_df['timestamp']):
                logger.debug(f"'{self.symbol}' 차트 데이터: 'timestamp' 컬럼이 숫자형입니다. Unix 초로 간주하여 DatetimeIndex로 변환합니다.")
                # Set the index. The original 'timestamp' column is consumed. The new index might be named 'timestamp'.
                processed_df.set_index(pd.to_datetime(processed_df['timestamp'], unit='s', errors='coerce'), inplace=True)
            else:
                logger.warning(f"'{self.symbol}' 차트 데이터: 'timestamp' 컬럼이 알 수 없는 타입입니다. 일반 변환 시도.")
                processed_df.set_index(pd.to_datetime(processed_df['timestamp'], errors='coerce'), inplace=True)
            
            if processed_df.index.hasnans:
                logger.warning(f"'{self.symbol}' 차트 데이터: timestamp 변환 후 NaT 발생. 해당 행 제거.")
                processed_df = processed_df[~processed_df.index.isna()]

        elif 'Date' in processed_df.columns and pd.api.types.is_datetime64_any_dtype(processed_df['Date']):
            logger.debug(f"'{self.symbol}' 차트 데이터: 'Date' 컬럼을 DatetimeIndex로 사용합니다.")
            processed_df.set_index(pd.to_datetime(processed_df['Date']), inplace=True)
            if processed_df.index.hasnans:
                processed_df = processed_df[~processed_df.index.isna()]
        else:
            logger.error(f"{self.symbol} 차트 데이터에 유효한 DatetimeIndex 또는 변환 가능한 'timestamp'/'Date' 컬럼이 없습니다.")
            self.current_data = None; self.clear_chart_items(); self.price_plot.setTitle(f"{self.symbol} ({self.timeframe_combo.currentText()}) - 시간 데이터 형식 오류"); return

        if not isinstance(processed_df.index, pd.DatetimeIndex) or processed_df.empty:
            logger.error(f"{self.symbol} 차트 데이터: DatetimeIndex 설정에 실패했거나 데이터가 비어있습니다.")
            self.current_data = None; self.clear_chart_items(); self.price_plot.setTitle(f"{self.symbol} ({self.timeframe_combo.currentText()}) - DatetimeIndex 오류"); return
            
        # If the index is now named 'timestamp', rename it to prevent conflict with the upcoming 'timestamp' column.
        if processed_df.index.name == 'timestamp':
            processed_df.index.name = None # Clear the index name or set to something like 'datetime_index'

        rename_map_ohlcv = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        processed_df.rename(columns=lambda c: rename_map_ohlcv.get(c, c.lower()), inplace=True)

        # Add a 'timestamp' column (integer Unix seconds) for internal ChartWidget use.
        # Now there should be no conflict as the index (if it was named 'timestamp') has been renamed.
        processed_df['timestamp_col_int'] = (processed_df.index.astype(np.int64) // 10**9).astype(int) # Use a different name temporarily
        # If the crosshair/other logic strictly needs a column named 'timestamp':
        if 'timestamp' in processed_df.columns and processed_df['timestamp'].dtype != np.int64 : # If a 'timestamp' column already exists and is not our int version
             del processed_df['timestamp'] # remove it before adding the new one
        processed_df.rename(columns={'timestamp_col_int': 'timestamp'}, inplace=True)


        required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp'] 
        if not all(col in processed_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in processed_df.columns]
            logger.error(f"{self.symbol} 차트 데이터에 필수 컬럼 {missing} 중 일부가 누락되었습니다. 현재 컬럼: {processed_df.columns.tolist()}")
            self.current_data = None; self.clear_chart_items(); self.price_plot.setTitle(f"{self.symbol} ({self.timeframe_combo.currentText()}) - 데이터 컬럼 오류"); return

        # Sort by the DatetimeIndex itself, which is usually the desired chronological order.
        # The 'timestamp' column will be sorted accordingly.
        self.current_data = processed_df.sort_index() 
        # self.current_data = processed_df.sort_values(by='timestamp') # This should now work if you prefer sorting by the int column

        self._redraw_base_chart()
        if self.current_data_with_ta is not None and not self.current_data_with_ta.empty:
             self.update_technical_indicators_on_chart(self.current_data_with_ta)

        if self.current_data is not None and not self.current_data.empty and 'timestamp' in self.current_data.columns: # Check the column
            # x_indices (0, 1, 2, ...) are used for plotting, not raw timestamps
            min_idx = self.current_data.reset_index(drop=True).index.min() # Use integer index for range
            max_idx = self.current_data.reset_index(drop=True).index.max()
            self.price_plot.setXRange(min_idx, max_idx, padding=0.02)
            self.volume_plot.setXRange(min_idx, max_idx, padding=0.02)
            self.price_plot.enableAutoRange(axis='y')
            self.volume_plot.enableAutoRange(axis='y')
        else:
            self.price_plot.autoRange()
            self.volume_plot.autoRange()

    def _redraw_base_chart(self):
        self.clear_chart_items(clear_indicators=False) # 기본 차트 아이템 클리어
        if self.current_data is None or self.current_data.empty:
            self.price_plot.setTitle(f"{self.symbol} ({self.timeframe_combo.currentText()}) - 데이터 없음")
            return

        # 연속적인 인덱스 사용
        # self.current_data는 update_historical_data에서 이미 'timestamp'와 소문자 ohlcv를 가짐
        x_indices = np.arange(len(self.current_data))
        # self.current_data에 x_index 컬럼 추가 (다른 곳에서 참조 가능하도록)
        # DataFrame에 직접 컬럼을 추가하는 것보다, x_indices 변수를 직접 사용하는 것이 더 깔끔할 수 있습니다.
        # 하지만 diff에서 current_data['x_index']를 사용하므로 일관성을 위해 유지하거나,
        # x_indices를 plot 함수에 직접 전달하는 방식으로 통일합니다.
        # 여기서는 self.current_data에 추가하는 것으로 유지 (diff와 동일하게)
        self.current_data['x_index'] = x_indices


        current_chart_type = self.chart_type_combo.currentText()
        timeframe_str = self.timeframe_combo.currentText()

        bar_width = 0.8  # 거래량 바 및 캔들스틱의 너비 (인덱스 단위)

        # X축 커스텀 설정 (CustomDateAxisItem 사용)
        # 기존 축이 있다면 제거 (diff의 setParentItem(None) 방식 사용)
        current_bottom_axis = self.volume_plot.getAxis('bottom')
        if current_bottom_axis is not None: # 기존 축 제거
            current_bottom_axis.setParentItem(None) # GraphicsLayout에서 시각적으로 제거
            # self.volume_plot.layout.removeItem(current_bottom_axis) # 레이아웃에서도 제거 (더 확실한 방법)

        # CustomDateAxisItem은 실제 Unix 타임스탬프 배열을 필요로 함
        self.date_axis = CustomDateAxisItem(self.current_data['timestamp'].values, orientation='bottom')
        self.volume_plot.setAxisItems({'bottom': self.date_axis})

        # 거래량 바 (x 좌표로 x_indices 사용)
        self.volume_bar_item = pg.BarGraphItem(
            x=x_indices,
            height=self.current_data['volume'].values,
            width=bar_width, # 인덱스 단위의 너비
            brush=(80,80,150,180),
            pen=pg.mkPen(None)
        )
        self.volume_plot.addItem(self.volume_bar_item)

        if current_chart_type == '라인':
            self.price_line_item = self.price_plot.plot(
                x=x_indices, # x_indices 사용
                y=self.current_data['close'].values,
                pen=pg.mkPen('c', width=2),
                name="종가"
            )
        elif current_chart_type == '캔들스틱':
            candlestick_data_tuples = []
            # self.current_data.iterrows()는 인덱스(DatetimeIndex)와 row(Series)를 반환
            # enumerate를 사용하여 x_indices와 동일한 순서의 정수 인덱스를 얻음
            for idx, (_original_dt_index, row) in enumerate(self.current_data.iterrows()):
                # 'open', 'high', 'low', 'close' 컬럼 사용 (소문자)
                if all(col in row and pd.notnull(row[col]) for col in ['open', 'high', 'low', 'close']):
                    candlestick_data_tuples.append((idx, row['open'], row['high'], row['low'], row['close']))

            if candlestick_data_tuples:
                if self.candlestick_item is None:
                    self.candlestick_item = CandlestickItem(candlestick_data_tuples)
                    self.price_plot.addItem(self.candlestick_item)
                else:
                    self.candlestick_item.setData(candlestick_data_tuples)
            elif self.candlestick_item is not None: # 데이터가 없는데 아이템이 남아있으면 제거
                 self.price_plot.removeItem(self.candlestick_item)
                 self.candlestick_item = None


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
        self.current_data_with_ta = data_with_indicators.copy()
        if data_with_indicators is None or data_with_indicators.empty:
            logger.warning("차트에 표시할 기술적 지표 데이터가 부적절합니다.")
            self.clear_indicator_plots()
            return

        # current_data가 없으면 업데이트하지 않음
        if self.current_data is None or self.current_data.empty:
            logger.warning("기본 차트 데이터가 없어 기술적 지표를 표시할 수 없습니다.")
            return

        logger.debug(f"차트에 기술적 지표 업데이트 시작 (데이터 {len(data_with_indicators)}개)")
        
        # current_data와 동일한 인덱스 사용
        if 'x_index' in self.current_data.columns:
            x_indices = self.current_data['x_index'].values
        else:
            x_indices = np.arange(len(self.current_data))

        # SMA 플롯
        sma_cols = [col for col in data_with_indicators.columns if col.startswith('SMA_')]
        sma_colors = ['#FFA500', '#FF69B4', '#DA70D6']  # 주황, 핫핑크, 오키드
        
        for i, col in enumerate(sma_cols):
            if col in data_with_indicators and pd.api.types.is_numeric_dtype(data_with_indicators[col]):
                # NaN 값 제거
                mask = ~pd.isna(data_with_indicators[col])
                x_valid = x_indices[mask]
                y_valid = data_with_indicators[col].values[mask]
                
                if len(x_valid) > 0:
                    pen = pg.mkPen(color=sma_colors[i % len(sma_colors)], width=2)
                    if col in self.sma_plot_items and self.sma_plot_items[col] is not None:
                        self.sma_plot_items[col].setData(x=x_valid, y=y_valid)
                    else:
                        self.sma_plot_items[col] = self.price_plot.plot(
                            x=x_valid, y=y_valid, pen=pen, name=col
                        )
                    logger.debug(f"{col} 플롯 추가/업데이트됨")

        # EMA 플롯
        ema_cols = [col for col in data_with_indicators.columns if col.startswith('EMA_')]
        ema_colors = ['#00CED1', '#32CD32', '#1E90FF']  # 다크터콰이즈, 라임그린, 도저블루
        
        for i, col in enumerate(ema_cols):
            if col in data_with_indicators and pd.api.types.is_numeric_dtype(data_with_indicators[col]):
                mask = ~pd.isna(data_with_indicators[col])
                x_valid = x_indices[mask]
                y_valid = data_with_indicators[col].values[mask]
                
                if len(x_valid) > 0:
                    pen = pg.mkPen(color=ema_colors[i % len(ema_colors)], width=2, style=Qt.DashLine)
                    if col in self.ema_plot_items and self.ema_plot_items[col] is not None:
                        self.ema_plot_items[col].setData(x=x_valid, y=y_valid)
                    else:
                        self.ema_plot_items[col] = self.price_plot.plot(
                            x=x_valid, y=y_valid, pen=pen, name=col
                        )
                    logger.debug(f"{col} 플롯 추가/업데이트됨")

        # 볼린저 밴드 플롯
        bb_middle_col = next((col for col in data_with_indicators.columns if col.startswith('BB_Middle_')), None)
        if bb_middle_col:
            try:
                window_str = re.search(r'BB_Middle_(\d+)', bb_middle_col).group(1)
            except AttributeError:
                logger.warning(f"볼린저 밴드 기간 추출 실패: {bb_middle_col}")
                window_str = "20"

            bb_upper_col = f'BB_Upper_{window_str}'
            bb_lower_col = f'BB_Lower_{window_str}'

            bb_cols = [(bb_upper_col, '#FFB6C1', Qt.SolidLine),     # 라이트핑크
                       (bb_middle_col, '#DDA0DD', Qt.DotLine),      # 플럼
                       (bb_lower_col, '#FFB6C1', Qt.SolidLine)]     # 라이트핑크

            for col, color, style in bb_cols:
                if col in data_with_indicators and pd.api.types.is_numeric_dtype(data_with_indicators[col]):
                    mask = ~pd.isna(data_with_indicators[col])
                    x_valid = x_indices[mask]
                    y_valid = data_with_indicators[col].values[mask]
                    
                    if len(x_valid) > 0:
                        pen = pg.mkPen(color=color, width=1, style=style)
                        if col in self.bollinger_plot_items and self.bollinger_plot_items[col] is not None:
                            self.bollinger_plot_items[col].setData(x=x_valid, y=y_valid)
                        else:
                            self.bollinger_plot_items[col] = self.price_plot.plot(
                                x=x_valid, y=y_valid, pen=pen, name=col
                            )
            
            logger.debug(f"볼린저 밴드 ({window_str}) 플롯 추가/업데이트됨")
            

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

    def on_mouse_moved_on_price_plot(self, pos_arg_from_signal):
        if self.current_data is None or self.current_data.empty:
            self.hide_crosshair()
            return

        if not isinstance(pos_arg_from_signal, tuple) or not pos_arg_from_signal:
            logger.warning(f"on_mouse_moved_on_price_plot: 예상치 못한 인자 타입 또는 빈 튜플: {pos_arg_from_signal}")
            self.hide_crosshair()
            return
        
        scene_pos_qpointf = pos_arg_from_signal[0]
        if not isinstance(scene_pos_qpointf, QPointF):
            logger.warning(f"on_mouse_moved_on_price_plot: 튜플의 요소가 QPointF가 아님: {scene_pos_qpointf}")
            self.hide_crosshair()
            return

        # 가격 플롯의 ViewBox 가져오기
        vb = self.price_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos_qpointf): # 마우스가 가격 플롯 영역 밖에 있으면 숨김
            self.hide_crosshair()
            return

        mouse_point_in_view = vb.mapSceneToView(scene_pos_qpointf)
        mouse_x_view = mouse_point_in_view.x() # View 좌표계의 X값 (차트의 X축 값, 여기서는 인덱스)
        mouse_y_view = mouse_point_in_view.y() # View 좌표계의 Y값 (차트의 Y축 값, 가격)

        self.v_line.setPos(mouse_x_view)
        self.h_line.setPos(mouse_y_view)
        self.v_line.show()
        self.h_line.show()

        # self.current_data는 DatetimeIndex를 가지고 있고, 'timestamp' 컬럼은 int Unix 초를 가짐.
        # X축은 CustomDateAxisItem 때문에 실제로는 self.current_data.reset_index().index (0, 1, 2...) 기준으로 그려짐.
        # 따라서 mouse_x_view는 이 0, 1, 2... 스케일의 값이어야 함.

        # 가장 가까운 데이터 포인트의 인덱스 찾기
        # mouse_x_view는 플롯 아이템의 x 좌표 (여기서는 0부터 시작하는 정수 인덱스)
        # _redraw_base_chart에서 x_indices = np.arange(len(self.current_data))를 사용했으므로,
        # mouse_x_view를 반올림하여 가장 가까운 정수 인덱스를 얻음.
        index = int(round(mouse_x_view))

        if not (0 <= index < len(self.current_data)):
            self.hide_crosshair() # price_text_item도 숨겨짐
            return
        
        # self.current_data는 DatetimeIndex를 가짐. iloc으로 접근.
        try:
            data_point = self.current_data.iloc[index]
        except IndexError:
            self.hide_crosshair()
            return

        # 날짜 문자열 생성: data_point.name은 DatetimeIndex의 해당 인덱스 값 (Timestamp 객체)
        dt_object = data_point.name # DatetimeIndex에서 가져온 Timestamp 객체
        timeframe = self.timeframe_combo.currentText()
        
        if '일' in timeframe or '주' in timeframe or '월' in timeframe: # 일/주/월봉
            date_str = dt_object.strftime('%Y-%m-%d')
        else: # 분봉, 시간봉 등
            date_str = dt_object.strftime('%Y-%m-%d %H:%M')

        o = data_point.get('open', np.nan)
        h = data_point.get('high', np.nan)
        l = data_point.get('low', np.nan)
        c = data_point.get('close', np.nan)
        v = data_point.get('volume', np.nan)

        def format_val(val, precision=2, is_volume=False):
            if pd.isna(val): return "-"
            return f"{val:,.{precision}f}" if not is_volume else f"{val:,.0f}"

        html_text = f"""<div style='font-family: Consolas, "Courier New", monospace; font-size: 9pt; color: #E0E0E0; background-color:rgba(30,30,30,220); padding: 5px 8px; border-radius: 3px; border: 1px solid #606060;'>
                        <strong style='color:#FFFFFF;'>{date_str}</strong><br>
                        O: <span style='color:#FFD700;'>{format_val(o)}</span> H: <span style='color:#80FF80;'>{format_val(h)}</span><br>
                        L: <span style='color:#FF8080;'>{format_val(l)}</span> C: <span style='color:#80D0FF;'>{format_val(c)}</span><br>
                        Vol: <span style='color:#C0C0C0;'>{format_val(v, is_volume=True)}</span><br>
                        X(idx): <span style='color:#C0C0C0;'>{index} ({mouse_x_view:.2f})</span> Y(가격): <span style='color:#C0C0C0;'>{mouse_y_view:.2f}</span>
                        </div>"""
        self.price_text_item.setHtml(html_text)
        
        view_range_x = vb.viewRange()[0] # 현재 보이는 X축 범위 (인덱스 기준)
        text_anchor_x_view = mouse_x_view # 마우스 X 위치 (인덱스 기준)
        text_anchor_y_view = mouse_y_view # 마우스 Y 위치 (가격 기준)

        # TextItem의 예상 너비 (뷰 좌표계 단위, 대략적)
        # 이 부분은 실제 TextItem 크기에 따라 동적으로 계산하거나, 고정 오프셋 사용이 더 간단할 수 있음
        est_text_width_pixels = 150 
        pixel_width_in_view = vb.pixelWidth() if vb.pixelWidth() > 0 else 0.00001 # 0 방지
        est_text_width_view = est_text_width_pixels * pixel_width_in_view
        
        x_offset_view = (view_range_x[1] - view_range_x[0]) * 0.02 # 뷰 너비의 2% 오프셋 (인덱스 단위)

        # 텍스트 아이템이 오른쪽 경계를 벗어나는지 확인
        if (text_anchor_x_view + x_offset_view + est_text_width_view) > view_range_x[1]:
            # 오른쪽 경계 침범 시, 텍스트를 마우스 왼쪽에 표시
            self.price_text_item.setAnchor((1,1)) # 앵커를 오른쪽 상단으로
            final_text_pos_x = mouse_x_view - x_offset_view # 마우스보다 왼쪽으로
        else:
            # 기본적으로 마우스 오른쪽에 표시
            self.price_text_item.setAnchor((0,1)) # 앵커를 왼쪽 상단으로
            final_text_pos_x = mouse_x_view + x_offset_view # 마우스보다 오른쪽으로
        
        self.price_text_item.setPos(final_text_pos_x, text_anchor_y_view)
        self.price_text_item.show()

    def hide_crosshair(self, event=None): # event 인자는 사용하지 않으므로 Optional 처리하거나 제거 가능
        if self.v_line: self.v_line.hide()
        if self.h_line: self.h_line.hide()
        if self.price_text_item: self.price_text_item.hide()
        # date_text_item은 현재 기본적으로 사용되지 않으므로, 초기화되었다면 숨김 처리
        if hasattr(self, 'date_text_item') and self.date_text_item: # self.date_text_item이 None일 수 있으므로 체크
             self.date_text_item.hide()
             