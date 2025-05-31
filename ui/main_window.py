# ui/main_window.py

import sys
import os # os 모듈 추가
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from ui.widgets.chart_widget import ChartWidget
from ui.widgets.data_widget import DataWidget
from ui.widgets.trading_widget import TradingWidget
from ui.dialogs.settings_dialog import SettingsDialog
import logging # 로깅 추가
import pandas as pd

logger = logging.getLogger(__name__) # 로거 설정

class MainWindow(QMainWindow):
    """메인 윈도우"""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        if self.controller:
            self.controller.main_window_ref = self  # 추가
        self.setWindowTitle("통합 주식 분석 및 트레이딩 시스템")
        self.setGeometry(100, 100, 1600, 900) # 너비 증가 (필요에 따라 조정)

        # 중앙 위젯 설정
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # UI 구성
        self._create_menu_bar()
        self._create_toolbar()
        self._create_layout()
        self._create_status_bar()
        
        self._connect_controller_signals() # 컨트롤러 시그널 연결

        # 스타일 설정
        self._set_style() # 스타일 적용 호출

        # 타이머 설정 (실시간 업데이트)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui_data) # 메서드명 변경 (데이터 업데이트 명시)
        self.update_timer.start(5000)  # 5초마다 업데이트 (API 호출 빈도 고려)

        logger.info("메인 윈도우 초기화 완료")

        # 초기 데이터 로드 (MainWindow가 완전히 준비된 후 호출되도록 QTimer 사용)
        # self.show() 이후에 호출되는 것이 더 안정적일 수 있으나,
        # main.py에서 main_window.show() 이후에 호출하는 것도 방법임.
        # 여기서는 __init__ 마지막에 QTimer로 지연 호출 시도.
        QTimer.singleShot(100, self._initial_data_load) # 0.1초 후 _initial_data_load 호출
        logger.info("_initial_data_load 호출 예약됨.")


    def _connect_controller_signals(self):
        if self.controller:
            try:
                self.controller.historical_data_updated_signal.connect(self.handle_historical_data_updated)
                self.controller.technical_indicators_updated_signal.connect(self.handle_technical_indicators_updated) # *** 수정/추가 ***
                self.controller.realtime_quote_updated_signal.connect(self.handle_realtime_quote_updated)
                self.controller.news_data_updated_signal.connect(self.handle_news_updated)
                self.controller.analysis_result_updated_signal.connect(self.handle_analysis_updated)
                self.controller.economic_data_updated_signal.connect(self.handle_economic_data_updated) # 새 시그널 연결
                self.controller.connection_status_changed_signal.connect(self.handle_connection_status_changed)
                self.controller.status_message_signal.connect(self.handle_status_message)
                self.controller.task_feedback_signal.connect(self.handle_task_feedback)
                logger.info("컨트롤러 시그널 연결 완료.")
            except AttributeError as e:
                logger.error(f"컨트롤러 시그널 연결 중 AttributeError: {e}. Controller에 해당 시그널이 정의되었는지 확인하세요.")
            except Exception as e:
                logger.error(f"컨트롤러 시그널 연결 중 예외 발생: {e}", exc_info=True)
        else:
            logger.warning("컨트롤러가 없어 시그널을 연결할 수 없습니다.")
        
    def _create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('파일(&F)')
        load_action = QAction(QIcon.fromTheme("document-open"), '설정 불러오기(&L)', self)
        load_action.setShortcut(QKeySequence.Open)
        load_action.setStatusTip("설정 파일을 불러옵니다.")
        load_action.triggered.connect(self._load_settings)
        file_menu.addAction(load_action)
        save_action = QAction(QIcon.fromTheme("document-save"), '설정 저장(&S)', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.setStatusTip("현재 설정을 파일에 저장합니다.")
        save_action.triggered.connect(self._save_settings)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        exit_action = QAction(QIcon.fromTheme("application-exit"), '종료(&X)', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip("애플리케이션을 종료합니다.")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        data_menu = menubar.addMenu('데이터(&D)')
        refresh_action = QAction(QIcon.fromTheme("view-refresh"), '데이터 새로고침(&R)', self)
        refresh_action.setShortcut('F5')
        refresh_action.setStatusTip("최신 데이터로 새로고침합니다.")
        refresh_action.triggered.connect(self._refresh_data)
        data_menu.addAction(refresh_action)
        strategy_menu = menubar.addMenu('전략(&T)')
        strategy_settings_action = QAction('전략 설정...', self)
        strategy_settings_action.setStatusTip("트레이딩 전략 설정을 엽니다.")
        strategy_settings_action.triggered.connect(self._open_strategy_dialog)
        strategy_menu.addAction(strategy_settings_action)
        help_menu = menubar.addMenu('도움말(&H)')
        about_action = QAction('정보(&A)', self)
        about_action.setStatusTip("애플리케이션 정보를 표시합니다.")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_toolbar(self):
        toolbar = self.addToolBar('메인 툴바')
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.start_button = QPushButton(QIcon.fromTheme("media-playback-start"), '시작')
        self.start_button.setStatusTip("자동 트레이딩 시스템을 시작합니다.")
        self.start_button.clicked.connect(self._start_trading)
        toolbar.addWidget(self.start_button)
        self.stop_button = QPushButton(QIcon.fromTheme("media-playback-stop"), '중단')
        self.stop_button.setStatusTip("자동 트레이딩 시스템을 중단합니다.")
        self.stop_button.clicked.connect(self._stop_trading)
        self.stop_button.setEnabled(False)
        toolbar.addWidget(self.stop_button)
        toolbar.addSeparator()
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
        self.symbol_combo.setStatusTip("분석할 주식 종목을 선택합니다.")
        self.symbol_combo.currentTextChanged.connect(self._on_symbol_changed)
        toolbar.addWidget(QLabel('종목:'))
        toolbar.addWidget(self.symbol_combo)
        toolbar.addSeparator()
        settings_button = QPushButton(QIcon.fromTheme("preferences-system"), '설정')
        settings_button.setStatusTip("애플리케이션 설정을 엽니다.")
        settings_button.clicked.connect(self._open_settings_dialog)
        toolbar.addWidget(settings_button)

    def _create_layout(self):
        main_splitter = QSplitter(Qt.Vertical)
        top_splitter = QSplitter(Qt.Horizontal)
        self.chart_widget = ChartWidget(self.controller)
        top_splitter.addWidget(self.chart_widget)
        self.data_widget = DataWidget(self.controller)
        top_splitter.addWidget(self.data_widget)
        top_splitter.setStretchFactor(0, 2)
        top_splitter.setStretchFactor(1, 1)
        # top_splitter.setSizes([800, 400]) # 크기는 창 크기에 따라 자동 조절되도록 둘 수 있음
        self.trading_widget = TradingWidget()
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.trading_widget)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        # main_splitter.setSizes([650, 250])
        layout = QVBoxLayout()
        layout.addWidget(main_splitter)
        self.central_widget.setLayout(layout)

    def _create_status_bar(self):
        self.status_bar = self.statusBar()
        self.connection_status = QLabel("연결: 대기중")
        self.data_status = QLabel("데이터: 초기화 중...") # 초기 메시지 변경
        self.strategy_status = QLabel("전략: 중단")
        self.status_bar.addPermanentWidget(self.connection_status)
        self.status_bar.addPermanentWidget(self.data_status) # 작업 피드백용
        self.status_bar.addPermanentWidget(self.strategy_status)
        self.status_bar.showMessage("시스템 준비 완료", 5000)


    def _set_style(self):
        try:
            # 실행 파일 위치 기준으로 경로 설정 (PyInstaller 고려)
            if getattr(sys, 'frozen', False):
                current_dir = os.path.dirname(sys.executable)
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__)) # main_window.py 위치 기준
            
            # ui/styles/dark_style.qss 경로 설정
            # main_window.py는 ui 폴더 내에 있으므로, 한 단계 위로 올라가서 다시 ui/styles로 접근
            project_root_ui = os.path.abspath(os.path.join(current_dir, "..")) # ui 폴더의 부모 (프로젝트 루트)
            qss_file_path = os.path.join(project_root_ui, "ui", "styles", "dark_style.qss")
            
            # 만약 main_window.py가 프로젝트 루트에 있다면 아래와 같이 직접 구성
            # qss_file_path = os.path.join(current_dir, "ui", "styles", "dark_style.qss")

            if os.path.exists(qss_file_path):
                with open(qss_file_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
                logger.info(f"스타일 시트 로드 성공: {qss_file_path}")
            else:
                logger.warning(f"스타일 시트 파일을 찾을 수 없습니다: {qss_file_path}. 기본 스타일이 적용될 수 있습니다.")
        except Exception as e:
            logger.error(f"스타일 시트 로드 중 오류 발생: {e}", exc_info=True)

    def _start_trading(self):
        if self.controller and hasattr(self.controller, 'start'): # 컨트롤러의 start 메서드 확인
            # self.controller.start() # 이 start는 MainController의 start로, workflow/scheduler 시작용.
            # 실제 트레이딩 시스템/전략 시작 로직은 별도로 호출해야 할 수 있음
            # 현재는 MainController의 start/stop이 앱 전체의 백그라운드 작업 제어로 사용됨.
            # 트레이딩 "전략"의 시작/중단은 trading_widget을 통해 별도 관리될 수 있음.
            # 여기서는 버튼 상태만 변경하고, 실제 전략 제어는 TradingWidget과 Controller 연동 필요
            
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.strategy_status.setText("전략: <font color='green'>실행중 (UI)</font>") # UI 상태만 변경
            self.status_bar.showMessage("트레이딩 시스템 (UI상) 시작됨", 5000)
            logger.info("트레이딩 시스템 (UI상) 시작됨. 실제 전략 실행은 컨트롤러/전략 모듈 확인 필요.")
        else:
            QMessageBox.warning(self, "오류", "컨트롤러 또는 시작 메서드가 초기화되지 않았습니다.")
            logger.error("컨트롤러 또는 start 메서드가 없어 트레이딩 시작 불가")

    def _stop_trading(self):
        if self.controller and hasattr(self.controller, 'stop'): # 컨트롤러의 stop 메서드 확인
            # self.controller.stop() # 앱의 백그라운드 작업 중지
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.strategy_status.setText("전략: <font color='orange'>중단 (UI)</font>") # UI 상태만 변경
            self.status_bar.showMessage("트레이딩 시스템 (UI상) 중단됨", 5000)
            logger.info("트레이딩 시스템 (UI상) 중단됨.")
        else:
            logger.error("컨트롤러 또는 stop 메서드가 없어 트레이딩 중단 불가")

    def _on_symbol_changed(self, symbol: str):
        logger.info(f"선택된 종목 변경: {symbol}")
        if self.controller: # 컨트롤러에 현재 심볼 알려주기
            self.controller.set_current_symbol(symbol)

        if hasattr(self.chart_widget, 'set_symbol'):
            self.chart_widget.set_symbol(symbol)
        if hasattr(self.data_widget, 'set_symbol'):
            self.data_widget.set_symbol(symbol)
        if hasattr(self.trading_widget, 'set_symbol'):
            self.trading_widget.set_symbol(symbol)
        
        current_timeframe = self.chart_widget.timeframe_combo.currentText() if self.chart_widget else '1일'
        if self.controller:
            # 과거 데이터 요청이 가장 먼저 시작되어야 TA 계산 및 분석으로 이어짐
            self.controller.request_historical_data(symbol, current_timeframe, force_fetch_from_api=False) # force_fetch_from_api는 필요에 따라 조정
            # 나머지 데이터 요청은 비동기적으로 진행
            self.controller.request_realtime_quote(symbol)
            self.controller.request_news_data(f"{symbol} stock news OR {symbol}", force_fetch_from_api=False) # 심볼 변경 시 뉴스는 강제 업데이트 안 함 (선택)
            # 경제 캘린더는 심볼 변경과 무관하게 주기적으로 업데이트되므로 여기서 호출 안 함
            
    def _refresh_data(self):
        current_symbol = self.symbol_combo.currentText()
        current_timeframe = self.chart_widget.timeframe_combo.currentText() if self.chart_widget else '1일'

        logger.info(f"'{current_symbol}' ({current_timeframe}) 데이터 수동 새로고침 요청") # logger 사용

        if self.chart_widget:
            self.chart_widget.price_plot.setTitle(f"{current_symbol} ({current_timeframe}) - 데이터 새로고침 중...")
            # ChartWidget의 on_timeframe_or_symbol_changed를 직접 호출하여 데이터 요청
            if hasattr(self.chart_widget, 'on_timeframe_or_symbol_changed'):
                 self.chart_widget.on_timeframe_or_symbol_changed()
            else: # 또는 Controller를 통해 직접 요청
                if self.controller and hasattr(self.controller, 'request_historical_data'):
                    self.controller.request_historical_data(current_symbol, current_timeframe)
        if self.data_widget:
            self.data_widget.clear_all_tabs_data() # 데이터 위젯도 초기화

        if self.controller:
            if hasattr(self.controller, 'request_historical_data'):
                self.controller.request_historical_data(current_symbol, current_timeframe)
            if hasattr(self.controller, 'request_realtime_quote'):
                self.controller.request_realtime_quote(current_symbol)
            if hasattr(self.controller, 'request_news_data'):
                self.controller.request_news_data(f"{current_symbol} stock OR {current_symbol} news")
        else:
            logger.warning("컨트롤러가 없어 데이터 새로고침 불가")
            if hasattr(self, 'data_status'): self.data_status.setText(f"데이터 ({current_symbol}): <font color='red'>컨트롤러 없음</font>")

    def _update_ui_data(self):
        """UI 실시간 데이터 업데이트 (컨트롤러로부터 데이터 가져오기)"""
        # 이 메서드는 주기적으로 controller에게 데이터 업데이트를 요청하거나,
        # controller가 이벤트를 통해 데이터를 푸시하면 여기서 받아서 각 위젯에 전달할 수 있습니다.
        # 현재는 _refresh_data가 데이터 요청을 트리거하고,
        # controller는 EventManager를 통해 'data_updated_for_ui' 같은 이벤트를 발행하여
        # MainWindow가 구독하고 있다가 위젯에 전달하는 방식이 더 효율적일 수 있습니다.
        # 지금은 간단히 주기적으로 새로고침을 호출하는 형태로 두겠습니다.
        # self._refresh_data() # 너무 잦은 호출은 API 제한을 유발할 수 있음

        # 대신 controller가 데이터를 가지고 있고, UI가 이를 polling 하는 방식
        if self.controller:
            current_symbol = self.symbol_combo.currentText()
            # 실시간 호가 업데이트
            if hasattr(self.controller, 'request_realtime_quote'): # 컨트롤러에 실시간 데이터 요청
                self.controller.request_realtime_quote(current_symbol)

            # 포지션 업데이트 (예시)
            # positions = self.controller.get_portfolio_positions()
            # if positions and self.trading_widget:
            #     self.trading_widget.update_positions_display(positions)


    def _load_settings(self):
        if not self.controller or not hasattr(self.controller, 'settings'):
            QMessageBox.warning(self, "오류", "설정 관리자가 준비되지 않았습니다.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "설정 파일 열기", self.controller.settings.get_config_directory(), "JSON Files (*.json)"
        )
        if filename:
            if self.controller.settings.load_config(filename):
                self.status_bar.showMessage("설정 불러오기 완료", 5000)
                QMessageBox.information(self, "성공", "설정을 성공적으로 불러왔습니다.")
                logger.info(f"설정 파일 로드: {filename}")
                if hasattr(self.controller, 'apply_updated_settings'):
                    self.controller.apply_updated_settings() # 로드 후 설정 적용
            else:
                QMessageBox.critical(self, "오류", "설정 파일 로드에 실패했습니다.")
                logger.error(f"설정 파일 로드 실패: {filename}")

    def _save_settings(self):
        if not self.controller or not hasattr(self.controller, 'settings'):
            QMessageBox.warning(self, "오류", "설정 관리자가 준비되지 않았습니다.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "설정 파일 저장", self.controller.settings.get_config_directory(), "JSON Files (*.json)"
        )
        if filename:
            if not filename.endswith(".json"): filename += ".json"
            if self.controller.settings.save_config(filename):
                self.status_bar.showMessage("설정 저장 완료", 5000)
                QMessageBox.information(self, "성공", "설정을 성공적으로 저장했습니다.")
                logger.info(f"설정 파일 저장: {filename}")
            else:
                QMessageBox.critical(self, "오류", "설정 파일 저장에 실패했습니다.")
                logger.error(f"설정 파일 저장 실패: {filename}")

    def _open_settings_dialog(self):
        if not self.controller or not hasattr(self.controller, 'settings'):
            QMessageBox.warning(self, "오류", "설정 관리자가 준비되지 않았습니다.")
            logger.error("설정 대화상자 열기 실패: 컨트롤러 또는 설정 객체 없음")
            return
        dialog = SettingsDialog(self.controller.settings, self)
        if dialog.exec_() == QDialog.Accepted: # Dialog.Accepted (QDialog 임포트 확인)
            self.status_bar.showMessage("설정 업데이트 완료 (저장됨)", 5000)
            logger.info("설정 업데이트 완료 (저장됨)")
            if hasattr(self.controller, 'apply_updated_settings'):
                self.controller.apply_updated_settings()
                
    def _open_strategy_dialog(self):
        """전략 설정 대화상자 열기 (구현 예정)"""
        # TODO: 전략 설정 관련 다이얼로그 (예: ML전략 파라미터, 기술적 분석 조건 등)
        QMessageBox.information(self, "알림", "전략 설정 기능은 현재 개발 중입니다.")
        logger.info("전략 설정 대화상자 요청 (미구현)")


    def _show_about(self):
        """정보 대화상자 표시"""
        QMessageBox.about(self, "정보",
            "<b>통합 주식 분석 및 트레이딩 시스템</b><br>"
            "버전 0.1.0 (Alpha)<br><br>"
            "본 시스템은 PyQt5와 다양한 데이터 분석 라이브러리를 활용하여<br>"
            "주식 분석 및 자동 트레이딩 환경을 제공하는 것을 목표로 합니다.<br><br>"
            "<b>주의:</b> 교육 및 연구 목적으로 개발되었으며, 실제 투자 결정은 신중히 하시기 바랍니다."
        )

    def closeEvent(self, event: QCloseEvent): # 타입 힌트 추가
        """종료 이벤트 처리"""
        reply = QMessageBox.question(self, '종료 확인',
            '정말로 프로그램을 종료하시겠습니까?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)

        if reply == QMessageBox.Yes:
            logger.info("애플리케이션 종료 시작")
            if self.controller:
                self.controller.stop() # 컨트롤러의 정리 작업 수행
            event.accept()
        else:
            logger.info("애플리케이션 종료 취소")
            event.ignore()

    @pyqtSlot(str, object) # historical_data_updated_signal의 인자에 맞게
    def handle_historical_data_updated(self, symbol: str, data_df_object: object):
        logger.info(f"시그널 수신: MainWindow.handle_historical_data_updated (심볼: {symbol}, 데이터 타입: {type(data_df_object)})")
        if not isinstance(data_df_object, pd.DataFrame):
            logger.error(f"잘못된 과거 데이터 타입 수신: {type(data_df_object)}")
            if self.chart_widget and hasattr(self.chart_widget, 'price_plot') and symbol == self.symbol_combo.currentText():
                 self.chart_widget.price_plot.setTitle(f"{symbol} ({self.chart_widget.timeframe_combo.currentText()}) - 데이터 타입 오류")
            self.handle_task_feedback(f"{symbol} 과거 데이터", "UI 업데이트 실패 (타입 오류)")
            return

        data_df = data_df_object
        if symbol == self.symbol_combo.currentText():
            if self.chart_widget:
                logger.debug(f"{symbol} 원본 과거 데이터 ({len(data_df)} 행) ChartWidget으로 전달.")
                self.chart_widget.update_historical_data(data_df.copy()) # ChartWidget은 원본 데이터를 받아 기본 차트를 그림
            
            if data_df.empty:
                self.handle_task_feedback(f"{symbol} 과거 데이터", "데이터 없음 (UI)")
            else:
                # 기술적 지표 계산은 Controller에서 historical_data_updated_signal과 별도로 technical_indicators_updated_signal을 통해 전달됨
                # self.handle_task_feedback(f"{symbol} 과거 데이터", "차트 업데이트됨 (UI)") # 이 메시지는 technical_indicators_updated 후로 옮기는 것이 나을 수 있음
                pass 
        else:
            logger.debug(f"수신된 과거 데이터 ({symbol})가 현재 선택된 심볼({self.symbol_combo.currentText()})과 달라 UI 업데이트 건너뜀.")

    # 기술적 지표 업데이트를 위한 새 슬롯 추가
    @pyqtSlot(str, str, object)
    def handle_technical_indicators_updated(self, symbol: str, timeframe: str, data_with_indicators_obj: object):
        logger.info(f"시그널 수신: MainWindow.handle_technical_indicators_updated (심볼: {symbol}, 기간: {timeframe})")
        if not isinstance(data_with_indicators_obj, pd.DataFrame):
            logger.error(f"잘못된 기술적 지표 데이터 타입 수신: {type(data_with_indicators_obj)}")
            self.handle_task_feedback(f"{symbol} 기술적 지표", "UI 업데이트 실패 (데이터 타입 오류)")
            return

        data_with_indicators_df = data_with_indicators_obj
        
        if symbol == self.symbol_combo.currentText(): # 현재 선택된 심볼에 대해서만 업데이트
            # DataWidget의 기술적 지표 탭 업데이트
            if self.data_widget and hasattr(self.data_widget, 'update_technical_indicators_display'):
                self.data_widget.update_technical_indicators_display(symbol, timeframe, data_with_indicators_df.copy())
                logger.debug(f"{symbol} ({timeframe}) 기술적 지표 DataWidget에 업데이트됨.")

            # ChartWidget에 기술적 지표 플롯 업데이트
            if self.chart_widget and hasattr(self.chart_widget, 'update_technical_indicators_on_chart'):
                # ChartWidget은 이미 update_historical_data에서 기본 데이터를 받았으므로,
                # 여기서는 지표가 포함된 데이터로 지표만 다시 그리도록 함.
                # ChartWidget의 timeframe_combo.currentText()와 수신된 timeframe이 일치하는지도 확인 필요
                if self.chart_widget.timeframe_combo.currentText() == timeframe:
                    self.chart_widget.update_technical_indicators_on_chart(data_with_indicators_df.copy())
                    logger.debug(f"{symbol} ({timeframe}) 기술적 지표 ChartWidget에 업데이트됨.")
                else:
                    logger.debug(f"ChartWidget의 현재 시간대({self.chart_widget.timeframe_combo.currentText()})와 수신된 지표 데이터의 시간대({timeframe})가 달라 차트 지표 업데이트 건너뜀.")
            self.handle_task_feedback(f"{symbol} ({timeframe}) 기술적 지표", "UI 업데이트 완료")
        else:
            logger.debug(f"수신된 기술적 지표({symbol})가 현재 선택된 심볼({self.symbol_combo.currentText()})과 달라 UI 업데이트 건너뜀.")
            
    def handle_realtime_quote_updated(self, symbol: str, quote_data: dict):
        logger.info(f"시그널 수신: MainWindow.handle_realtime_quote_updated (심볼: {symbol})")
        if symbol == self.symbol_combo.currentText() and self.data_widget:
            self.data_widget.update_realtime_quote_display(quote_data)
            logger.debug(f"{symbol} 실시간 호가 UI 업데이트됨: {quote_data.get('price')}")
        # 실시간 호가는 상태바의 data_status를 직접 업데이트하지 않고, DataWidget 내부에서만 표시

    def handle_news_updated(self, symbol_or_keywords: str, news_list: list):
        logger.info(f"시그널 수신: MainWindow.handle_news_updated (키워드: {symbol_or_keywords}, 뉴스 {len(news_list)}개)")
        # 현재 선택된 심볼과 뉴스 키워드가 연관있는지 확인하는 로직 필요 (선택적)
        # 예를 들어, self.symbol_combo.currentText()가 symbol_or_keywords에 포함되는지 등
        current_symbol = self.symbol_combo.currentText()
        if current_symbol in symbol_or_keywords and self.data_widget: # 단순 포함 관계 확인
            self.data_widget.update_news_display(news_list)
            logger.debug(f"'{symbol_or_keywords}' 관련 뉴스 DataWidget에 업데이트됨.")

    def handle_analysis_updated(self, symbol: str, analysis_results: dict):
        logger.info(f"시그널 수신: MainWindow.handle_analysis_updated (심볼: {symbol})")
        if symbol == self.symbol_combo.currentText() and self.data_widget:
            self.data_widget.update_analysis_display(analysis_results)
            logger.debug(f"{symbol} 분석 결과 DataWidget에 업데이트됨.")

    def handle_connection_status_changed(self, status_message: str, is_connected: bool):
        logger.info(f"시그널 수신: MainWindow.handle_connection_status_changed ({status_message}, 연결됨: {is_connected})")
        color = "lightgreen" if is_connected else "red"
        if hasattr(self, 'connection_status'):
            self.connection_status.setText(f"연결: <font color='{color}'>{status_message}</font>")

    def handle_task_feedback(self, task_name: str, status: str):
        logger.info(f"시그널 수신: MainWindow.handle_task_feedback ('{task_name}': {status})")
        if hasattr(self, 'data_status') and self.data_status:
            feedback_message = f"{task_name}: {status}"
            self.data_status.setText(feedback_message)
            # logger.info(f"작업 피드백 UI 업데이트: {feedback_message}") # 너무 빈번할 수 있어 debug로 변경 또는 제거

            if "실패" in status or "오류" in status:
                self.data_status.setStyleSheet("QLabel { color : #F44336; }") # 빨간색
            elif "완료" in status or "로드 완료" in status:
                self.data_status.setStyleSheet("QLabel { color : #4CAF50; }") # 초록색
            else: # "요청 시작", "진행 중", "API 호출 중" 등
                self.data_status.setStyleSheet("QLabel { color : #FFEB3B; }") # 노란색
        else:
            logger.warning(f"data_status 라벨이 없어 작업 피드백 표시 불가: {task_name} - {status}")

    def handle_status_message(self, message: str, timeout: int = 3000):
        if hasattr(self, 'status_bar') and self.status_bar:
            self.status_bar.showMessage(message, timeout)
        else:
            logger.warning(f"상태바가 없어 메시지 표시 불가: {message}")


    def _initial_data_load(self):
        current_symbol = self.symbol_combo.currentText()
        if hasattr(self, 'chart_widget') and self.chart_widget:
            current_timeframe = self.chart_widget.timeframe_combo.currentText()
        else:
            current_timeframe = '1일'
            logger.warning("ChartWidget이 아직 준비되지 않아 기본 시간대('1일')를 사용합니다.")

        if self.controller:
            logger.info(f"초기 데이터 로드 시작: 심볼={current_symbol}, 시간대={current_timeframe}")
            if hasattr(self, 'data_status') and self.data_status:
                 self.data_status.setText(f"데이터 ({current_symbol}): <font color='yellow'>초기 로딩 중...</font>")
            if hasattr(self, 'chart_widget') and self.chart_widget and hasattr(self.chart_widget, 'price_plot'):
                 self.chart_widget.price_plot.setTitle(f"{current_symbol} ({current_timeframe}) - 데이터 로딩 중...")
                 self.chart_widget.clear_chart_items()

            if hasattr(self.controller, 'request_historical_data'):
                self.controller.request_historical_data(current_symbol, current_timeframe)
            else:
                logger.error("컨트롤러에 'request_historical_data' 메서드가 없습니다.")
            
            if hasattr(self.controller, 'request_realtime_quote'):
                self.controller.request_realtime_quote(current_symbol)
            if hasattr(self.controller, 'request_news_data'):
                 self.controller.request_news_data(f"{current_symbol} stock OR {current_symbol} news")
            
            # 경제 캘린더 데이터 초기 로드 추가 (예: 향후 7일)
            if hasattr(self.controller, 'request_economic_calendar_update'):
                self.controller.request_economic_calendar_update(days_future=7) 
            else:
                logger.error("컨트롤러에 'request_economic_calendar_update' 메서드가 없습니다.")

        else:
            logger.error("컨트롤러가 없어 초기 데이터 로드를 수행할 수 없습니다.")
            if hasattr(self, 'data_status') and self.data_status:
                self.data_status.setText(f"데이터 ({current_symbol}): <font color='red'>컨트롤러 오류</font>")

    # 경제 캘린더 데이터 처리 핸들러 추가
    @pyqtSlot(object) # 인자가 pd.DataFrame이므로 object로 받음
    def handle_economic_data_updated(self, economic_data_df: pd.DataFrame):
        logger.info(f"시그널 수신: MainWindow.handle_economic_data_updated (데이터 {len(economic_data_df) if economic_data_df is not None else 'None'}개)")
        if self.data_widget and hasattr(self.data_widget, 'update_economic_indicators_display'):
            self.data_widget.update_economic_indicators_display(economic_data_df)
            logger.debug(f"경제 캘린더 데이터 DataWidget에 업데이트됨 ({len(economic_data_df) if economic_data_df is not None else 0}개).")
        else:
            logger.warning("DataWidget 또는 update_economic_indicators_display 메서드가 없어 경제 캘린더 UI 업데이트 불가.")
