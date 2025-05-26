# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from core.controller import MainController
from utils.logger import setup_logger # 임포트 확인
from config.settings import Settings
import logging # logging 임포트

def main():
    # 로깅 설정 (가장 먼저 호출, DEBUG 레벨로 상세 로깅)
    # setup_logger()의 name 인자를 생략하거나 None으로 전달하면 루트 로거에 설정됨.
    # 이렇게 하면 다른 모듈에서 logging.getLogger(__name__)으로 가져오는 로거들도 이 설정을 상속받음.
    setup_logger(name=None, level=logging.DEBUG) # 루트 로거에 DEBUG 레벨 설정

    main_logger = logging.getLogger(__name__) # main 모듈용 로거
    main_logger.info("애플리케이션 시작 준비...")

    # 설정 로드
    settings = Settings() # base_path 설정 필요 시 수정
    main_logger.info(f"설정 파일 경로: {settings.config_file_path}")
    main_logger.info(f"AlphaVantage Key 로드됨: {'Yes' if settings.api_config.alpha_vantage_key else 'No'}")
    main_logger.info(f"NewsAPI Key 로드됨: {'Yes' if settings.api_config.news_api_key else 'No'}")


    # PyQt 애플리케이션 생성
    app = QApplication(sys.argv)
    main_logger.info("QApplication 생성 완료.")

    # 메인 컨트롤러 생성
    try:
        controller = MainController(settings)
        main_logger.info("MainController 생성 완료.")
    except Exception as e:
        main_logger.critical(f"MainController 생성 중 치명적 오류: {e}", exc_info=True)
        sys.exit(1) # 컨트롤러 생성 실패 시 종료


    # 메인 윈도우 생성
    try:
        main_window = MainWindow(controller)
        main_logger.info("MainWindow 생성 완료.")
        main_window.show()
        main_logger.info("MainWindow 표시됨.")
    except Exception as e:
        main_logger.critical(f"MainWindow 생성 또는 표시 중 치명적 오류: {e}", exc_info=True)
        sys.exit(1) # 메인 윈도우 문제 시 종료


    # 컨트롤러 시작 (UI 표시 후)
    try:
        if hasattr(main_window, '_initial_data_load'): # 초기 데이터 로드 함수가 있다면 호출
            main_window._initial_data_load()
        controller.start() # WorkflowEngine, Scheduler 시작
        main_logger.info("MainController 시작됨.")
    except Exception as e:
        main_logger.error(f"MainController 시작 중 오류: {e}", exc_info=True)


    # 애플리케이션 실행
    try:
        exit_code = app.exec_()
        main_logger.info(f"애플리케이션 종료 (종료 코드: {exit_code}).")
        sys.exit(exit_code)
    except Exception as e:
        main_logger.critical(f"애플리케이션 실행 중 치명적 오류: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()