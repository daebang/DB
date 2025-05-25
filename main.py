import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from core.controller import MainController
from utils.logger import setup_logger
from config.settings import Settings

def main():
    # 로깅 설정
    setup_logger()
    
    # 설정 로드
    settings = Settings()
    
    # PyQt 애플리케이션 생성
    app = QApplication(sys.argv)
    
    # 메인 컨트롤러 생성
    controller = MainController(settings)
    
    # 메인 윈도우 생성
    main_window = MainWindow(controller)
    main_window.show()
    
    # 애플리케이션 실행
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
