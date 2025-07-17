import logging
from logging.handlers import RotatingFileHandler
import time

class Logger:
    def __init__(self, log_file="casestudy3.log", max_bytes=1_000_000, backup_count=5):
        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.DEBUG)

        # 避免重复添加 handler
        if not self.logger.handlers:
            handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.log_file = log_file

    def log(self, message, level="info"):
        """
        写入日志并打印到控制台。
        支持 level: "info", "debug", "warning", "error", "critical"
        """
        if level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
        else:
            self.logger.info(message)

        print(message)

    def log_start_time(self):
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.log(f"Start time: {start_time}")

# 示例使用
if __name__ == "__main__":
    logger = Logger("mylog.log")
    logger.log_start_time()
    logger.log("This is an info message.")
    logger.log("This is a debug message.", level="debug")
    logger.log("This is an error message.", level="error")
