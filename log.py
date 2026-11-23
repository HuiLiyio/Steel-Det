import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self, filename, level='info', when='D', backupCount=7, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                 max_size=None):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        # 文件处理器（限制单个文件大小）
        # rotating_handler = RotatingFileHandler(
        #     filename,
        #     maxBytes=max_size,
        #     backupCount=backupCount,
        #     encoding='utf-8'
        # )
        # rotating_handler.setFormatter(format_str)  # 设置文件里写入的格式

        # 时间处理器（按日期轮转）
        timed_handler = TimedRotatingFileHandler(
            filename,
            when='midnight',
            interval=1,
            backupCount=backupCount,
            encoding='utf-8'
        )
        timed_handler.setFormatter(format_str)  # 设置文件里写入的格式

        # 添加处理器到 logger
        self.logger.addHandler(sh)
        #self.logger.addHandler(rotating_handler)
        self.logger.addHandler(timed_handler)

    def get_logger(self):
        return self.logger
# use logging
