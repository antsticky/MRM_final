import os
import enum
import inspect
import datetime


class LoggerLevel(enum.Enum):
    CRITICAL = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3

    def __eq__(self, other):
        return self.value == other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value


class Logger:
    def __init__(self, level=LoggerLevel.WARNING):
        self._level = level

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, new_level):
        self._level = new_level

    def __msg(self, message, filename, ln, level):
        if level <= self.level:
            print(f"{datetime.datetime.now()}[{level.name}] - {os.path.basename(filename)}:{ln} - {message}")
        else:
            pass

    def critical(self, message):
        frame, filename, ln, fn, lolc, idx = inspect.getouterframes(inspect.currentframe())[1]
        self.__msg(message=message, filename=filename, ln=ln, level=LoggerLevel.CRITICAL)

    def warning(self, message):
        frame, filename, ln, fn, lolc, idx = inspect.getouterframes(inspect.currentframe())[1]
        self.__msg(message=message, filename=filename, ln=ln, level=LoggerLevel.WARNING)

    def info(self, message):
        frame, filename, ln, fn, lolc, idx = inspect.getouterframes(inspect.currentframe())[1]
        self.__msg(message=message, filename=filename, ln=ln, level=LoggerLevel.INFO)

    def debug(self, message):
        frame, filename, ln, fn, lolc, idx = inspect.getouterframes(inspect.currentframe())[1]
        self.__msg(message=message, filename=filename, ln=ln, level=LoggerLevel.DEBUG)


logger = Logger()
