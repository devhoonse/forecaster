import logging
import logging.config
import json

from ..common.SingletonInstance import SingletonInstance
from ..conf.ApplicationConfiguration import ApplicationConfiguration


class LogHandler(SingletonInstance):
    def __init__(self):
        pass
    
    @classmethod
    def init(cls, config: ApplicationConfiguration):
        
        with open(config.find("Server", "logging.conf"), "rt") as io_wrapper:
            config = json.load(io_wrapper)
#         print(f"logging config == {config}")
        logging.config.dictConfig(config)

    def get_logger(self, name: str):
        return logging.getLogger(name)

    def get_loggers(self):
        loggers = {"ROOT": logging.getLogger()}
#         print(f"logging.root.manager.loggerDict == {logging.root.manager.loggerDict}")
        for name in logging.root.manager.loggerDict:
            loggers.update({name: logging.getLogger(name)})
        return loggers
