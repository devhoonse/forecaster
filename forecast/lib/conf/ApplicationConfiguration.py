import os
import configparser

from ..common.SingletonInstance import SingletonInstance


class ApplicationConfiguration(SingletonInstance):
    def __init__(self):
        self.__config = configparser.RawConfigParser()
        self.__config.optionxform = str
        
    def init(self, properties_file: str):
        self.__config.read(properties_file)
    
    def find(self, section: str, name: str):
#         value = None
#         section = self.__config.get(section, None)
#         if section is not None:
#             return section.get(name, None)
        return self.__config[section][name]
