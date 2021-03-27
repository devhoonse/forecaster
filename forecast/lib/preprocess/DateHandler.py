import datetime

from ..common.SingletonInstance import SingletonInstance
from ...conf import date_range


class DateHandler(SingletonInstance):    # todo: define methods
    
    @staticmethod
    def check_date(x, fmt: str = '%Y-%m-%d %H:%M:%S'):
        filter1 = date_range.start_date < datetime.datetime.strptime(x, fmt) 
        filter2 = date_range.end_date > datetime.datetime.strptime(x, fmt) 
        return(filter1 & filter2)
    
    @staticmethod
    def days_between(d1, d2, d1_fmt: str = '%Y-%m-%d %H:%M:%S', d2_fmt: str = '%Y-%m-%d %H:%M:%S'):
        d1 = datetime.datetime.strptime(d1, d1_fmt)
        d2 = datetime.datetime.strptime(d2, d2_fmt)
        return abs((d2 - d1).days)

