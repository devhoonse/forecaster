# import built-ins
import time
import pickle
import os
import datetime
import argparse
import logging

# import third-parties
import jaydebeapi
import pandas as pd

# import user-defined library
from ..lib.common.SingletonInstance import SingletonInstance
from ..lib.conf.ApplicationConfiguration import ApplicationConfiguration
from ..lib.logger.LogHandler import LogHandler
from ..lib.dao.DatabaseManager import DatabaseManager
from ..lib.preprocess.Preprocessing import Preprocessing

# settings
modeler_properties_path = './forecast/conf/modeler.properties'


def main_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="rate")
    return parser


def main(args: argparse.Namespace) -> None:
    """
    설정 파일에 지정된 TIBERO D/B 로부터 조회된 데이터를 
    직렬화된 .pickle 확장자 파일로 저장합니다.
    """
    
    # 설정 정보를 가져옵니다.
    configuration = ApplicationConfiguration.instance()
    configuration.init(modeler_properties_path)
    
    # 로그 핸들러를 초기화합니다.
    LogHandler.instance().init(configuration)
    
    # 로거 객체입니다
    logger_debug = LogHandler.instance().get_logger('debug-logger')
    logger_error = LogHandler.instance().get_logger('error-logger')
    
    try:
        # main logic inside try block for catching any caught or uncaught exceptions
    
        # 직렬화된 파일명에 포함될 YYYYMMDD 문자열은 메인 함수 내에서 알아서 잡도록 하였습니다
        yyyymmdd = datetime.datetime.now().strftime("%Y%m%d")
        src_path = configuration.find("Server", "serialized.file").format(
            model_id=args.model_id, 
            yyyymmdd=yyyymmdd
        )
        logger_debug.debug(f"src_path == {src_path}")
        logger_debug.debug(
            f"LogHandler.instance().get_loggers() == {LogHandler.instance().get_loggers()}")

        # D/B 접속 및 조회 요청을 위임하기 위한 매니저 객체를 초기화합니다
        database_manager = DatabaseManager(configuration)
        
        # 앞 단계에서 직렬화된 .pickle 데이터를 역직렬화하여 메모리로 불러옵니다.
        data_frame: pd.DataFrame = database_manager.deserialize_from_pickle(src_path)
        logger_debug.debug(f"data_frame deserialized: {data_frame is not None}")
        logger_debug.debug(f"data_frame[:10] == {data_frame[:10]}")
        
        # 요청받은 모델에 맞는 전처리를 수행합니다.
        preprocessed = Preprocessing.run(args.model_id, data_frame)
        logger_debug.debug(f"preprocessed[:10] == {preprocessed[:10]}")
        
#         # 체결된 접속을 해제합니다.
#         disconnection_success: bool = database_manager.disconnect_database()
#         logger_debug.debug(f"D/B has been disconnected: {disconnection_success}")
        
    except Exception as e:
        # for any caught or uncaught exceptions
        logger_debug.error(f"Exception {str(e)}")
        logger_error.error(f"Exception {str(e)}")


if __name__ == '__main__':
    args = main_argparser().parse_args()
    main(args)