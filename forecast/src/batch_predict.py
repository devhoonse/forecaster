# import built-ins
import pickle
import sys
import os
import datetime
import argparse
import json

# import 3rd-party libraries
import pandas as pd

# import user-defined library
from ..lib.conf.ApplicationConfiguration import ApplicationConfiguration
from ..lib.app import MainEnvironment


# 명령행 매개변수를 정의합니다.
def main_argparser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--properties_path', 
        type=str, 
        default=''    # 입력된 값이 없으면, 예측용 test 세트 데이터로 오늘 일자 데이터를 사용합니다.
    )
    parser.add_argument(
        '--yyyymmdd', 
        type=str, 
        default=datetime.datetime.now().strftime("%Y%m%d")    # 입력된 값이 없으면, 예측용 test 세트 데이터로 오늘 일자 데이터를 사용합니다.
    )
    
    return parser


def main(args: argparse.Namespace) -> None:
    """main 함수입니다."""
    
    # 설정 정보를 가져옵니다.
    modeler_properties_path = args.properties_path
    if not os.path.exists(modeler_properties_path):
        raise FileNotFoundError(f"modeler_properties_path == {modeler_properties_path}")
    configuration = ApplicationConfiguration.instance()
    configuration.init(modeler_properties_path)
    
    env = MainEnvironment.instance()
    env.init(configuration, args)
    env.main()
    

# 메인 함수를 호출합니다.
if __name__ == '__main__':
    args = main_argparser().parse_args()
    main(args)
