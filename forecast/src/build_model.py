# import built-ins
import datetime
import argparse
import json

# import user-defined library
from ..lib.conf.ApplicationConfiguration import ApplicationConfiguration
from ..lib.app import MainEnvironment


# 설정 정보를 가져옵니다.
modeler_properties_path = './forecast/conf/modeler.properties'
configuration = ApplicationConfiguration.instance()
configuration.init(modeler_properties_path)


# 명령행 매개변수를 정의합니다.
def main_argparser() -> argparse.ArgumentParser:
    
    with open(configuration.find("Model", "definition.json"), "rt") as io_wrapper:
        config = json.load(io_wrapper)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="occr")
    parser.add_argument('--train_date', type=str, default="")    # todo: 적합할 모델에 따라 다른 학습일자를 사용하도록 -> MainEnvironment 초기화 시 conf 파일을 읽도록
    parser.add_argument('--yyyymmdd', type=str, default=datetime.datetime.now().strftime("%Y%m%d"))
    
    return parser


# 메인 함수입니다.
def main(args: argparse.Namespace) -> None:
    """
    설정 파일에 지정된 TIBERO D/B 로부터 조회된 데이터를 
    직렬화된 .pickle 확장자 파일로 저장합니다.
    """
    
    env = MainEnvironment()
    env.init(configuration, args)
    
    fit_data = env.load_data(use_for='fit')
    fit_data_preprocessed = env.preprocess_data(fit_data)
    
    pred_data = env.load_data(use_for='predict')
    pred_data_preprocessed = env.preprocess_data(pred_data)
    
    env.fit(fit_data_preprocessed)
    
    predicted_result = env.predict(pred_data_preprocessed)
    
    print(predicted_result)


# 메인 함수를 호출합니다.
if __name__ == '__main__':
    args = main_argparser().parse_args()
    main(args)
