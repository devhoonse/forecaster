# import built-ins
import time
import pickle
import sys
import os
import datetime
import argparse
import logging
import json

# import third-parties
import jaydebeapi
import pandas as pd

# import user-defined library
from .common.SingletonInstance import SingletonInstance
from .conf.ApplicationConfiguration import ApplicationConfiguration
from .logger.LogHandler import LogHandler
from .dao.DatabaseManager import DatabaseManager
from .preprocess.Preprocessing import Preprocessing
from .model import ModelManager


# # 설정 정보를 가져옵니다.
# modeler_properties_path = './forecast2/conf/modeler.properties'
# configuration = ApplicationConfiguration.instance()
# configuration.init(modeler_properties_path)


class MainEnvironment(SingletonInstance):
    """
    메인에서 실행되어야 할 동작들을 정리한 클래스입니다.
    런타임 내에서 싱글톤 객체로 관리됩니다.
    """
    
    
    def __init__(self):
        """생성자입니다. 멤버 변수를 선언합니다."""
        
        # 개별 모델별 명령행 매개변수를 정의합니다.
        self.ARGS_DICT = {
            "pdmg": argparse.Namespace(
                model_id="pdmg",
                train_date="",    # "" 로 설정하면 model_hyperparams.json 에 정의된 train_date 일자를 사용하게 됩니다.
                yyyymmdd=""
            ),
            "hdmg": argparse.Namespace(
                model_id="hdmg",
                train_date="",    # "" 로 설정하면 model_hyperparams.json 에 정의된 train_date 일자를 사용하게 됩니다.
                yyyymmdd=""
            ),
            "occr": argparse.Namespace(
                model_id="occr",
                train_date="",    # "" 로 설정하면 model_hyperparams.json 에 정의된 train_date 일자를 사용하게 됩니다.
                yyyymmdd=""
            )
        }
        
        self.DATA_DICT = {    # 21/02/08
            "fit": {
                "pdmg": dict(),
                "hdmg": dict(),
                "occr": dict()
            },
            "predict": {
                "pdmg": dict(),
                "hdmg": dict(),
                "occr": dict()
            }
        }
        
        self.model_id = ""
        self.train_date = ""
        self.yyyymmdd = ""
        
        self.model_data_path = ""
        self.model_path = ""
        self.input_data_path = ""
        
        self.config = None
        
        self.logger_debug = None
        self.logger_error = None
        self.database_manager = None
        self.modeler = None
        
        self.batch_size = 0
        self.current_step = 0
        
        self.target_scheme = ''
        self.target_table = ''
    
    def __in_step(self):    # , df: pd.DataFrame
        count_total = self.__get_count(self.yyyymmdd)
        return self.current_step * self.batch_size in range(0, count_total, self.batch_size)    # 21.02.04
    
    def __reset_step(self):
        self.current_step = 0
        
    def __progress_step(self):
        self.current_step += 1
    
    def init(self, config: ApplicationConfiguration, args: argparse.Namespace) -> None:
        """인스턴스 초기화를 수행합니다."""
        
        self.config = config
        
        # 로그 핸들러를 초기화합니다.
        LogHandler.instance().init(config)

        # 로거 객체입니다
        self.logger_debug = LogHandler.instance().get_logger('debug-logger')
        self.logger_error = LogHandler.instance().get_logger('error-logger')

        # for debugging - 로그 콘솔에 출력합니다.
        self.logger_debug.debug(f"self.model_data_path == {self.model_data_path}")
        self.logger_debug.debug(f"self.model_path == {self.model_path}")
        self.logger_debug.debug(f"self.input_data_path == {self.input_data_path}")
        self.logger_debug.debug(
            f"LogHandler.instance().get_loggers() == {LogHandler.instance().get_loggers()}")
        
        # D/B 접속 및 조회 요청을 위임하기 위한 매니저 객체를 초기화합니다
        self.database_manager = DatabaseManager.instance()
        self.database_manager.init(config)
        
        # 모델러 인스턴스를 초기화합니다.
        self.modeler = ModelManager.instance()
        self.modeler.init(config)
        
        # 21/02/09
        self.yyyymmdd = args.yyyymmdd
        self.ARGS_DICT['pdmg'].yyyymmdd = args.yyyymmdd
        self.ARGS_DICT['hdmg'].yyyymmdd = args.yyyymmdd
        self.ARGS_DICT['occr'].yyyymmdd = args.yyyymmdd
        
        # 21/02/04
        self.batch_size = int(config.find("Predict", "batch.size"))
        
        # 21/02/16
        self.target_scheme = config.find("Target", "target.scheme.name")
        self.target_table = config.find("Target", "target.table.name")
 
    def setup_args(self, args: argparse.Namespace):
        
        self.model_id = args.model_id
        if args.train_date:
            self.train_date = args.train_date
            self.logger_debug.debug(f"from args self.train_date == {self.train_date}")
        else:
            self.train_date = self.__get_train_date(self.config)
            if not self.train_date:    # 21/02/09
                self.train_date = datetime.datetime.now().strftime("%Y%m%d")     # 21/02/09
            self.logger_debug.debug(f"from conf self.train_date == {self.train_date}")
            
        self.yyyymmdd = args.yyyymmdd
        self.model_data_path = self.config.find("Server", "fit.data").format(
            model_id=self.model_id, 
            yyyymmdd=self.train_date
        )
        self.model_path = self.config.find("Server", "serialized.model").format(
            model_id=self.model_id, 
            train_date=self.train_date
        )
        self.input_data_path = self.config.find("Server", "predict.data").format(
            model_id=self.model_id, 
            yyyymmdd=self.yyyymmdd
        )
    
    def main(self):
        """
        todo: refactor into MainEnvironment class...
        """
        
        while self.__in_step():

            ret: pd.DataFrame = pd.DataFrame()
            for model_id, arg_config in self.ARGS_DICT.items():

                # todo: 테스트용 임시처리, 나중에 삭제
    #             arg_config.yyyymmdd = "20210112"

                # 
                # 명령행 매개변수로 넘겨받은 값을 현재 환경에 반영합니다.
                self.setup_args(arg_config)

                prediction = self.run_prediction()['Y_pred']

                if isinstance(prediction, pd.DataFrame):
                    ret = pd.concat([ret, prediction], join='outer', axis=1)
                elif isinstance(prediction, dict):
                    for model_name, model_prediction in prediction.items():
                        ret = pd.concat([ret, model_prediction], join='outer', axis=1)

    #                     self.__reset_step()

            self.logger_debug.debug(f"ret")
            self.logger_debug.debug(f"columns == {ret.columns}")
            self.logger_debug.debug(ret)

            self.logger_debug.debug(f"to DB")
            ret = self.convert_to_etl_format(ret)
            self.logger_debug.debug(f"columns == {ret.columns}")
            self.logger_debug.debug(ret)

    #                 # todo: 나중엔 피클 저장 필요없으니 지우도록.
    #                 if not os.path.exists(os.path.dirname(result_pkl_path)):
    #                     os.makedirs(os.path.dirname(result_pkl_path))
    #                 with open(result_pkl_path.replace('.pickle', f'_{self.current_step}.pickle'), 'wb') as f:
    #                     pickle.dump(ret, f, pickle.HIGHEST_PROTOCOL)

            # todo:
            self.insert_into_result(ret)    # todo: 나중엔 다시 복원시키도록

            self.__progress_step()

        return 0
    
    def convert_to_etl_format(self, df: pd.DataFrame):
        """
        todo: refactor into MainEnvironment class...
        """

        ret = df.rename(      # todo: 가라로 하드코딩함.. 나중에 설정 파일로 빼자...
            columns={
#                 'OFMN': 'OFMN',

                'mlp': 'FIRS_PDMG_DNGR_GRDV',
                'rf': 'SETI_PDMG_DNGR_GRDV',
                'xgb': 'THD_PDMG_DNGR_GRDV',
                'ensemble': 'PDMG_GNRZ_DNGR_GRDV',

                'hdmg_index_hdmg': 'HDMG_PROP_RSKV',    # 이게 변경후 칼럼
                'hdmg_grade': 'HDMG_DNGR_GRDV',

                'occr_index_fire': 'FIRE_OCCR_PROP_RSKV',
                'occr_grade': 'FIRE_OCCR_DNGR_GRDV',
            }
        )

    #     # todo: 나중에 지울 것... 가라 입력용
    #     df['HDMG_GNRZ_DNGR_GRDV'] = df['PDMG_GNRZ_DNGR_GRDV']
    #     df['FIRS_HDMG_DNGR_GRDV'] = df['FIRS_PDMG_DNGR_GRDV']
    #     df['SETI_HDMG_DNGR_GRDV'] = df['SETI_PDMG_DNGR_GRDV']
    #     df['THD_HDMG_DNGR_GRDV'] = df['THD_PDMG_DNGR_GRDV']

#         condition_on_rows = (df.index == df['hdmg_OFMN']) & (df['hdmg_OFMN'] == df['occr_OFMN'])

        ret['OFMN'] = ret.index
        ret['BASE_DE_CD'] = self.yyyymmdd     # todo: AN_STDE(분석기준일자: 이름 변경됨) vs ETL_LSDE(적재기준일자) 
        ret['AN_TOLN'] = 'PYTHON'            # todo: 브라이틱스쪽 결과물은 BRIGHTICS 로
        ret['ETL_LPID'] = f'{"PYTH"}_{self.target_table}'    # TB_MVD_OFF_RSAN_INDX
        ret['ETL_LSDE'] = datetime.datetime.now().strftime('%Y%m%d')
        ret['ETL_LDDT'] = datetime.datetime.now()

    #     df = df[[
    #         'OFMN', 'AN_STDE', 'AN_TOLN',
    #         'FIRE_OCCR_DNGR_GRDV', 'FIRE_OCCR_PROP_RSKV',
    #         'PDMG_GNRZ_DNGR_GRDV', 'FIRS_PDMG_DNGR_GRDV', 'SETI_PDMG_DNGR_GRDV', 'THD_PDMG_DNGR_GRDV',
    #         'HDMG_DNGR_GRDV', 'HDMG_PROP_RSKV',
    #         'ETL_LPID', 'ETL_LSDE', 'ETL_LDDT'
    #     ]]
#         ret.columns.duplicated()
        self.logger_debug.debug(f"ret.columns.duplicated() == {ret.columns.duplicated()}")
        ret = ret.loc[:, ~ret.columns.duplicated()]
        
    

    #     df = df.loc[condition_on_rows, :]    # 

        return ret[[
                    'OFMN', 'BASE_DE_CD', 'AN_TOLN',
                    'FIRE_OCCR_DNGR_GRDV', 'FIRE_OCCR_PROP_RSKV',
                    'PDMG_GNRZ_DNGR_GRDV', 'FIRS_PDMG_DNGR_GRDV', 'SETI_PDMG_DNGR_GRDV', 'THD_PDMG_DNGR_GRDV',
                    'HDMG_PROP_RSKV', 'HDMG_DNGR_GRDV',
                    'ETL_LPID', 'ETL_LSDE', 'ETL_LDDT'
                ]]
    
    # 각 지표별 예측 결과를 반환하는 함수입니다.
    def run_prediction(self) -> None:
        """
        
        """
        
        # fixme: 직렬화 관련 처리추가 (직렬화된 모델 객체가 있으면, 그것을 사용하도록)
        if not os.path.exists(self.model_path):    # Fixme: True
            
            fit_data = self.load_data(use_for='fit', base_de_cd=self.train_date)
            fit_data_preprocessed = self.preprocess_data(fit_data)    # todo: 데이터에 따라 다른 전처리 수행하도록
            
            self.fit(fit_data_preprocessed, self.model_path)
            
        else:
            self.load_model()

        pred_data = self.load_data(use_for='predict', base_de_cd=self.yyyymmdd, step_no=self.current_step, batch_size=self.batch_size)
        pred_data_preprocessed = self.preprocess_data(pred_data)  # todo: 데이터에 따라 다른 전처리 수행하도록

        predicted_result = self.predict(pred_data_preprocessed)
        
#         # fixme: AN_STDE (분석기준일자) 칼럼 값을 여기서 채우도록 했음...
#         predicted_result['AN_STDE'] = self.train_date

        return predicted_result
        
    def predict(self, data_preprocessed: dict):
        
        predict_options = {"key": self.model_id}
        predict_options.update(data_preprocessed)
        self.logger_debug.debug(f"predict_options == {predict_options}")
        
        predicted = self.modeler.predict(**predict_options)
        self.logger_debug.debug(f"data prediction: {predicted}")
        
        return {
            "INPUT": data_preprocessed,
            "Y_pred": predicted
        }
        
    def load_model(self):
        self.modeler.deserialize(self.model_id, self.model_path)

    def fit(self, data_preprocessed: dict, model_path: str):
        
        fit_options = {"key": self.model_id}
        fit_options.update(data_preprocessed)
        self.logger_debug.debug(f"fit_options == {fit_options}")
        
        fitted_model = self.modeler.fit(**fit_options)
        self.logger_debug.debug(f"model fitted: {fitted_model}")
        
        model_serialization_success: bool = self.modeler.serialize_to_pickle(self.model_id, self.model_path)
        
    def preprocess_data(self, data: pd.DataFrame) -> dict:
        """
        불러온 데이터를 전처리 합니다.
        """
        
        data_preprocessed = Preprocessing.run(self.model_id, data)
        
        # for-debugging : 콘솔에 출력합니다.
        self.logger_debug.debug(f"data_preprocessed == {data_preprocessed}")
        
        return data_preprocessed
        
    def load_data(self, use_for: str, base_de_cd: str, step_no: int = -1, batch_size: int = -1):
        """
        데이터를 로드합니다.
        todo: use_for 값에 따라 불러와야 할 데이터를 판단합니다.
        """
        
        data_path: str = self.__get_data_path(use_for)
        
        self.logger_debug.debug(f"{os.path.exists(data_path)} : {use_for}_data_path == {data_path}")
        if not os.path.exists(data_path):
            # 요청받은 학습 일자에 대해 이미 실행된 적이 있으면 수행하지 않는 블록입니다.

            # D/B 에 접속합니다.
            connection_success: bool = self.database_manager.connect_database()
            self.logger_debug.debug(f"D/B has been connected: {connection_success}")

            # 요청받은 모델 ID 에 따라 조회문을 선택하여 요청합니다.
            data = self.database_manager.execute_sql(self.model_id, use_for, base_de_cd)
            self.DATA_DICT[use_for][self.model_id].update({base_de_cd: data})    # 21/02/08
            self.logger_debug.debug(f"data == {data}")

            # 체결된 접속을 해제합니다.
            disconnection_success: bool = self.database_manager.disconnect_database()
            self.logger_debug.debug(f"D/B has been disconnected: {disconnection_success}")

            # 메모리에 fetch 한 테이블 객체를 지정된 경로에 직렬화하여 저장합니다.
            serialization_success: bool = self.database_manager.serialize_to_pickle(
                data_frame=data, 
                dst_path=data_path
            )
            self.logger_debug.debug(f"Serialization into .pickle file: {serialization_success}\t{data_path}")

        data = self.DATA_DICT[use_for][self.model_id].get(base_de_cd, None)    # 21/02/08
        if data is None:
            # 직렬화된 .pickle 전처리 데이터를 역직렬화하여 메모리로 불러옵니다.
            self.logger_debug.debug(f"os.getcwd(): {os.getcwd()}")
            data: pd.DataFrame = self.database_manager.deserialize_from_pickle(data_path)
            self.DATA_DICT[use_for][self.model_id].update({base_de_cd: data})    # 21/02/08

            # for-debugging : 콘솔에 출력합니다.
            self.logger_debug.debug(f"raw deserialized: {data is not None}")
            self.logger_debug.debug(f"{use_for}_data == {data}")
        
        # todo: step_no 지정되지 않았을 경우에 대한 분기 추가
        if step_no == -1:
            
            return data
        
        else:
            idx_from = step_no * batch_size
            idx_to = (step_no + 1) * batch_size - 1
            self.logger_debug.debug(f"idx_from == {idx_from} / idx_to == {idx_to}")
            
            return data.loc[idx_from: idx_to, :]
    
    def insert_into_result(self, df: pd.DataFrame):
        """
        todo: 예측결과를 D/B 로 적재하는 동작입니다.
        """
        
        # D/B 에 접속합니다.
        connection_success: bool = self.database_manager.connect_database()
        self.logger_debug.debug(f"D/B has been connected: {connection_success}")

        # 21.02.04 조건절 해제
#         # todo: 데이터 적재 조건
#         occr_existence = df['FIRE_OCCR_DNGR_GRDV'].isna().apply(lambda boolean: not boolean)
#         pdmg_existence = df['PDMG_GNRZ_DNGR_GRDV'].isna().apply(lambda boolean: not boolean)
#         hdmg_existence = df['HDMG_DNGR_GRDV'].isna().apply(lambda boolean: not boolean)
#         filter_condition = occr_existence & pdmg_existence & hdmg_existence
#         df = df.loc[filter_condition, :]
        
        # INSERT 문을 작성합니다.
        colnames = df.columns.tolist() # TB_MVD_OFF_RSAN_INDX
        sql = f"""
        INSERT INTO {self.target_scheme}.{self.target_table} ({', '.join(colnames)})
        VALUES ({', '.join(['?' for i in range(len(colnames[:-1]))])}, SYSDATE)"""
        
        # 적재할 데이터를 배열로 정리합니다.
        data_array = list(map(lambda row: row[:-1], df.values.tolist()))
        
        # INSERT 문과 적재할 데이터 배열을 주고 실행합니다.
        self.logger_debug.debug(f"sql == {sql}")
        self.logger_debug.debug(f"colnames == {colnames}")
        self.logger_debug.debug(f"data_array[:3] == {data_array[:3]}")
        self.database_manager.executemany(
            sql=sql,
            data_array=data_array
        )        

        # 체결된 접속을 해제합니다.
        disconnection_success: bool = self.database_manager.disconnect_database()
        self.logger_debug.debug(f"D/B has been disconnected: {disconnection_success}")
        
    # 21/02/08
    def __get_count(self, base_de_cd: str):
        
        # D/B 에 접속합니다.
        connection_success: bool = self.database_manager.connect_database()
        self.logger_debug.debug(f"D/B has been connected: {connection_success}")
        
        # SELECT COUNT 결과 값을 가져옵니다.
        count = self.database_manager.get_data_count_total(base_de_cd)
        self.logger_debug.debug(f"count ({type(count)}) =  {count}")
        
        # 체결된 접속을 해제합니다.
        disconnection_success: bool = self.database_manager.disconnect_database()
        self.logger_debug.debug(f"D/B has been disconnected: {disconnection_success}")
        
        return count
    
    def __get_train_date(self, config: ApplicationConfiguration, model_name: str = ""):
        
        with open(config.find("Model", "definition.json"), "rt") as io_wrapper:
            model_config = json.load(io_wrapper)
        
        if model_name:
            conf = model_config[self.model_id][model_name]
        else:
            conf = list(model_config[self.model_id].values())[0]
        
        return conf["train_date"]
    
    def __get_data_path(self, use_for: str):
        
        data_path = {
            "fit": self.model_data_path,
            "predict": self.input_data_path
        }.get(use_for, None)
        
        if not data_path:
            raise KeyError
            
        return data_path
