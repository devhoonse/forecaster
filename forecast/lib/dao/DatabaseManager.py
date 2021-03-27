import os
import pickle

import jaydebeapi
import pandas as pd

from ..common.SingletonInstance import SingletonInstance
from ..conf.ApplicationConfiguration import ApplicationConfiguration
from ..logger.LogHandler import LogHandler


class DatabaseManager(SingletonInstance):
    def init(self, config: ApplicationConfiguration):
        """
        생성자입니다.
        """
        
        self.__logger_debug = LogHandler.instance().get_logger('debug-logger')
        self.__logger_error = LogHandler.instance().get_logger('error-logger')
        
        self.__db_conf_path = config.find("Database", "connection.conf")
        self.__sql_path = config.find("Source", "query.directory")
        
        self.__logger_debug.debug(f"self.__db_conf_path == {self.__db_conf_path}")
        self.__logger_debug.debug(f"self.__sql_path == {self.__sql_path}")
        
        self.__db_conf = dict()
        self.__sql_map = dict()
        
        # 21/02/08
        self.__data_count_sql_path = os.path.join(self.__sql_path, 'data_count.sql')
        with open(self.__data_count_sql_path, 'r') as io_wrapper:
            self.__data_count_sql = io_wrapper.read()
            
        # 21/02/19
        self.__data_last_de_src_sql_path = os.path.join(self.__sql_path, 'data_count.sql')
        with open(self.__data_last_de_src_sql_path, 'r') as io_wrapper:
            self.__data_last_de_src_sql = io_wrapper.read()
        
        # 21/02/19
        
        
        self.__data_count = dict()
        
        self.__conn = None
        
        self.get_db_connection_conf()
        self.get_sql_map()
        
        self.__logger_debug.debug(f"<START> connect_database()")

    def get_db_connection_conf(self) -> dict:
        """
        """
        if self.check_db_conf():
            self.__read_db_conf()
        self.__logger_debug.debug(f"self.__db_conf == {self.__db_conf}")
        return self.__db_conf
        
    def get_sql_map(self) -> dict:
        if self.check_sql_map():
            self.__build_sql_map()
        self.__logger_debug.debug(f"self.__sql_map == {self.__sql_map}")
        return self.__sql_map
        
    def check_db_conf(self) -> bool:
        self.__logger_debug.debug(f"self.__db_conf == {self.__db_conf}")
        return self.__db_conf == dict()
        
    def check_sql_map(self) -> bool:
        self.__logger_debug.debug(f"self.__sql_map == {self.__sql_map}")
        return self.__sql_map == dict()
    
    @property
    def is_connected(self) -> bool:
#         self.__logger_debug.debug(f"self.__conn is not None == {self.__conn is not None}")
        return self.__conn is not None
    
    def __read_db_conf(self):
        
        # import DB configuration
        with open(self.__db_conf_path, 'r') as io_wrapper:
            db_conf_py = io_wrapper.read()
        exec(db_conf_py, locals())
        
        # assign config data to protected member variable
        self.__db_conf = locals()['db_conf']
        self.__logger_debug.debug(f"self.__db_conf == {self.__db_conf}")
        
    def __build_sql_map(self):
        """
        """
        self.__logger_debug.debug(f"os.getcwd() == {os.getcwd()}")
        self.__logger_debug.debug(f"self.__db_conf == {self.__db_conf}")
        
        for dirname in os.listdir(self.__sql_path):
            
            dirpath: str = os.path.join(self.__sql_path, dirname)
            self.__logger_debug.debug(f"dirpath == {dirpath}")
            if not os.path.isdir(dirpath):
                continue
            
            self.__sql_map.update({dirname: dict()})
            for filename in os.listdir(dirpath):
                self.__logger_debug.debug(f"filename == {filename}")
                
                name: str = os.path.splitext(filename)[0]
                extension = os.path.splitext(filename)[-1].lower()
                if extension != '.sql':
                    continue
                    
                filepath: str = os.path.join(dirpath, filename)
                self.__logger_debug.debug(f"filepath == {filepath}")
                    
                with open(filepath, 'r') as io_wrapper:
                    self.__sql_map[dirname].update({name: io_wrapper.read()})
            
#             name: str = os.path.splitext(filename)[0]
#             extension: str = os.path.splitext(filename)[-1].lower()
#             self.__logger_debug.debug(f"name == {name}")
#             self.__logger_debug.debug(f"extension == {extension}")
            
#             if extension != '.sql':
#                 continue
                
#             filepath: str = os.path.join(self.__sql_path, filename)
#             self.__logger_debug.debug(f"filepath == {filepath}")
            
#             with open(filepath, 'r') as io_wrapper:
#                 self.__sql_map.update({name: io_wrapper.read()})
#                 self.__logger_debug.debug(f"__sql_map[{name}] == {self.__sql_map[name]}")

    def get_sql(self, key: str, use_for: str) ->  str:
        """
        SQL 문을 들고 옵니다.
        적절하지 않은 키를 요청받으면 빈 문자열을 반환합니다.
        """
        if self.check_sql_map():
            self.__build_sql_map()
        self.__logger_debug.debug(f"key == {key}")
        self.__logger_debug.debug(f"self.__sql_map.get({use_for}, dict()).get({key}, None) == {self.__sql_map.get(use_for, dict()).get(key, None)}")
        
        return self.__sql_map.get(use_for, dict()).get(key, None)
    
    def connect_database(self) -> bool:
        """
        """
        
        if self.is_connected:
            return True
        
        self.__logger_debug.debug(f"<START> connect_database()")
        
        # get JDBC Connection
        try:
            self.__conn = jaydebeapi.connect(
                self.__db_conf['DRV'], 
                self.__db_conf['URL'], 
                [self.__db_conf['ID'], self.__db_conf['PWD']], 
                self.__db_conf['JAR']
            )
        except:
            self.__logger_debug.error(f"FAIL")
            self.__logger_error.error(f"FAIL")
            return False
        else:
            self.__logger_debug.debug(f"<SUCCESS> CONNECTED")
            return True
        
    def disconnect_database(self) -> bool:
        """
        """
        
        if not self.is_connected:
            return True    # todo: 이거 이상해.
        
        # release JDBC Connection
        try:
            self.__conn = self.__conn.close()
        except:
            self.__logger_debug.error(f"FAIL")
            self.__logger_error.error(f"FAIL")
            return False
        else:
            self.__logger_debug.debug(f"<SUCCESS> Disconnected")
            return True

#     def insert_df_to_dbtable(self, df: pd.DataFrame):
#         """
#         todo: fill the logic
#         """
        
#         cursor = self.__conn.cursor()
        
#         cursor.executemany(f"""
#             INSERT INTO {}.{} 
#             ({}) 
#             VALUES 
#             ({})
#         """, df.values.tolist())
        
#         cursor.close()
        
    # 21/02/08
    def get_data_count_total(self, base_de_cd: str):
        """
        """
        
        data_count_total = self.__data_count.get(base_de_cd, None)
        
        if data_count_total is None:
        
            sql = self.__data_count_sql.format(BASE_DE_CD=base_de_cd)
            self.__logger_debug.debug(f"sql == {sql}")

            cursor = self.__conn.cursor()
            cursor.execute(sql)

            description = cursor.description
            header = list(map(lambda subscriptable : subscriptable[0], description))
            data = cursor.fetchall()

            cursor.close()

            data_frame = pd.DataFrame(data)
            self.__logger_debug.debug(f"data_frame == {data_frame}")
            data_frame.columns = header
            
            self.__data_count.update({base_de_cd: data_frame.iloc[0, 0]})
            
            data_count_total = self.__data_count.get(base_de_cd, None) 
        
        return data_count_total
    
    def execute(self, sql: str):
        self.__logger_debug.debug(f"<START> execute with sql=={sql}")
        
        if not self.is_connected:
            return None
        
        cursor = self.__conn.cursor()
        cursor.execute(sql)
    
    def execute_select(self, sql: str):
        self.__logger_debug.debug(f"<START> execute with sql=={sql}")
        
        if not self.is_connected:
            return None
        
        cursor = self.__conn.cursor()
        cursor.execute(sql)
        
        description = cursor.description
        header = list(map(lambda subscriptable : subscriptable[0], description))
        data = cursor.fetchall()
        
        cursor.close()
        
        data_frame = pd.DataFrame(data)
        self.__logger_debug.debug(f"data_frame == {data_frame}")
        data_frame.columns = header
        
        return data_frame
        
    
    def executemany(self, sql: str, data_array: list):
        """
        sql 문에 명시된 배치단위 CRUD 를 수행합니다.
        """
        
        self.__logger_debug.debug(f"<START> executemany_sql with sql=={sql}")
        
        if not self.is_connected:
            return None
        
        cursor = self.__conn.cursor()
        cursor.executemany(sql, data_array)
        cursor.close()
        
        self.__logger_debug.debug(f"<FINISH> executemany_sql with sql=={sql}")
    
    def execute_sql(self, key: str, use_for: str, base_de_cd: str):
        """
        todo: RENAME
        요청받은 데이터를 조회하고 dataframe 객체로 반환합니다.
        key: 지표명 ( hdmg=인명피해 / pdmg=재산피해 / occr=화재발생위험 )
        use_for: 용도 ( fit=학습용 / predict=예측용 )
        base_de_cd: 마트에서 가져올 데이터의 기준일자 값. ( yyyymmdd )
        """
        
        self.__logger_debug.debug(f"<START> execute_sql with key=={key} / use_for=={use_for}")
        
        sql = self.get_sql(key, use_for)
        if sql is None:
            self.__logger_debug.debug(f"self.__sql_map == {self.__sql_map}")
            self.__logger_debug.debug(f"self.__sql_map.keys() == {self.__sql_map.keys()}")
            raise KeyError(f".sql file Not Found with key=={key} / use_for=={use_for}")
        
        if not self.is_connected:
            return None
        
        self.__logger_debug.debug(f"sql.format(BASE_DE_CD=base_de_cd) == {sql.format(BASE_DE_CD=base_de_cd)}")
        
        cursor = self.__conn.cursor()
        cursor.execute(sql.format(BASE_DE_CD=base_de_cd))
        
        description = cursor.description
        header = list(map(lambda subscriptable : subscriptable[0], description))
        data = cursor.fetchall()
        
        cursor.close()
        
        data_frame = pd.DataFrame(data)
        self.__logger_debug.debug(f"data_frame == {data_frame}")
        data_frame.columns = header
        
        return data_frame
    
    def serialize_to_pickle(self, data_frame: pd.DataFrame, dst_path: str) -> bool:
        """
        """
        
        self.__logger_debug.debug(f"<START> serialize_to_pickle")
        
        if not os.path.exists(os.path.dirname(dst_path)):
            os.makedirs(os.path.dirname(dst_path))
        
        try:
            with open(dst_path, 'wb') as io_wrapper:
                pickle.dump(data_frame, io_wrapper, pickle.HIGHEST_PROTOCOL)
        except:
            self.__logger_debug.error(f"FAIL")
            self.__logger_error.error(f"FAIL")
            return False
        else:
            self.__logger_debug.debug(f"<SUCCESS> SAVED TO >> {dst_path}")
            return True
    
    def deserialize_from_pickle(self, src_path: str) -> pd.DataFrame:
        """
        """
        
        self.__logger_debug.debug(f"<START> deserialize_from_pickle")
        
        if not os.path.exists(src_path):
            return None
        
        try:
            with open(src_path, 'rb') as io_wrapper:
                data = pickle.load(io_wrapper)
        except:
            self.__logger_debug.error(f"FAIL")
            self.__logger_error.error(f"FAIL")
            return None
        else:
            self.__logger_debug.debug(f"<SUCCESS> LOADED FROM << {src_path}")
            return data
        