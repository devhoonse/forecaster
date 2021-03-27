import os
import pickle
import json
import re
import math
from collections import OrderedDict
from typing import Optional, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.sparse
from lifelines import CoxPHFitter
from lifelines import utils
from lifelines.exceptions import StatError
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import NotFittedError
from matplotlib import pyplot as plt
from joblib import parallel_backend

from .common.SingletonInstance import SingletonInstance
from .logger.LogHandler import LogHandler
from .conf.ApplicationConfiguration import ApplicationConfiguration
# from ..conf.model import *


def fitmethod(func):
    
    def fitmethod(*args, **kwargs):
        
        REGEX = r'^_fit_(.*)[_](.*)'
        model_id, model_name = re.search(REGEX, func.__name__).groups()
        kwargs.update({
            "model_id": model_id,
            "model_name": model_name
        })
        
        return func(*args, **kwargs)
    
    return fitmethod


def predictmethod(func):
    
    def predictmethod(*args, **kwargs):
        
        REGEX = r'^_predict_(.*)[_](.*)'
        model_id, model_name = re.search(REGEX, func.__name__).groups()
        kwargs.update({
            "model_id": model_id,
            "model_name": model_name
        })
        
        return func(*args, **kwargs)
    
    return predictmethod


class ModelManager(SingletonInstance):

    __logger_debug = LogHandler.instance().get_logger('debug-logger')
    __logger_error = LogHandler.instance().get_logger('error-logger')
    
    __model_config = dict()
    __model = dict()
    
    __model_path = ""
    
    def __init__(self):
        pass
    
    def init(self, config: ApplicationConfiguration):
        
        self.__logger_debug.debug(f"<START> ModelManager.init( )")
        
        # 배치 사이즈 설정 추가 : 21/02/04
        self.batch_size = int(config.find("Predict", "batch.size"))
        
        with open(config.find("Model", "definition.json"), "rt") as io_wrapper:
            config = json.load(io_wrapper)
        self.__class__.__model_config = config
        self.__logger_debug.debug(f"models config == {config}")
                
        for key, option in config.items():
            builder = getattr(self.__class__, f'_build_{key}')
            setattr(self, f'__{key}', builder(**option))
            self.__logger_debug.debug(f"self.__{key} == {getattr(self, f'__{key}')}")
                
    def serialize_to_pickle(self, key: str, dst_path: str) -> bool:
        """
        todo: pickling DesignInfoMatrix is not supported yet..
        """
        
        self.__logger_debug.debug(f"<START> serialize_to_pickle ({key}, {dst_path})")
        self.__logger_debug.debug(f"self.__class__.__model[key] == {self.__class__.__model[key]}")
        
        if not os.path.exists(os.path.dirname(dst_path)):
            os.makedirs(os.path.dirname(dst_path))
        
        try:
            with open(dst_path, 'wb') as io_wrapper:
                pickle.dump(
#                     self.__class__.__model[key], 
                    getattr(self, f'__{key}', dict()),
                    io_wrapper,
#                     pickle.HIGHEST_PROTOCOL
                )
        except Exception as e:
            self.__logger_debug.error(f"{str(e)}")
            self.__logger_debug.error(f"FAIL")
            self.__logger_error.error(f"FAIL")
            os.remove(dst_path)
            return False
        else:
            # for debugging 
#             self.__logger_debug.debug(f"self.__class__.__model[key]['mlp']['model'].coefs_ == {self.__class__.__model[key]['mlp']['model'].coefs_}")
            
            self.__logger_debug.debug(f"<SUCCESS> SAVED TO >> {dst_path}")
            return True
    
    def deserialize(self, key: str, src_path: str) -> bool:
        
        self.__logger_debug.debug(f"<START> serializing {key} from {src_path}")
        
        if not os.path.exists(src_path):
            return None
        
        try:
            with open(src_path, 'rb') as io_wrapper:
#                 self.__class__.__model.update({key: pickle.load(io_wrapper)})
                setattr(self, f'__{key}', pickle.load(io_wrapper))
        except:
            self.__logger_debug.error(f"FAIL")
            self.__logger_error.error(f"FAIL")
            return False
        else:
            self.__logger_debug.debug(f"<SUCCESS> LOADED FROM << {src_path}")
            self.__logger_debug.debug(f"getattr(self, f'__{key}') == {getattr(self, f'__{key}')}")
            # for debugging 
#             self.__logger_debug.debug(f"self.__class__.__model[key]['mlp']['model'].coefs_ == {self.__class__.__model[key]['mlp']['model'].coefs_}")
#             self.__logger_debug.debug(f"self.__class__.__model[key]['mlp'].check_is_fitted() == {self.__class__.__model[key]['mlp'].check_is_fitted()}")
            return True
    
    def predict(self, key: str, apply_ensemble: bool = True, *args, **kwargs):
        
        self.__logger_debug.debug(f"<START> predict with {kwargs}")
        self.__logger_debug.debug(f"__model == {self.__model}")
        
        
        ret = dict()
        for classname in self.__model_config[key].keys():
            ret.update(
                {classname: self.__predict(key, classname, *args, **kwargs)}
            )
#         self.__class__.__model.update({key: ret})    
    
        if len(ret) > 1 and apply_ensemble:
            ret.update({"ensemble": self.ensemble(ret)})
            ret = self.ensemble(ret)
            return ret
        return ret
    
    def __predict(self, key: str, classname: str, *args, **kwargs):
        
        kwargs.update(self.__model_config[key][classname]['predict'])
        
        self.__logger_debug.debug(f"<START> __predict with {kwargs}")
        
        predictor = getattr(self, f'_predict_{key}_{classname}')
        
        return predictor(self, *args, **kwargs)
    
    def ensemble(self, prediction: dict):
        self.__logger_debug.debug(f"<START> ensemble with {prediction}")
        
#         prediction_df = pd.DataFrame(prediction)    # todo: 여기 좀 손봐야할지도...
        prediction_df = pd.concat(prediction.values(), axis=1)
        
        self.__logger_debug.debug(f"prediction == {prediction}")
        
        prediction_df["ensemble"] = prediction_df.apply(
            lambda votes: self.__choose_from(votes), 
            axis=1
        )
        return prediction_df
    
    @staticmethod
    def __choose_from(votes):
        """
        todo: 로직 정리
        """
        counter = votes.value_counts(dropna=False)
        counter_maximums = counter[counter == counter.max(skipna=True)]
        counter_maximum_idxs = counter_maximums.index
        
        nof_maximums = counter_maximums.shape[0]
        
        if nof_maximums == 1:
            # case 1 : 1 1 2
            # case 2 : 1 1 1
            return counter_maximum_idxs[0]
        elif nof_maximums > 1:
            # case : 1 2 3
            # case : 1 3 4
            return math.ceil(np.median(counter_maximum_idxs.values))    # todo: median ?
        else:
            raise Exception(f"Cannot Determine the Vote with {row}")
    
    @predictmethod
    def _predict_occr_CoxPHFitter(self, *args, **kwargs):
        
        self.__logger_debug.debug(f"<START> _fit_occr_CoxPHFitter( )")
        
        model_id = kwargs.pop('model_id')
        model_name = kwargs.pop('model_name')
        
        self.__logger_debug.debug(f"model_id == {model_id}")
        self.__logger_debug.debug(f"model_name == {model_name}")
        
        data_params = self.__class__.__model_config[model_id][model_name]["data_params"]
        assert set(data_params).issubset(set(kwargs.keys()))
        
        data = kwargs['df']
        
        self.__logger_debug.debug(f"<START> _predict_occr_CoxPHFitter")
        self.__logger_debug.debug(f"args = {args}")
        self.__logger_debug.debug(f"kwargs = {kwargs}")
        
        model = self.get_model(model_id, model_name)
        self.__logger_debug.debug(f"model == {model}")
        
        self.__logger_debug.debug(f"data.columns == {data.columns}")
        self.__logger_debug.debug(f"'BDNG_RF_SE_DUMMY' in data.columns == {'BDNG_RF_SE_DUMMY' in data.columns}")
        
        self.__logger_debug.debug(f"predicting ... with {kwargs}")
        test = model.predict_cumulative_hazard(data)
        data['index_fire'] = data.index.to_series().apply(lambda x : test.loc[0,x])
        self.__logger_debug.debug(f"data['index_fire'].dtype == {data['index_fire'].dtype}")
        
        # todo: 구간화
        # 변경 전 로직
#         q = kwargs['grade']['q']
#         labels = kwargs['grade']['labels']
#         data['grade'] = pd.qcut(data['index_fire'], 
#                                 q=q, 
#                                 labels=labels,
#                                 duplicates="drop")
        # 변경 후 로직
        cutters = kwargs['grade']['cutters']
        labels = kwargs['grade']['labels']
        data['grade'] = pd.cut(data['index_fire'], 
                               [-np.inf] + cutters + [np.inf],
                               labels=False
                              )
        
        data['grade'] = data['grade'].apply(lambda grade: int(grade + 1))
        self.__logger_debug.debug(f"data['index_fire'].describe() == {data['index_fire'].describe()}")
        self.__logger_debug.debug(f"data['grade'].value_counts() == {data['grade'].value_counts()}")
        
        data['OFMN'] = data.index
        data.rename(
            columns={
                'OFMN': f'{model_id}_OFMN',
                'index_fire': f'{model_id}_index_fire',
                'grade': f'{model_id}_grade'
            }, 
            inplace=True
        )
        
        return data[[f'{model_id}_OFMN', f'{model_id}_index_fire', f'{model_id}_grade']]
    
    @predictmethod
    def _predict_hdmg_HdmgHeuristic(self, *args, **kwargs):
        
        self.__logger_debug.debug(f"<START> _fit_occr_CoxPHFitter( )")
        
        model_id = kwargs.pop('model_id')
        model_name = kwargs.pop('model_name')
        
        self.__logger_debug.debug(f"model_id == {model_id}")
        self.__logger_debug.debug(f"model_name == {model_name}")
        
        data_params = self.__class__.__model_config[model_id][model_name]["data_params"]
        assert set(data_params).issubset(set(kwargs.keys()))
        
        data = kwargs['df']
        
        self.__logger_debug.debug(f"<START> _predict_occr_CoxPHFitter")
        self.__logger_debug.debug(f"args = {args}")
        self.__logger_debug.debug(f"kwargs = {kwargs}")
        
        model = self.get_model(model_id, model_name)
        self.__logger_debug.debug(f"model == {model}")
        
        self.__logger_debug.debug(f"data.columns == {data.columns}")
        self.__logger_debug.debug(f"'BDNG_RF_SE_DUMMY' in data.columns == {'BDNG_RF_SE_DUMMY' in data.columns}")
        
        self.__logger_debug.debug(f"predicting ... with {kwargs}")
#         test = model.predict_cumulative_hazard(data)
        data['index_hdmg'] = model.predict(data)    # 21/02/09
        self.__logger_debug.debug(f"data['index_hdmg'].dtype == {data['index_hdmg'].dtype}")
        
        # todo: 구간화
        # 변경 전 로직
#         q = kwargs['grade']['q']
#         labels = kwargs['grade']['labels']
#         data['grade'] = pd.qcut(data['index_fire'], 
#                                 q=q, 
#                                 labels=labels,
#                                 duplicates="drop")
        # 변경 후 로직
        cutters = kwargs['grade']['cutters']
        labels = kwargs['grade']['labels']
        data['grade'] = pd.cut(data['index_hdmg'], 
                               [-np.inf] + cutters + [np.inf],
                               labels=False
                              )
        
        data['grade'] = data['grade'].apply(lambda grade: int(grade + 1))
        self.__logger_debug.debug(f"data['index_hdmg'].describe() == {data['index_hdmg'].describe()}")
        self.__logger_debug.debug(f"data['grade'].value_counts() == {data['grade'].value_counts()}")
        
        data['OFMN'] = data.index
        data.rename(
            columns={
                'OFMN': f'{model_id}_OFMN',
                'index_hdmg': f'{model_id}_index_hdmg',
                'grade': f'{model_id}_grade'
            }, 
            inplace=True
        )
        
        return data[[f'{model_id}_OFMN', f'{model_id}_index_hdmg', f'{model_id}_grade']]
    
    @predictmethod
    def _predict_pdmg_mlp(self, *args, **kwargs):
        return self.__predict_pdmg_classifier(*args, **kwargs)
    
    @predictmethod
    def _predict_pdmg_rf(self, *args, **kwargs):
        return self.__predict_pdmg_classifier(*args, **kwargs)
    
    @predictmethod
    def _predict_pdmg_rfstump(self, *args, **kwargs):
        return self.__predict_pdmg_classifier(*args, **kwargs)
    
    @predictmethod
    def _predict_pdmg_rfensemble(self, *args, **kwargs):
        return self.__predict_pdmg_classifier(*args, **kwargs)
    
    @predictmethod
    def _predict_pdmg_xgb(self, *args, **kwargs):
        self.__logger_debug.debug(f"<START> _fit_occr_CoxPHFitter( )")
        
        model_id = kwargs.pop('model_id')
        model_name = kwargs.pop('model_name')
        
        self.__logger_debug.debug(f"model_id == {model_id}")
        self.__logger_debug.debug(f"model_name == {model_name}")
        
        data_params = self.__class__.__model_config[model_id][model_name]["data_params"]
        assert set(data_params).issubset(set(kwargs.keys()))
        
        X = kwargs['X']
        Y = kwargs['Y']
        
        self.__logger_debug.debug(f"<START> _predict_occr_CoxPHFitter")
        self.__logger_debug.debug(f"args = {args}")
        self.__logger_debug.debug(f"kwargs = {kwargs}")
        
        model_info = self.get_model(model_id, model_name)
        model = model_info['model']
        x_columns = model_info['x_columns']
        self.__logger_debug.debug(f"model == {model}")
#         self.__logger_debug.debug(f"x_columns == {x_columns}")
        
#         self.__logger_debug.debug(f"data.columns == {X.columns}")
        
#         self.__logger_debug.debug(f"column remapping ...")
#         header = pd.DataFrame(
#             OrderedDict(
#                 zip(x_columns, 
#                     [[] for i in range(len(x_columns))]
#                 )
#             )
#         )
#         X = pd.concat([header, X], join='outer')
#         X = X[x_columns]
        self.__logger_debug.debug(f"X == {X}")
        
        self.__logger_debug.debug(f"predicting ...")
        y_pred = model.predict(X, Y)
        
        ret = pd.DataFrame({
            model_name: y_pred
        })
        ret.index = X.index
        ret[model_name] = ret[model_name].apply(lambda grade: int(grade + 1))
        
        return ret
    
    def __predict_pdmg_classifier(self, *args, **kwargs):
        self.__logger_debug.debug(f"<START> _fit_occr_CoxPHFitter( )")
        
        model_id = kwargs.pop('model_id')
        model_name = kwargs.pop('model_name')
        
        self.__logger_debug.debug(f"model_id == {model_id}")
        self.__logger_debug.debug(f"model_name == {model_name}")
        
        data_params = self.__class__.__model_config[model_id][model_name]["data_params"]
        assert set(data_params).issubset(set(kwargs.keys()))
                
        X = kwargs['X']
        
        self.__logger_debug.debug(f"<START> _predict_occr_CoxPHFitter")
        self.__logger_debug.debug(f"args = {args}")
        self.__logger_debug.debug(f"kwargs = {kwargs}")
        
        model_info = self.get_model(model_id, model_name)
        model = model_info['model']
#         x_columns = model_info['x_columns']
#         self.__logger_debug.debug(f"model == {model}")
#         self.__logger_debug.debug(f"x_columns == {x_columns}")
        
#         self.__logger_debug.debug(f"X.columns == {X.columns}")
        
#         self.__logger_debug.debug(f"column remapping ...")
#         header = pd.DataFrame(
#             OrderedDict(
#                 zip(x_columns, 
#                     [[] for i in range(len(x_columns))]
#                 )
#             )
#         )
#         X = pd.concat([header, X], join='outer')
#         X = X[x_columns]
        self.__logger_debug.debug(f"X == {X}")
        self.__logger_debug.debug(f"X.columns == {list(X.columns)}")
        self.__logger_debug.debug(f"X.isna().sum() == {X.isna().sum()}")
        
        self.__logger_debug.debug(f"predicting ...")
        
        with parallel_backend('threading', n_jobs=-1):
            y_pred = model.predict(X)
        
        ret = pd.DataFrame({
            model_name: y_pred
        })
        ret.index = X.index
        ret[model_name] = ret[model_name].apply(lambda grade: int(grade + 1))
        
        return ret
        
    def fit(self, key: str, *args, **kwargs):
        
        self.__logger_debug.debug(f"fitting {key} with {kwargs}")
        
        self.__logger_debug.debug(f"/*BEFORE*/ self.__class__.__model[{key}] == {self.__class__.__model.get(key)}")
        
        ret = dict()
        for classname in self.__model_config[key].keys():
            
            ret.update(
                {classname: self.__fit(key, classname, *args, **kwargs)}
            )
            
        self.__class__.__model.update({key: ret})
        setattr(self, f'__{key}', ret)    # 210128
        self.__logger_debug.debug(f"/*AFTER*/ self.__class__.__model[{key}] == {self.__class__.__model[key]}")
        
        return ret

    def __fit(self, key: str, classname: str, *args, **kwargs):
        
        self.__logger_debug.debug(f"<START> __fit")
        
        self.__logger_debug.debug(f"self.__model_config[{key}][{classname}]['fit'] == {self.__model_config[key][classname]['fit']}")
        kwargs.update(self.__model_config[key][classname]['fit'])    # 모델 설정정보 json 에 적혀있는 값을 가져옴
        self.__logger_debug.debug(f"fitting {key}/{classname} with {kwargs}")
        
        model = self.get_model(key, classname)
        self.__logger_debug.debug(f"model == {model}")
        
        fit = getattr(self, f'_fit_{key}_{classname}')
        self.__logger_debug.debug(f"fit == {fit}")
        model = fit(self, *args, **kwargs)

        return model
    
    @fitmethod
    def _fit_occr_CoxPHFitter(self, *args, **kwargs):
        
        self.__logger_debug.debug(f"<START> _fit_occr_CoxPHFitter( )")
        
        model_id = kwargs.pop('model_id')
        model_name = kwargs.pop('model_name')
        
        self.__logger_debug.debug(f"model_id == {model_id}")
        self.__logger_debug.debug(f"model_name == {model_name}")
        
        data_params = self.__class__.__model_config[model_id][model_name]["data_params"]
        self.__logger_debug.debug(f"data_params == {data_params}")
        assert set(data_params).issubset(set(kwargs.keys()))
        
        self.__logger_debug.debug(f"args == {args}")
        self.__logger_debug.debug(f"kwargs == {kwargs}")
        
        # for debugging
        fields_info = dict(zip(kwargs['df'].columns, map(lambda colname: kwargs['df'][colname].dtype, kwargs['df'].columns)))
        self.__logger_debug.debug(f"/*COLUMNS*/ kwargs == {fields_info}")
        
        model = self.get_model(model_id, model_name)
        self.__logger_debug.debug(f"model == {model}")
        
        self.__logger_debug.debug(f"kwargs['df'].isna().sum() == {kwargs['df'].isna().sum()}")
        
        self.__logger_debug.debug(f"fitting ...")
        model.fit(**kwargs)
        
        return model
    
    @fitmethod
    def _fit_hdmg_HdmgHeuristic(self, *args, **kwargs):
        
        self.__logger_debug.debug(f"<START>")
        
        model_id = kwargs.pop('model_id')
        model_name = kwargs.pop('model_name')
        
        self.__logger_debug.debug(f"model_id == {model_id}")
        self.__logger_debug.debug(f"model_name == {model_name}")
        
        data_params = self.__class__.__model_config[model_id][model_name]["data_params"]
        self.__logger_debug.debug(f"data_params == {data_params}")
        assert set(data_params).issubset(set(kwargs.keys()))
        
        self.__logger_debug.debug(f"args == {args}")
        self.__logger_debug.debug(f"kwargs == {kwargs}")
        
        # for debugging
        fields_info = dict(zip(kwargs['df'].columns, map(lambda colname: kwargs['df'][colname].dtype, kwargs['df'].columns)))
        self.__logger_debug.debug(f"/*COLUMNS*/ kwargs == {fields_info}")
        
        model = self.get_model(model_id, model_name)
        self.__logger_debug.debug(f"model == {model}")
        
        self.__logger_debug.debug(f"df == {kwargs['df'].isna().sum()}")
#         self.__logger_debug.debug(f"df['BDNG_RF_SE_DUMMY'].value_counts() == {kwargs['df']['BDNG_RF_SE_DUMMY'].value_counts(dropna=False)}")
        
        self.__logger_debug.debug(f"fitting ...")
        model.fit(**kwargs)
        
        return model
    
    @fitmethod
    def _fit_pdmg_mlp(self, *args, **kwargs):
        
        return self.__fit_pdmg_classifier(*args, **kwargs)
    
    @fitmethod
    def _fit_pdmg_rf(self, *args, **kwargs):
        
        return self.__fit_pdmg_classifier(*args, **kwargs)
    
    @fitmethod
    def _fit_pdmg_rfstump(self, *args, **kwargs):
        
        return self.__fit_pdmg_classifier(*args, **kwargs)
    
    @fitmethod
    def _fit_pdmg_rfensemble(self, *args, **kwargs):
        
        return self.__fit_pdmg_classifier(*args, **kwargs)
    
    @fitmethod
    def _fit_pdmg_xgb(self, *args, **kwargs):
        
        self.__logger_debug.debug(f"<START> _fit_( )")
        
        model_id = kwargs.pop('model_id')
        model_name = kwargs.pop('model_name')
        
        self.__logger_debug.debug(f"model_id == {model_id}")
        self.__logger_debug.debug(f"model_name == {model_name}")
        
        data_params = self.__class__.__model_config[model_id][model_name]["data_params"]
        self.__logger_debug.debug(f"data_params == {data_params}")
        assert set(data_params).issubset(set(kwargs.keys()))
        
        self.__logger_debug.debug(f"args == {args}")
        self.__logger_debug.debug(f"kwargs == {kwargs}")
        
        model = self.get_model(model_id, model_name)['model']
        self.__logger_debug.debug(f"model == {model}")
        
        self.__logger_debug.debug(f"fitting ...")
        
        X = kwargs.pop('X')
        Y = kwargs.pop('Y')
        self.__logger_debug.debug(f"X == {X}")
        self.__logger_debug.debug(f"Y == {Y}")
        
        model.fit(X, np.ravel(Y), **kwargs)
        
        ret = {'model': model, 'x_columns': X.columns.tolist()}
        self.__logger_debug.debug(f"ret == {ret}")
        
        return ret
    
    def __fit_pdmg_classifier(self, *args, **kwargs):
        self.__logger_debug.debug(f"<START> _fit_( )")
        
        model_id = kwargs.pop('model_id')
        model_name = kwargs.pop('model_name')
        
        self.__logger_debug.debug(f"model_id == {model_id}")
        self.__logger_debug.debug(f"model_name == {model_name}")
        
        data_params = self.__class__.__model_config[model_id][model_name]["data_params"]
        self.__logger_debug.debug(f"data_params == {data_params}")
        assert set(data_params).issubset(set(kwargs.keys()))
        
        self.__logger_debug.debug(f"args == {args}")
        self.__logger_debug.debug(f"kwargs == {kwargs}")
        
        model = self.get_model(model_id, model_name)['model']
        self.__logger_debug.debug(f"model == {model}")
        
        self.__logger_debug.debug(f"fitting ...")
        
        X = kwargs['X']
        Y = kwargs['Y']
        self.__logger_debug.debug(f"X == {X}")
        self.__logger_debug.debug(f"Y == {Y}")
        
        self.__logger_debug.debug(f"X.isna().sum() == {X.isna().sum()}")
        model.fit(X, np.ravel(Y))
        
        ret = {'model': model, 'x_columns': X.columns.tolist()}
        self.__logger_debug.debug(f"ret == {ret}")
        
        return ret
    
    def get_model(self, key: str, classname: str):
        model = self.__get_model(key, classname)
        self.__logger_debug.debug(f"model == {model}")
        if model is None:
            raise KeyError
        return model
    
    def __get_model(self, key: str, classname: str):
        
        return getattr(self, f'__{key}', dict()).get(classname, None)
    
    @classmethod
    def _build_pdmg(cls, **kwargs):
        return {
            "mlp": {
                'model': MLPClassifier(**kwargs["mlp"]["__init__"]), 
                'x_columns': []
            },
            "rf": {
                'model': BaggingTree(**kwargs["rf"]["__init__"]),
                'x_columns': []
            },
            "xgb": {
                'model': XgbModel(**kwargs["xgb"]["__init__"]),
                'x_columns': []
            },
        }
    
    @classmethod
    def _build_hdmg(cls, **kwargs):
        return {
            "HdmgHeuristic": HdmgHeuristic(**kwargs["HdmgHeuristic"]["__init__"])
        }
    
    @classmethod
    def _build_occr(cls, **kwargs):
        return {
            "CoxPHFitter": CoxPHFitterEntity(**kwargs["CoxPHFitter"]["__init__"])
        }

    
#     @classmethod
#     def __get_model_class(cls, target: str, from_clause=None):
        
#         fromlist = cls.__build_fromlist(from_clause)
#         mod = __import__(target, fromlist=fromlist)
        
#         return mod

#     @classmethod
#     def __build_fromlist(cls, from_clause=None, splitter: str = ","):
#         if not from_clause:
#             return []
#         return from_clause.split(splitter)


class HdmgHeuristic:
    
#     SIDO_HOLD_PCNT_WEIGHT = {'11'	:1.02 
#               , '26'	:0.98 
#               , '27'	:0.97 
#               , '28'	:0.98
#               , '29'	:0.96
#               , '30'	:0.96
#               , '31'	:0.96
#               , '36'	:0.95
#               , '41'	:1.05
#               , '42'	:0.98
#               , '43'	:0.97
#               , '44'	:0.98
#               , '45'	:0.97
#               , '46'	:0.98
#               , '47'	:0.99
#               , '48'	:0.98
#               , '50'	:0.96
#     }
    
    def __init__(self, **kwargs):
        self.dict_sigg = dict()
        self.dict_sido = dict()
        self.SIDO_HOLD_PCNT_WEIGHT = kwargs['SIDO_HOLD_PCNT_WEIGHT']
#         print(f"kwargs['SIDO_HOLD_PCNT_WEIGHT'] == {kwargs['SIDO_HOLD_PCNT_WEIGHT']}")
        
#         self.__dict__.update(kwargs)
    
    def fit(self, df: pd.DataFrame):
        
        df['PCNT_HDMG'] = df.apply(lambda row: row['N_HDMG'] / row['N'], axis=1)
        
        self.dict_sigg = dict(zip(df.SGG_CD.to_list(), df.PCNT_HDMG.to_list()))
        
        for sidocd in set(df["SIDOCD"]):
            n = df[df["SIDOCD"] == sidocd][["N","N_HDMG"]].sum().N
            n_hdmg = df[df["SIDOCD"] == sidocd][["N","N_HDMG"]].sum().N_HDMG
            prob = n_hdmg / n

            self.dict_sido.update({sidocd: prob})
    
    def predict(self, df: pd.DataFrame):
        
        max_prob: float = max(self.dict_sido.values())
        min_weight: float = min(self.SIDO_HOLD_PCNT_WEIGHT.values())
            
        return df.apply(lambda row: self.__hdmg_prob(row["SGG_CD"], row['FSTT_SEPR_DTC'], max_prob, min_weight), axis=1)
    
    def __hdmg_prob(self, sggcd: str, fstt_sepr_dtc: float, max_prob: float, min_weight: float):
        
        sidocd = sggcd[:2] if sggcd is not None else ''
    
        prob = self.dict_sigg.get(
            sggcd,
            self.dict_sido.get(
                sidocd, 
                max_prob
            )
        )
        
        weight = self.SIDO_HOLD_PCNT_WEIGHT.get(
            sidocd,
            min_weight
        )
        
        if not pd.isna(fstt_sepr_dtc):
            ret = ( 0.0054 * np.log( 1 + fstt_sepr_dtc ) + prob ) / weight    # or 0.0154
        else:
            ret = ( 0.0054 + prob ) / weight
            
        return ret


class BaggingTree(BaggingClassifier):
    
    def __init__(self, **kwargs):
        
        kwargs.update({
            "base_estimator": DecisionTreeClassifier(**kwargs["base_estimator"]["__init__"])
        })
        print(kwargs)
        
        super(BaggingTree, self).__init__(**kwargs)
        
    def __repr__(self):
        return super().__repr__()


class XgbModel:
    
    def __init__(self):
        
        self.__booster: xgb.core.Booster = None
            
    def __repr__(self):
        
        return self.__booster.__repr__()
        
    
    def fit(self, 
            X_train, Y_train, 
            booster: str, 
            objective: str, 
            num_class: int, 
            num_round: int
           ) -> None:
        
        X_train = pd.DataFrame(np.array(X_train))
        dtrain = xgb.DMatrix(X_train, Y_train)
        
        param = {
            'booster': booster, 
            'objective': objective, 
            'num_class': num_class
        }
#         watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        
        self.__booster = xgb.train(param, dtrain, num_round)
    
    def predict(self, X, Y):
        
        if not self.is_trained:
            raise Exception(f"Before Prediction, Please Execute the .fit( ) method first.")
        
        X = pd.DataFrame(np.array(X))
        dmatrix = xgb.DMatrix(X, Y)
        
        Y_pred = self.__booster.predict(dmatrix)
        Y_pred = Y_pred.astype(np.int64)
        
        return Y_pred

    @property
    def is_trained(self):
        return self.__booster is not None

class CoxPHFitterEntity(CoxPHFitter):
    
    def __repr__(self):
        return super().__repr__()
    
    def get_stats(self):
        return{
            "df": self.df ,
            "aic": self.aic,
            "deviance": self.deviance ,
            "concordance": self.concordance
        }
    
    @property
    def df(self):
        return self.log_likelihood_ratio_test().degrees_freedom
    
    @property
    def aic(self):
        try:
            return self.AIC_
        except StatError:
            return self.AIC_partial_
    
    @property
    def deviance(self):
        return -utils.quiet_log2(self.log_likelihood_ratio_test().p_value)        
    
    @property
    def concordance(self):
        return self.concordance_index_

    