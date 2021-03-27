from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from ..common.SingletonInstance import SingletonInstance
from ..logger.LogHandler import LogHandler
from ..preprocess.DateHandler import DateHandler
from ...conf import dummies


class Preprocessing(SingletonInstance):
    
    __logger_debug = LogHandler.instance().get_logger('debug-logger')
    __logger_error = LogHandler.instance().get_logger('error-logger')

    @classmethod
    def run(cls, key: str, data: pd.DataFrame, *args, **kwargs):
        __func = cls.__get_function(key)
        if not __func:
            raise KeyError
        return __func(data, *args, **kwargs)
    
    @classmethod
    def __get_function(cls, key: str):
        return {
            "pdmg": cls.__pdmg,
            "hdmg": cls.__hdmg,
            "occr": cls.__occr,
        }.get(key, None)
    
    @classmethod
    def __get_dummies_by_hash_value(cls, var_name: str, df: pd.DataFrame):
        dum_domain = sorted(dummies.hash_map[var_name].values())
        dum_domain_sr = pd.Series(dum_domain)
        
        tgt_sr = df[var_name].apply(
            lambda var_val: dummies.hash_map[var_name].get(var_val, dummies.hash_map[var_name].get('정보없음'))
        )
        
        sr = pd.concat([dum_domain_sr, tgt_sr], axis=0)
        ret = pd.get_dummies(sr)[dum_domain_sr.shape[0]:]
        
        return ret
    
    @classmethod
    def __get_dummies(cls, var_name: str, df: pd.DataFrame, prefix: str):
        
        dum_domain = sorted(dummies.hash_map[var_name].keys())
        dum_domain_sr = pd.Series(dum_domain)
        
        cls.__logger_debug.debug(f"dum_domain == {dum_domain}")
        
#         bf_dum.columns = list(map(lambda colname: f'BDNG_FORM_SE_CDN_{colname}', bf_dum.columns))    # 210128
        
#         tgt_df = df[[var_name]]
        
        # 21/02/02 - 등록 안된 값 '정보없음' 처리
        tgt_sr = df[var_name].apply(lambda var_val: var_val if var_val in dum_domain else '정보없음')
        
        sr = pd.concat([dum_domain_sr, tgt_sr], axis=0)
#         cls.__logger_debug.debug(f"sr == {sr}")

        cls.__logger_debug.debug(f"sr.shape == {sr.shape}")
        cls.__logger_debug.debug(f"sr.value_counts() == {sr.value_counts()}")
        
        ret2 = pd.get_dummies(sr)[dum_domain_sr.shape[0]:]
        ret2.columns = ret2.columns.to_series().apply(lambda colname: f"{prefix}_{colname}")
        
        cls.__logger_debug.debug(f"ret2 {ret2.shape} == {ret2}")
        
        return ret2
        
#         mlb = MultiLabelBinarizer()
        
#         onehot_array = mlb.fit_transform(sr)[dum_domain_sr.shape[0]:]
        
#         cls.__logger_debug.debug(f"dum_domain == {dum_domain}")
#         cls.__logger_debug.debug(f"onehot_array == {onehot_array}")
#         cls.__logger_debug.debug(f"mlb.classes_ == {mlb.classes_}")        
        
#         return pd.DataFrame(
#             onehot_array,
#             columns=list(map(lambda colname: f'BDNG_FORM_SE_CDN_{colname}', mlb.classes_)),
#             index=df.index
#         )
    
    @classmethod
    def __pdmg(cls, data: pd.DataFrame, *args, **kwargs):
        """
        화재피해량(재산) 예측 모델 데이터에 대한 전처리를 정의합니다.
        77번 서버의 03. 화재 재산피해 예측 노트북에 사용된 로직입니다.
        """
        
        cls.__logger_debug.debug(f"<START> PREPROCESSING OCCR")
        cls.__logger_debug.debug(f"data.columns == {data.columns}")
        
        
#         # 임시 ( 21/01/13 일자 모델 생성하려고 ... )
#         data['BDNG_FORM_SE_CDN'] = data['BDNG_FORM_SE_NM'].fillna("정보없음")
#         bf_dum = cls.__get_dummies_by_hash_value('BDNG_FORM_SE_CDN', data)   # , prefix='BF'

#         data['BDNG_RF_SE_CDN'] = data['BDNG_RF_SE_NM'].fillna("정보없음")
#         br_dum = cls.__get_dummies_by_hash_value('BDNG_RF_SE_CDN', data)    # , prefix='BR'

#         data['BDNG_STRC_SE_CDN'] = data['BDNG_STRC_SE_NM'].fillna("정보없음")
#         bs_dum = cls.__get_dummies_by_hash_value('BDNG_STRC_SE_CDN', data)   # , prefix='BS'

#         data['BDNG_MPP_CDN'] = data['BDNG_MPP_NM'].fillna("정보없음")
#         bm_dum = cls.__get_dummies_by_hash_value('BDNG_MPP_CDN', data)      # , prefix='BM'

#         data_1 = pd.concat([data, bf_dum, bs_dum, br_dum, bm_dum], axis = 1)
#         data_1.index = data['OFMN']

#         Y = data_1['PRTY_DMGE_SBSM_AMT'].fillna(0.0)    # PRTY_DMGE_SBSM_AMT : fixme: 나중에 마트 테이블 정상화되면 지워주자...
#         q25, q50, q75 = np.log(Y).quantile(0.25), np.log(Y).quantile(0.5), np.log(Y).quantile(0.75)

#         log_Y = data_1['PRTY_DMGE_SBSM_AMT'].apply(lambda x : np.log(x+1))
#         log_dum_Y = data_1['PRTY_DMGE_SBSM_AMT'].apply(lambda x : 0 if x ==0 else 1 if np.log(x) < q25 else 3 if np.log(x) > q75 else 2)

#         # fixme: 나중에 마트 테이블 정상화되면 지워주자...
#         data_1['FFN_ADTN_CO'] = data_1['FFN_CO'].fillna(0.0)
#         data_1['BEUP_CNT'] = data_1['BEUP_CO'].fillna(0.0)
#         data_1['DGST_MNGE_TRGT_CNT'] = data_1['DGST_CO'].fillna(0.0)

#         X = data_1[    # 과태료부과건수, 다중이용업소수, 위험물관리수 'FFN_CO', 'BEUP_CO', 'DGST_CO'
#             ['FFN_ADTN_CO', 'BEUP_CNT', 'DGST_MNGE_TRGT_CNT'] + list(bf_dum.columns) + list(bs_dum.columns) + list(bm_dum.columns) + list(br_dum.columns)
#         ]

#         ret = {
#             "X": X,
#             "Y": log_dum_Y
#         }
        
        
        # BAK       
               
        data['BDNG_FORM_SE_CDN'] = data['BDNG_FORM_SE_CDN'].fillna("정보없음")
        #data['BDNG_FORM_SE_DUMMY'] = data['BDNG_FORM_SE_NM'].apply(lambda x: dummy_custom_variable("BDNG_FORM_SE_NM", x))
        # bf_dum = pd.get_dummies(data['BDNG_FORM_SE_DUMMY'])

        # 21/02/01 에 로직 변경
#         bf_dum = pd.get_dummies(data['BDNG_FORM_SE_CDN'])    
        bf_dum = cls.__get_dummies_by_hash_value('BDNG_FORM_SE_CDN', data)   # , prefix='BF'
#         bf_dum.columns = list(map(lambda colname: f'BDNG_FORM_SE_CDN_{colname}', bf_dum.columns))    # 210128
        cls.__logger_debug.debug(f"bf_dum.columns == {bf_dum.columns}")

        data['BDNG_RF_SE_CDN'] = data['BDNG_RF_SE_CDN'].fillna("정보없음")
        #data['BDNG_RF_SE_DUMMY'] = data['BDNG_RF_SE_NM'].apply(lambda x: dummy_custom_variable("BDNG_RF_SE_NM", x))
        #br_dum = pd.get_dummies(data['BDNG_RF_SE_DUMMY'])

        # 21/02/01 에 로직 변경
#         br_dum = pd.get_dummies(data['BDNG_RF_SE_CDN'])
        br_dum = cls.__get_dummies_by_hash_value('BDNG_RF_SE_CDN', data)    # , prefix='BR'
#         br_dum.columns = list(map(lambda colname: f'BDNG_RF_SE_CDN_{colname}', br_dum.columns))    # 210128
        cls.__logger_debug.debug(f"br_dum.columns == {br_dum.columns}")

        data['BDNG_STRC_SE_CDN'] = data['BDNG_STRC_SE_CDN'].fillna("정보없음")
        #data['BDNG_STRC_SE_DUMMY'] = data['BDNG_STRC_SE_NM'].apply(lambda x: dummy_custom_variable("BDNG_STRC_SE_NM", x))
        #bs_dum = pd.get_dummies(data['BDNG_STRC_SE_DUMMY'])

        # 21/02/01 에 로직 변경
#         bs_dum = pd.get_dummies(data['BDNG_STRC_SE_CDN'])
        bs_dum = cls.__get_dummies_by_hash_value('BDNG_STRC_SE_CDN', data)    # , prefix='BS'
#         bs_dum.columns = list(map(lambda colname: f'BDNG_STRC_SE_CDN_{colname}', bs_dum.columns))    # 210128
        cls.__logger_debug.debug(f"bs_dum.columns == {bs_dum.columns}")

        data['BDNG_MPP_CDN'] = data['BDNG_MPP_CDN'].fillna("정보없음")
        #data['BDNG_MPP_DUMMY'] = data['BDNG_MPP_NM'].apply(lambda x: dummy_custom_variable("BDNG_MPP_NM", x))
        #bm_dum = pd.get_dummies(data['BDNG_MPP_DUMMY'])

        # 21/02/01 에 로직 변경
#         bm_dum = pd.get_dummies(data['BDNG_MPP_CDN'])
        bm_dum = cls.__get_dummies_by_hash_value('BDNG_MPP_CDN', data)   # , prefix='BM'
#         bm_dum.columns = list(map(lambda colname: f'BDNG_MPP_CDN_{colname}', bm_dum.columns))    # 210128
        cls.__logger_debug.debug(f"bm_dum.columns == {bm_dum.columns}")

        data_1 = pd.concat([data, bf_dum, bs_dum, br_dum, bm_dum], axis = 1)
        data_1.index = data['OFMN']

        Y = data_1['PRTY_DMGE_AMT'].fillna(0.0)    # PRTY_DMGE_SBSM_AMT : fixme: 나중에 마트 테이블 정상화되면 지워주자...
        q25, q50, q75 = np.log(Y).quantile(0.25), np.log(Y).quantile(0.5), np.log(Y).quantile(0.75)

        log_Y = data_1['PRTY_DMGE_AMT'].apply(lambda x : np.log(x+1))
        log_dum_Y = data_1['PRTY_DMGE_AMT'].apply(lambda x : 0 if x ==0 else 1 if np.log(x) < q25 else 3 if np.log(x) > q75 else 2)
#         log_dum_Y_test = data['PRTY_DMGE_SBSM_AMT'].apply(lambda x : 0 if np.log(x) < q50 else 1)

        # fixme: 나중에 마트 테이블 정상화되면 지워주자...
        data_1['FFN_ADTN_CO'] = data_1['FFN_ADTN_CO'].fillna(0.0)
        data_1['BEUP_CNT'] = data_1['BEUP_CNT'].fillna(0.0)
        data_1['DGST_MNGE_TRGT_CNT'] = data_1['DGST_MNGE_TRGT_CNT'].fillna(0.0)

        X = data_1[    # 과태료부과건수, 다중이용업소수, 위험물관리수 'FFN_CO', 'BEUP_CO', 'DGST_CO'
            ['FFN_ADTN_CO', 'BEUP_CNT', 'DGST_MNGE_TRGT_CNT'] + list(bf_dum.columns) + list(bs_dum.columns) + list(bm_dum.columns) + list(br_dum.columns)
        ]

#         data_1['ALL_AR'][data_1['ALL_AR'].isna()] = data_1['ALL_AR'].mean()
#         data_1['TTL_FLRC'][data_1['TTL_FLRC'].isna()] = data_1['TTL_FLRC'].mean()

#         grp = data_1[['BDNG_MPP_DUMMY']]

        # todo: train/test 분할까지 여기서 다 하고 반환해??
#         X_train, X_test, Y_train, Y_test = train_test_split(X, log_Y, test_size=0.4, random_state=1004, stratify=grp)

#         X_train, X_test, Y_train, Y_test = train_test_split(X, log_dum_Y, test_size=0.5, random_state=1004)
#         ret = {
#             "X": pd.concat([X_train, X_test]),
#             "Y": pd.concat([Y_train, Y_test])
#         }
        ret = {
            "X": X,
            "Y": log_dum_Y
        }
        
        cls.__logger_debug.debug(f"<SUCCESS>  PREPROCESSING OCCR")
        cls.__logger_debug.debug(f"ret == {ret}")
        
#         return {
#             "X_train": X_train, 
#             "X_test": X_test, 
#             "Y_train": Y_train, 
#             "Y_test": Y_test
#         }


        return ret
        
    
    @classmethod
    def __hdmg(cls, data: pd.DataFrame, *args, **kwargs) -> dict:
        """
        화재피해량(인명) 예측 모델 데이터에 대한 전처리를 정의합니다.
        77번 서버의 04. 화재 인명피해 예측 노트북에 사용된 로직입니다.
        """
        
        cls.__logger_debug.debug(f"<START> PREPROCESSING HDMG")
        
        # 21/02/05
        cls.__logger_debug.debug(f"<START>")
        cls.__logger_debug.debug(f"data == {data}")
        cls.__logger_debug.debug(f"args == {args}")
        cls.__logger_debug.debug(f"kwargs == {kwargs}")
        
        # 21/02/09
        if 'OFMN' in data.columns:
            data.index = data['OFMN']
        
        data['SIDOCD'] = data['SGG_CD'].apply(lambda sggcd: sggcd[:2] if type(sggcd) is str else None)
        
        
#         # ============CoxPHFitter 로 변경 전 전처리 로직입니다.==============
#         data['BDNG_FORM_SE_NM'] = data['BDNG_FORM_SE_NM'].fillna("정보없음")
#         data['BDNG_FORM_SE_DUMMY'] = data['BDNG_FORM_SE_NM'].apply(lambda x: dummies.dummy_custom_variable("BDNG_FORM_SE_NM", x))
#         bf_dum = pd.get_dummies(data['BDNG_FORM_SE_DUMMY'])
#         data['BDNG_RF_SE_NM'] = data['BDNG_RF_SE_NM'].fillna("정보없음")
#         data['BDNG_RF_SE_DUMMY'] = data['BDNG_RF_SE_NM'].apply(lambda x: dummies.dummy_custom_variable("BDNG_RF_SE_NM", x))
#         br_dum = pd.get_dummies(data['BDNG_RF_SE_DUMMY'])
#         data['BDNG_STRC_SE_NM'] = data['BDNG_STRC_SE_NM'].fillna("정보없음")
#         data['BDNG_STRC_SE_DUMMY'] = data['BDNG_STRC_SE_NM'].apply(lambda x: dummies.dummy_custom_variable("BDNG_STRC_SE_NM", x))
#         bs_dum = pd.get_dummies(data['BDNG_STRC_SE_DUMMY'])
#         data['BDNG_MPP_NM'] = data['BDNG_MPP_NM'].fillna("정보없음")
#         data['BDNG_MPP_DUMMY'] = data['BDNG_MPP_NM'].apply(lambda x: dummies.dummy_custom_variable("BDNG_MPP_NM", x))
#         bm_dum = pd.get_dummies(data['BDNG_MPP_DUMMY'])
        
#         Y = data['TTL_PCNT']
        
#         data_1 = pd.concat([data, bf_dum, bs_dum, br_dum, bm_dum], axis = 1)
        
#         X = data_1[[
#             'bf_type_1','bf_type_2', 'bf_type_3', 
#             'bs_type_1', 'bs_type_2', 'bs_type_3',
#             'br_type_1', 'br_type_2', 'br_type_3', 
#             'bm_type_1', 'bm_type_2', 'bm_type_3'
#         ]]
        
#         grp = data_1[['BDNG_MPP_DUMMY']]
        
#         # todo: train/test 분할까지 여기서 다 하고 반환해??
#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1004, stratify=grp)
#         # =====================================================================
    
        # todo: occr 과 동일한 처리를 하도록 임시 설정입니다.
#         ret = cls.__occr(data, *args, **kwargs)
    
        cls.__logger_debug.debug(f"<SUCCESS>  PREPROCESSING OCCR")
        
        return {"df": data}
    
    @classmethod
    def __occr(cls, data: pd.DataFrame, *args, **kwargs) -> dict:
        """
        화재발생위험율 예측 모델 데이터에 대한 전처리를 정의합니다.
        77번 서버의 05.모델적용 데이터 적재 노트북에 사용된 로직입니다.
        """
        
        # 21/02/05
        cls.__logger_debug.debug(f"<START>")
        cls.__logger_debug.debug(f"data == {data}")
        cls.__logger_debug.debug(f"args == {args}")
        cls.__logger_debug.debug(f"kwargs == {kwargs}")
        
        data.index = data['OFMN']
        
        data['STATUS'] = data['FRST_FIRE_OCCR_DE'].apply(lambda de: de is not None)
        
        ret: pd.DataFrame = data[['STATUS']]
        ret['TIME_YEARS'] = data['BDNG_PSVT_YCNT']
        ret['BEUP_CNT'] = data['BEUP_CNT'].fillna(0.0)
        ret['FIRE_SFIV_HIST_AT'] = data['FIRE_SFIV_HIST_AT'].fillna(0.0)
        
        data['BDNG_RF_SE_CDN'] = data['BDNG_RF_SE_CDN'].fillna('정보없음')
        bdng_rf_se_dummy = cls.__get_dummies_by_hash_value('BDNG_RF_SE_CDN', data).iloc[:, 1:]
        ret = pd.concat([ret, bdng_rf_se_dummy], axis=1)
        
        data['BDNG_MPP_CDN'] = data['BDNG_MPP_CDN'].fillna('정보없음')
        bdng_mpp_dummy = cls.__get_dummies_by_hash_value('BDNG_MPP_CDN', data).iloc[:, 1:]
        ret = pd.concat([ret, bdng_mpp_dummy], axis=1)
                          
        data['SIDO_CD'] = data['SIDO_CD'].fillna('정보없음')
        sido_cd_dummy = cls.__get_dummies_by_hash_value('SIDO_CD', data).iloc[:, 1:]
        ret = pd.concat([ret, sido_cd_dummy], axis=1) 
        
        ret['INSC_HX_AT'] = data['FINS_SSCR_HIST_AT'].apply(lambda value: int(value == '1'))    # NOW_FINS_SSCR_AT
#         ret['FFN_YN'] = data['FFN_ADTN_CO'].apply(lambda x: 1 if x is not None else 0)
#         ret['DGST_YN'] = data['DGST_MNGE_TRGT_CNT'].apply(lambda x: 1 if x is not None else 0)

        cls.__logger_debug.debug(f"ret.columns == {ret.columns}")
        cls.__logger_debug.debug(f"<END>")
        
        
        return {"df": ret}
                          
        
#         cls.__logger_debug.debug(f"<START> PREPROCESSING OCCR")
#         cls.__logger_debug.debug(f"data == {data}")
#         cls.__logger_debug.debug(f"args == {args}")
#         cls.__logger_debug.debug(f"kwargs == {kwargs}")
        
#         #Drop rows with null values:   PRMS_DT : PRMS_DE
# #         data = data.dropna(subset=['PRMS_DE'])      # 210203 : 조건제거
# #         data = data[data['PRMS_DE'].apply(lambda x : DateHandler.check_date(x, '%Y%m%d'))]    # 21/02/02 : 적재 단계에서 형식이 적절하지 않은 데이터는 null 로 적재되어 있을 것
        
        
    
#         # todo:
#         # derivate status
#         data['STATUS'] = data['FRST_FIRE_OCCR_DE'].apply(lambda de: de is not None)

#         # todo: 
#         # PRMS_DT : PRMS_DE  (허가일)
#         # RCPT_DT(=신고접수일자) : FRST_FIRE_OCCR_DE(=화재발생일자 )
#         # SYS_DT : ETL_LDDT
#         # TIME_YEARS : BDNG_PSVT_YCNT
#         data['TIME_YEARS'] = data['BDNG_PSVT_YCNT'].apply(lambda ycnt: ycnt if ycnt in range(0, 1+30) else 30)    # 21/02/03
        
# #         data['TIME_DAYS'] = data.apply(lambda x : DateHandler.days_between(x['PRMS_DE'], x['RCPT_DT']) if x['STATUS'] == 1 else DateHandler.days_between(x['PRMS_DE'], x['ETL_LDDT']) , axis=1)
# #         data['TIME_YEARS'] = data['TIME_DAYS'].apply(lambda x : int(x/365))

#         # 21/02/03 : fillna 처리 추가
#         data['BEUP_CNT'] = data['BEUP_CNT'].fillna(0.0)
#         data['FIRE_SFIV_HIST_AT'] = data['FIRE_SFIV_HIST_AT'].fillna(0.0)
        
#         # BDNG_FORM_SE_NM : BDNG_FORM_SE_CDN
#         data['BDNG_FORM_SE_CDN'] = data['BDNG_FORM_SE_CDN'].fillna("정보없음")
#         data['BDNG_FORM_SE_DUMMY'] = data['BDNG_FORM_SE_CDN'].apply(lambda x: dummies.dummy_custom_variable("BDNG_FORM_SE_CDN", x))
        
#         # BDNG_RF_SE_NM : BDNG_RF_SE_CDN
#         data['BDNG_RF_SE_CDN'] = data['BDNG_RF_SE_CDN'].fillna("정보없음")
#         data['BDNG_RF_SE_DUMMY'] = data['BDNG_RF_SE_CDN'].apply(lambda x: dummies.dummy_custom_variable("BDNG_RF_SE_CDN", x))
        
#         # BDNG_STRC_SE_NM : BDNG_STRC_SE_CDN
#         data['BDNG_STRC_SE_CDN'] = data['BDNG_STRC_SE_CDN'].fillna("정보없음")
#         data['BDNG_STRC_SE_DUMMY'] = data['BDNG_STRC_SE_CDN'].apply(lambda x: dummies.dummy_custom_variable("BDNG_STRC_SE_CDN", x))
        
#         # BDNG_MPP_NM : BDNG_MPP_CDN
#         data['BDNG_MPP_CDN'] = data['BDNG_MPP_CDN'].fillna("정보없음")
#         data['BDNG_MPP_DUMMY'] = data['BDNG_MPP_CDN'].apply(lambda x: dummies.dummy_custom_variable("BDNG_MPP_CDN", x))
        
#         # SIDO_CD : SIDO_CD
#         data['SIDO_CD'] = data['SIDO_CD'].fillna("정보없음")
#         data['SIDO_CD_DUMMY'] = data['SIDO_CD'].apply(lambda x: dummies.dummy_custom_variable("SIDO_CD", x))
        
#         # SFMG_GRAD_CD : SFMG_GRAD_CD
#         data['SFMG_GRAD_CD'] = data['SFMG_GRAD_CD'].fillna("N")
#         data['SFMG_GRAD_CD_DUMMY'] = data['SFMG_GRAD_CD'].apply(lambda x: dummies.dummy_custom_variable("SFMG_GRAD_CD", x))
        
#         # todo: 
#         # INSC_HX_AT : FINS_SSCR_HIST_AT (화재보험가입이력여부 )
#         # MTG_END_DE : ??
# #         data['INSC_HX_AT'] = data['MTG_END_DE'].apply(lambda x : 1 if x is not None else 0)
#         data['INSC_HX_AT'] = data['NOW_FINS_SSCR_AT'].apply(lambda value: value == 'N')    # 21/02/03
        
#         # FFN_CO : FFN_ADTN_CO
#         data['FFN_YN'] = data['FFN_ADTN_CO'].apply(lambda x : 1 if x is not None else 0)    # 21/02/02
        
#         # DGST_CO : DGST_MNGE_TRGT_CNT
#         data['DGST_YN'] = data['DGST_MNGE_TRGT_CNT'].apply(lambda x : 1 if x is not None else 0)    # 21/02/02
        
#         # OFMN : OFMN
#         data.index = data['OFMN']
        
#         cls.__logger_debug.debug(f"<SUCCESS>  PREPROCESSING OCCR")

#         return {"df": data[['OFMN', 'TIME_YEARS', 'STATUS', 'BDNG_RF_SE_DUMMY', 'BDNG_MPP_DUMMY', 'BEUP_CNT', 'INSC_HX_AT', 'FIRE_SFIV_HIST_AT', 'SIDO_CD_DUMMY']]}