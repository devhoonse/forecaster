import sys

import pandas as pd
import jaydebeapi

from ..lib.dao.DatabaseManager import DatabaseManager
from ..lib.conf.ApplicationConfiguration import ApplicationConfiguration
from ..lib.logger.LogHandler import LogHandler


# TB_MVD_OFF_RSAN_INDX, TB_MVD_BDNG_OFFS_RSAN_INDX
sql_truncate = """
TRUNCATE TABLE {target_table_name}  -- 하드코딩 해결완료
;
"""

sql_last_de_src = """
SELECT max(BASE_DE_CD) AS LAST_DE
FROM {source_table_name}           -- 하드코딩 해결완료
WHERE 1 = 1
"""


def main():
    
    # 설정 정보를 가져옵니다.
    modeler_properties_path = sys.argv[1]
    configuration = ApplicationConfiguration.instance()
    configuration.init(modeler_properties_path)
    
    target_table_name = configuration.find("Target", "target.table.name")
    source_table_name = configuration.find("Source", "source.table.name")
    
    # 로그 핸들러를 초기화합니다.
    LogHandler.instance().init(configuration)
    
    database_manager = DatabaseManager.instance()
    database_manager.init(configuration)
    
    database_manager.connect_database()
    
    # truncate target table (ex) DWA.TB_MVD_BDNG_OFFS_RSAN_INDX
    database_manager.execute(sql_truncate.format(target_table_name=target_table_name))
    
    # select maximum of [BASE_DE_CD]
    val_last_de_src = database_manager.execute_select(
        sql_last_de_src.format(source_table_name=source_table_name)
    ).iloc[0, 0]
    
    database_manager.disconnect_database()
    
    print(f"{val_last_de_src}")
    

if __name__ == '__main__':
    main()
    
    
# ## Tibero DB 
# DB_DRV = "com.tmax.tibero.jdbc.TbDriver"
# DB_IP  = "10.175.148.53"
# DB_PORT= "8629"
# DB_SID = "FSIPRD"
# DB_ID  = "dwa"
# DB_PWD = "dwa1023"
# DB_URL = "jdbc:tibero:thin:@"+ DB_IP + ":" + DB_PORT + ":" + DB_SID
# Tibero_Jar = '/app/anaconda3/extlib/tibero6-jdbc.jar'


# # # TB_MVD_OFF_RSAN_INDX
# # sql_last_de_tgt = """
# # SELECT max(BASE_DE_CD) AS LAST_DE
# # FROM TB_MVD_BDNG_OFFS_RSAN_INDX      -- hardcoded
# # WHERE 1 = 1
# # """



# conn = jaydebeapi.connect(DB_DRV, DB_URL, [DB_ID, DB_PWD], Tibero_Jar)

# cur = conn.cursor()

# # 
# cur.execute(sql_last_de_src)
# val_last_de_src = cur.fetchall()[0][0]

# # 
# cur.execute(sql_last_de_tgt)
# val_last_de_tgt = cur.fetchall()[0][0]

# # 

# cur.execute(sql_truncate)


# print(f"{val_last_de_src}")

