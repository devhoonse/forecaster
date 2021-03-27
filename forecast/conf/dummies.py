# dummies 로직 개선
hash_map = {
    
    "SFMG_GRAD_CD": {       #안전조사 등급
        'A': "sg_type_1",
        'B': "sg_type_1",
        'C': "sg_type_1",
        'D': "sg_type_1",
        'N': "sg_type_1",
        
        'E': "sg_type_2"
    },
    
    "BDNG_FORM_SE_CDN": {    #건물 양식 : BDNG_FORM_SE_NM
        "기타": "bf_type_1", 
        "기타 식": "bf_type_1", 
        "조립식": "bf_type_1", 
        "한식(옥)": "bf_type_1" , 
        "절충식": "bf_type_1",
        
        "양식(옥)": "bf_type_2", 
        "일식": "bf_type_2", 
        "접수": "bf_type_2",
        
        "정보없음": "bf_type_3"
    },
    
    "BDNG_STRC_SE_CDN": {    # 건물 기둥 : BDNG_STRC_SE_NM
        '목조': "bs_type_1", 
        '벽돌조': "bs_type_1", 
        '블록조': "bs_type_1",
        '간이철골쇠파이프조': "bs_type_1", 
        '기타(건물구조조코드)': "bs_type_1",
        '기타 조': "bs_type_1",
        '간이목조': "bs_type_1",  
        '기타': "bs_type_1",  
        '도벽조': "bs_type_1",  
        '블럭조': "bs_type_1",  
        '비닐하우스 파이프조': "bs_type_1",  
        '시멘트벽돌조': "bs_type_1",  
        '시멘트블럭조': "bs_type_1",  
        '철골콘크리트조': "bs_type_1",  
        '치장벽돌조': "bs_type_1",  
        '컨테이너조': "bs_type_1",
        
        '샌드위치패널조': "bs_type_2", 
        '석조': "bs_type_2", 
        '정보없음': "bs_type_2", 
        '철골철근콘크리트조': "bs_type_2", 
        '철근콘크리트조': "bs_type_2",
        
        '철조': "bs_type_3", 
        '철골조': "bs_type_3"
    },
    
    "BDNG_RF_SE_CDN": {       # 지붕 구조 : BDNG_RF_SE_NM
        '블럭조': "br_type_1", 
        '비닐하우스': "br_type_1", 
        '샌드위치판넬': "br_type_1", 
        '석조': "br_type_1", 
        '초가': "br_type_1", 
        '칼라피복철판': "br_type_1",
        '컨테이너': "br_type_1", 
        '한식기와': "br_type_1",
        
        "슬라브가": "br_type_2", 
        "시멘트기와": "br_type_2",
        "정보없음": "br_type_2",
        
        "기타": "br_type_3", 
        "기타 즙": "br_type_3",
        "샌드위치패널": "br_type_3", 
        "와가": "br_type_3",
        '스레트가': "br_type_3", 
        '철근콘크리트조': "br_type_3",
        '기타(건물구조즙코드)': "br_type_3"
    },
    
    "BDNG_MPP_CDN": {    # 건물사용용도 : BDNG_MPP_NM
        '공동주택': "bm_type_1",  
        '공장': "bm_type_1",  
        '관광휴게시설': "bm_type_1",  
        '교육연구시설': "bm_type_1",  
        '교정및군사시설': "bm_type_1",  
        '근린생활시설': "bm_type_1",  
        '기타': "bm_type_1",  
        '노유자시설': "bm_type_1",  
        '동물및식물관련시설': "bm_type_1",  
        '묘지관련시설': "bm_type_1",  
        '문화및집회시설': "bm_type_1",  
        '문화재': "bm_type_1",  
        '발전시설': "bm_type_1",  
        '방송통신시설': "bm_type_1",  
        '수련시설': "bm_type_1",  
        '운동시설': "bm_type_1",  
        '운수시설': "bm_type_1",  
        '위락시설': "bm_type_1",  
        '위험물저장및처리시설': "bm_type_1",  
        '자원순환관련시설': "bm_type_1",  
        '장례식장': "bm_type_1",  
        '정보없음': "bm_type_1",  
        '주택': "bm_type_1",  
        '지하가': "bm_type_1", 
        '지하구': "bm_type_1",  
        '창고시설': "bm_type_1",
        
        '종교시설': "bm_type_2", 
        '판매시설': "bm_type_2", 
        '항공기및자동차관련시설': "bm_type_2",
        '업무시설': "bm_type_2", 
        '의료시설': "bm_type_2",
        
        '복합건축물': "bm_type_3", 
        '숙박시설': "bm_type_3"
    },
    
    "SIDO_CD": {       # 시도 정보
        '26': "sd_type_1", 
        '28': "sd_type_1", 
        '29': "sd_type_1", 
        '41': "sd_type_1", 
        '42': "sd_type_1", 
        '44': "sd_type_1", 
        '45': "sd_type_1", 
        '46': "sd_type_1", 
        '47': "sd_type_1", 
        '48': "sd_type_1", 
        '50': "sd_type_1",
        
        '27': "sd_type_2",
        '43': "sd_type_2", 
        '정보없음': "sd_type_2",
        
        '11': "sd_type_3",
        '30': "sd_type_3",
        
        '31': "sd_type_4",
        '36': "sd_type_4"
    },
}


# 변수 정제
def dummy_custom_variable(var, val):
    
    dummy_var = hash_map.get(var, dict()).get(val, hash_map.get(var, dict()).get("정보없음"))
    
    return dummy_var
