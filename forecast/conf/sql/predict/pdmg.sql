-- 변경 2차
SELECT OFMN
     , BDNG_FORM_SE_CDN
     , BDNG_RF_SE_CDN
     , BDNG_STRC_SE_CDN
     , BDNG_MPP_CDN
     , FFN_ADTN_CO    -- 변경
     , BEUP_CNT    -- 변경
     , DGST_MNGE_TRGT_CNT    -- 변경
     , PRTY_DMGE_AMT -- 변경
FROM DWA.TB_MTA_OBFF_RSAN_VABL
WHERE 1 = 1
AND BASE_DE_CD = '{BASE_DE_CD}'
-- AND ROWNUM < 1000    -- 너무 오래걸려서 임시 처리
ORDER BY OFMN