SELECT SGG_CD
     , COUNT(*)			AS N
     , SUM(TTL_YN)		AS N_HDMG 
FROM (
    SELECT SIDO_CN								-- [시도코드]
         , SGGN									-- [시군구코드] 뒤 3자리
         , SIDO_CN||SGGN AS SGG_CD				-- [시군구코드]
         , OFMN									-- [소방대상물ID]
         , CASE WHEN DPRS_PCNT + IJPS_PCNT > 0 
           THEN 1 
           ELSE 0 END AS TTL_YN					-- 인명피해 / [사상자발생여부]
         , DPRS_PCNT + IJPS_PCNT AS TTL_PCNT	-- 인명피해 / [사상자인원수]
         , DPRS_PCNT							-- 인명피해 / [사망자인원수]
         , IJPS_PCNT							-- 인명피해 / [부상자인원수]
    FROM TB_MTA_FRIG
    WHERE 1 = 1
    AND to_date(STMT_RCPT_DT, 'YYYYMMDDHH24MISS') > add_months(sysdate, -12*3)
    AND BCOP_CDN = '주거'
)
GROUP BY SGG_CD


-- SELECT A.OFMN
--      , A.PRMS_DE
--      , A.FIRE_RCPT_DE
--      , A.FRST_FIRE_OCCR_DE
--      , A.BDNG_PSVT_YCNT
--      , A.BEUP_CNT
--      , A.FIRE_SFIV_HIST_AT
--      , A.BDNG_FORM_SE_CDN
--      , A.BDNG_RF_SE_CDN
--      , A.BDNG_STRC_SE_CDN
--      , A.BDNG_MPP_CDN
--      , A.SIDO_CD
--      , A.SFMG_GRAD_CD
--      , A.NOW_FINS_SSCR_AT
--      , A.FFN_ADTN_CO
--      , A.DGST_MNGE_TRGT_CNT     
-- FROM DWA.TB_MTA_OBFF_RSAN_VABL	A
-- INNER JOIN (
-- 	SELECT B1.OFMN
-- 	FROM TB_WVD_OFFS B1
-- 	WHERE 1=1
-- 	AND   OFFS_TY_CD = '1000000001'
-- 	AND   USE_AT <> 'N'
-- 	AND   BF_BDNG_NO IS NOT NULL
-- ) B
-- ON A.OFMN = B.OFMN
-- WHERE 1 = 1
-- AND A.BASE_DE_CD = '{BASE_DE_CD}'
-- AND A.PRMS_DE IS NOT NULL
-- -- AND NVL(A.FIRE_SFIV_DE,1) < NVL(SUBSTR(A.FIRE_RCPT_DE,1,8),NVL(A.FIRE_SFIV_DE,1)+1)


-- -- SELECT * FROM DWA.STAT_TEST_BRIGHTICS
-- SELECT OFMN
--      , PRMS_DE
--      , FIRE_RCPT_DE
--      , FRST_FIRE_OCCR_DE
--      , BDNG_PSVT_YCNT
--      , BEUP_CNT
--      , FIRE_SFIV_HIST_AT
--      , BDNG_FORM_SE_CDN
--      , BDNG_RF_SE_CDN
--      , BDNG_STRC_SE_CDN
--      , BDNG_MPP_CDN
--      , SIDO_CD
--      , SFMG_GRAD_CD
--      , NOW_FINS_SSCR_AT
--      , FFN_ADTN_CO
--      , DGST_MNGE_TRGT_CNT     
-- FROM DWA.TB_MTA_OBFF_RSAN_VABL
-- WHERE 1 = 1
-- AND BASE_DE_CD = '{BASE_DE_CD}'
-- AND PRMS_DE IS NOT NULL
-- AND NVL(FIRE_SFIV_DE,1) < NVL(SUBSTR(FIRE_RCPT_DE,1,8),NVL(FIRE_SFIV_DE,1)+1)
-- AND ROWNUM < 5000    -- 너무 오래걸려서 임시 처리