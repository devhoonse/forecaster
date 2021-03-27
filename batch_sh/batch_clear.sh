#!/bin/bash

INPUT_PRED_DATA_ROOT=$1
MODEL_HDMG_DATA_ROOT=$2

# if $pwd is different from batch_sh directory, then move to there.
THIS_SH_BASEDIR=$(dirname "$0")
echo "THIS_SH_BASEDIR=$THIS_SH_BASEDIR"

# 1. clear mail wrote by crontab
$THIS_SH_BASEDIR/clear_mail.sh $THIS_SH_BASEDIR

# 2. clear pickles
$THIS_SH_BASEDIR/clear_pickles.sh $INPUT_PRED_DATA_ROOT $MODEL_HDMG_DATA_ROOT
