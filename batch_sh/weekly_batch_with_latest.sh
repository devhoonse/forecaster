#!/bin/bash

# for debug
echo "current >> $0"

# if $pwd is different from batch_sh directory, then move to there.
THIS_SH_BASEDIR=$(dirname "$0")
echo "THIS_SH_BASEDIR=$THIS_SH_BASEDIR"
cd $THIS_SH_BASEDIR
echo "last_de_cd.sh |  pwd >> $(pwd)"
echo "last_de_cd.sh | $1"

# read python ath from modeler.properties file
PYTHON_PATH=$(./read_properties.sh $1 python.path)
echo "PYTHON_PATH >> $PYTHON_PATH"

# move to module location directory
# the module location directory should be same with the one of modeler.properties
MODULE_BASEDIR=$(dirname "$1")
echo "MODULE_BASEDIR=$MODULE_BASEDIR"
cd $MODULE_BASEDIR

# get maximum of [BASE_DE_CD] column
LAST_BASE_DE_CD=$($PYTHON_PATH -m forecast.src.get_last_base_de_cd $1)
echo "$LAST_BASE_DE_CD"

# | xargs /data/nfa2020/weekly_batch_with_input.sh
cd $THIS_SH_BASEDIR
echo "./weekly_batch_with_input.sh $1 $LAST_BASE_DE_CD"
./weekly_batch_with_input.sh $1 $LAST_BASE_DE_CD
