#!/bin/bash

PYTHON_PATH=$(./read_properties.sh $1 python.path)

# move to module location directory
# the module location directory should be same with the one of modeler.properties
MODULE_BASEDIR=$(dirname "$1")
echo "MODULE_BASEDIR=$MODULE_BASEDIR"
cd $MODULE_BASEDIR

cd $MODULE_BASEDIR
if [ -z "$2" ]; then
	echo "pargs[1] is not set, current date will be used instead."
	nohup $PYTHON_PATH -m forecast.src.batch_predict --properties_path $1 &
else
	echo "yyyymmdd=$2"
	nohup $PYTHON_PATH -m forecast.src.batch_predict --properties_path $1 --yyyymmdd $2 &
fi


