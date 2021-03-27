#!/bin/bash

MODELER_PROPERTIES=/data/nfa2020/modeler.properties

# if $pwd is different from batch_sh directory, then move to there.
THIS_SH_BASEDIR=$(dirname "$0")
echo "THIS_SH_BASEDIR=$THIS_SH_BASEDIR"
# cd $THIS_SH_BASEDIR

# $1 should be /path/to/modeler.properties
MDL_DATA_ROOT=$($THIS_SH_BASEDIR/read_properties.sh $MODELER_PROPERTIES serialized.model.root)
PRD_DATA_ROOT=$($THIS_SH_BASEDIR/read_properties.sh $MODELER_PROPERTIES predict.data.root)
FIN_SIGN_FILE=$($THIS_SH_BASEDIR/read_properties.sh $MODELER_PROPERTIES finish.sign)
SGN_SCAN_TIME=$($THIS_SH_BASEDIR/read_properties.sh $MODELER_PROPERTIES sign.scan.time)
SGN_SCAN_CNT=$($THIS_SH_BASEDIR/read_properties.sh $MODELER_PROPERTIES sign.scan.count)


# clear legacy data
echo "$THIS_SH_BASEDIR/batch_clear.sh $PRD_DATA_ROOT $MDL_DATA_ROOT/hdmg"
$THIS_SH_BASEDIR/batch_clear.sh $PRD_DATA_ROOT $MDL_DATA_ROOT/hdmg

# wait until the sign file is created
echo "waiting for sign file $FIN_SIGN_FILE ... "
until [[ SGN_SCAN_CNT -lt 0 || -f $FIN_SIGN_FILE ]];
do
	echo "not yet...$SGN_SCAN_CNT"
	sleep $SGN_SCAN_TIME
	let SGN_SCAN_CNT=SGN_SCAN_CNT-1
done

if [ -f $FIN_SIGN_FILE ]
then
	# alert that file has been detected
	echo "file has been detected >> $FIN_SIGN_FILE"

	# as soon as the sign file is created, run the script
	echo "start weekly batch job"
	# cd $THIS_SH_BASEDIR
	echo "$THIS_SH_BASEDIR/weekly_batch_with_latest.sh $MODELER_PROPERTIES"
	$THIS_SH_BASEDIR/weekly_batch_with_latest.sh $MODELER_PROPERTIES
	echo "exit shell"

	# when finished, remove the sign file
	rm $FIN_SIGN_FILE
	echo "removed sign file $FIN_SIGN_FILE"
fi
