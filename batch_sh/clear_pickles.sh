#!/bin/bash


# $1 should be : /data/nfa2020/forecast/input/predict
for modelname in $(ls $1)
do
	echo $1/$modelname
	if [ $(ls -l -tr $1/$modelname | wc -l) -lt 5 ]; then
        	echo "less than 5"
	else
        	# remove first file in time order
        	echo "greater than 5"
		echo "delete >> $1/$modelname/$(ls -tr $1/$modelname | head -1)"
		rm  $1/$modelname/$(ls -tr $1/$modelname | head -1)

	fi
done


# $2 should be : /data/nfa2020/forecast/model/hdmg
echo $2
if [ $(ls -l -tr $2 | wc -l) -lt 5 ]; then
        echo "less than 5"
else
	# remove first file in time order
        echo "greater than 5"
	echo "delete >> $2/$(ls -tr $2 | head -1)"
	rm -rf $2/$(ls -tr $2 | head -1)
fi
