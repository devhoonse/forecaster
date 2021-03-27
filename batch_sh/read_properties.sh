#!/bin/bash

PROPERTY_FILE=$1

function getProperty {
	PROP_KEY=$1
	PROP_VALUE=$(cat $PROPERTY_FILE | grep "$PROP_KEY" | cut -d '=' -f2)
	echo $PROP_VALUE
}

# echo "reading property from $PROPERTY_FILE"
# echo "$2=$(getProperty $2)"
echo $(getProperty $2)
