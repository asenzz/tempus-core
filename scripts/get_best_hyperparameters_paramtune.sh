#!/bin/sh
#
# Extracts the best parameters from a paramtune log file. For best performance grep 'Loss count' out of the paramtune output once paramtune is stopped.
# This script is made so that at times when paramtune is unable to stop gracefully or finish the tweaking process the best parameters are extracted
# from its log output.
#

export INPUTFILE=$1
export FILTERED_FILE="/tmp/filtered"
export SLIDENUM=3
export DATASET=2023
export LEVELS=15
#90 for 5m
#100 for 1m

grep "Loss call count" ${INPUTFILE} > ${FILTERED_FILE}

#old code - without sql
for level in `seq $LEVELS $LEVELS` ; do cat $FILTERED_FILE | grep "level $level " | sed "s/.*count $SLIDENUM score //" | sort -fgr --key="1.1,2.1" |tail -n 6 ;done

#new code - produces update statement

for level in `seq 0 ${LEVELS} ` ; do export level;  cat $FILTERED_FILE | grep "level $level " | sed "s/.*count $SLIDENUM score//" | sort -r -n |tail -n 1 | sed "s/.*C: /update svr_parameters set svr_c=/" |sed "s/, epsilon: /,svr_epsilon=/"|sed "s/, kernel_param: /,svr_kernel_param=/" |sed  "s/ kernel_param2: /svr_kernel_param2=/" | sed "s/decremental_distance: /svr_decremental_distance= /" |sed "s/, svr_adjacent_levels_ratio: /, svr_adjacent_levels_ratio= /"| sed "s/kernel_type: /svr_kernel_type=/" | sed "s/lag_count: /lag_count=/" | sed "s/\$/ where dataset_id=${DATASET} and decon_level=$level /" | sed "s/$/;/"; done

#for paramtune config file
for level in `seq 0 $LEVELS ` ; do cat $FILTERED_FILE | grep "level $level " | sed "s/.*count $SLIDENUM score//" | sort -r -n | tail -n 1 | sed "s/.*C: /\"svr_c_$level\":\"/" | sed "s/\, epsilon: /\", \"svr_epsilon_$level\":\"/" | sed "s/, kernel_param: /\", \"svr_kernel_param_$level\":\"/" | sed "s/, kernel_param2: /\", \"svr_kernel_param2_$level\":\"/" | sed "s/, decremental_distance: /\", \"svr_decremental_distance_$level\":\"/" | sed "s/, svr_adjacent_levels_ratio: /\", \"svr_adjacent_levels_ratio_$level\":\"/" | sed "s/, kernel_type: /\", \"svr_kernel_type_$level\":\"/" | sed "s/, lag_count: /\", \"lag_count_$level\":\"/" | sed "s/$/\",/"; done
rm -f ${FILTERED_FILE}
