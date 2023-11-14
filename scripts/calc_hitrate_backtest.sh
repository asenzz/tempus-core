#!/bin/bash

if [[ -z "$2" ]]; then
  LIMIT=1000
else
  LIMIT=${2}
fi

if [[ -z "$3" ]]; then
  TAIL=1000
else
  TAIL=${3}
fi

for column_name in xauusd_avg_bid; do # ff in `ls /mnt/faststore/labels_*level_0_*csv`; do
#  column_name=`echo $ff | sed 's/.*_[[:digit:]]\+_\(.*\)_level_.*/\1/g'`
  grep "Mean absolute error to OHLC .*${column_name}" $1 | head -${LIMIT} | tail -${TAIL} > /tmp/ohlc_hitrate.txt
  zeroes=$(grep "Mean absolute error to OHLC 0, " /tmp/ohlc_hitrate.txt | grep ${column_name} | wc -l); alls=$(grep "Mean absolute error to OHLC " /tmp/ohlc_hitrate.txt | grep ${column_name} | wc -l) ; echo 100*$zeroes/$alls | bc -l 2> /dev/null
  if [[ -z "$2" ]]; then
    echo $column_name
    if [ -z $2 ]; then grep Cycle\ at $1 | tail -1; fi
    echo "Total $alls cycles."
  fi
done
