#!/bin/bash

if [[ -z "$2" ]]; then
  export LIMIT=1000
else
  export LIMIT=${2}
fi

if [[ -z "$3" ]]; then
  export TAIL=1000
else
  export TAIL=${3}
fi

for column_name in xauusd_avg_bid; do # ff in `ls /mnt/faststore/labels_*level_0_*csv`; do
#  column_name=`echo $ff | sed 's/.*_[[:digit:]]\+_\(.*\)_level_.*/\1/g'`
  grep "Mean absolute error to OHLC .*${column_name}" $1 | head -${LIMIT} | grep -v ' 1000 ' | tail -${TAIL} > /tmp/ohlc_prec.txt
  all_errors=$(sed -e 's/.* average \([^ ]\+\) .*/\1/;t;d' /tmp/ohlc_prec.txt)
  all_errors=$(sed -E 's/([+-]?[0-9.]+)[eE]\+?(-?)([0-9]+)/(\1*10^\2\3)/g' <<< "$all_errors")
  count=$(echo $all_errors | wc -w)
  if [[ -z $all_errors ]]; then
    total_error=0
  else
    total_error=$(echo $all_errors | sed -e 's/ /+/g' | bc -l)
  fi
  if [[ $total_error == 0 || $count == 0 ]]; then
    echo No cycles yet.
  else
    echo "$total_error/$count" | bc -l
  fi
  if [[ -z "$2" ]]; then
    echo $column_name
    if [ -z $2 ]; then grep Cycle\ at $1 | tail -1; fi
    echo "Total $count cycles."
  fi
done
