#!/bin/bash

export tail=/usr/bin/tail
export egrep=/bin/egrep
export sed=/bin/sed
export less=/usr/bin/less
export lastlines=1000000

if [ -z "$1" ]; then 1="/mnt/slowstore/var/log/svrdaemon_generate_forecasts.log"; fi

$tail -${lastlines} $1 | \
  $egrep '(Total|Position (46|69|92|115|139),|Recombine.*start row 0|Predictions filtered out|Lambda, gamma tune best score|Tuning parameters.*level 22|Best 0 score.*|Execution time of (Tune.*model 0|Recombine parameters|OEMD FIR)|Offline test offset|Validating [0-9.]+ gamma multipliers, .*level 0)' | \
  $sed 's/actual price .*, predicted price .*, last known .*, total MAE [^,]\+, //g'