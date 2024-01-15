#!/bin/bash

export tail=/usr/bin/tail
export egrep=/bin/egrep
export sed=/bin/sed
export less=/usr/bin/less
export lastlines=5000000

if [ -z "$1" ]; then 1="/mnt/slowstore/var/log/svrdaemon_generate_forecasts.log"; fi
levels="[0-9]+"
$tail -${lastlines} $1 | \
  $egrep "(_Auto epsco is.*decon level ${levels}|_Mean Z |pass score .* decon level ${levels}|Total|Position (46|69|92|115|139),|Recombine.*start row 0|Predictions filtered out|Lambda, gamma tune best score.*decon level ${levels},|Tuning parameters.*level ${levels}|!Best 0 score.*|Execution time of (Tune.*model 0|Recombine parameters|OEMD FIR)|Offline test offset|Validating [0-9.]+ gamma multipliers, .*level ${levels})" | \
  $sed 's/actual price .*, predicted price .*, last known .*, total MAE [^,]\+, //g'
