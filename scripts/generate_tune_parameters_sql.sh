#!/bin/bash
levels=64
column_names='eurusd_avg_bid'
input_queue_name=q_svrwave_eurusd_avg_3600
echo -e '\tId\tDataset Id\tInput Queue\tInput Column\tLevel\tCost\tEpsilon\tGamma\tLambda\tDecrement\tAdjacent\tKernel\tLag'
for ff in `ls /var/tmp/svr_parameters_*.tsv`; do
  tail -1 $ff;
done
for column_name in ${column_names}; do
  for l in `seq 0 1 $(($levels-1))`; do
    if (( $l % 2 != 0 || $l == $levels/2 )); then
      echo -e "\t0\t100\t${input_queue_name}\t${column_name}\t$l\t0\t0\t0\t0\t0\t0\t0\t0"
    fi
  done
done
