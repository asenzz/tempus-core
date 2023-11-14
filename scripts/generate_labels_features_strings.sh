#!/bin/bash

for ff in `ls /mnt/faststore/labels_*level_0_*csv`; do
  column_name=`echo $ff | sed 's/.*_[[:digit:]]\+_\(.*\)_level_.*/\1/g'`
  for l in `seq 0 2 63`; do
    echo "{\"`ls /mnt/faststore/labels_*${column_name}*level_${l}_*csv`\","
    echo "\"`ls /mnt/faststore/features_*${column_name}*level_${l}_*csv`\"},"
  done
done
