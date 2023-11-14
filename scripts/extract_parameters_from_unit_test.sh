#!/bin/bash
echo -e '\n'
for l in `seq 0 1 39`; do 
	if (( $l % 2 == 0 )); then
		grep level_${l}_ /var/tmp/parameters.log | sort -k10 | head -1| sed -r "s/.*MAE: ([^ ]+) .*C: ([^,]+), epsilon: ([^,]+), kernel_param: ([^,]+),.*/MAE: \1\t$l\t\2\t\3\t\4/g"
	else
		echo -e "MAE: 0\t$l\t0\t0\t0"
	fi
done
