#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

for yearstr in 2017 2018 2019 2020 2021; do 
	for monthstr in Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec; do 
		echo -n $yearstr-$monthstr: ; bash  ${DIR}/calc_precision_backtest.sh $1 $yearstr-$monthstr; echo ' '
	done; 
done
