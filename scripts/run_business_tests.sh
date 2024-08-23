#!/usr/bin/env bash

source ../scripts/setup_tempus_env.sh
killall -w SVRBusiness-tests

# rm -f /dev/shm/sem.svrwave_gpu_sem /tmp/SVRDaemon_*


if [[ $1 == "-d" ]]; then
	$DBG --ex 'catch throw' --ex run --args ./SVRBusiness-tests --gtest_filter="$2" | tee -a ${BUSINESSTEST_OUTPUT} 2>&1
else
	./SVRBusiness-tests --gtest_filter="$1" >> ${BUSINESSTEST_OUTPUT} 2>&1 &
fi
# cgclassify -g cpu,cpuset:Tempus $(pidof SVRDaemon)
