#!/usr/bin/env bash

export DAEMON_DIR=/mnt/faststore/repo/tempus-core/build
export DAEMON_OUTPUT=/mnt/slowstore/var/log/svrbacktest_tune.log
export VALIDATION_WINDOW=200
export SLIDE_COUNT=5

if [ -f ${DAEMON_OUTPUT} ]; then
  mv ${DAEMON_OUTPUT} /tmp/
fi

kill $(pidof SVRDaemon)
rm -f /dev/shm/sem.svrwave_gpu_sem /tmp/SVRDaemon_*
export SVRTUNE=1 # Always one
source ../scripts/setup_tempus_env.sh
if [[ $1 == "-d" ]]; then
	MKL_NUM_THREADS=${NUM_THREADS} OMP_NUM_THREADS=${NUM_THREADS} gdb --args ./SVRDaemon-blackbox-tests --gtest_filter='*backtest_xauusd*' | tee -a ${DAEMON_OUTPUT} 2>&1
else	
	MKL_NUM_THREADS=${NUM_THREADS} OMP_NUM_THREADS=${NUM_THREADS} ./SVRDaemon-blackbox-tests --gtest_filter='*backtest_xauusd*'  >> ${DAEMON_OUTPUT} 2>&1 &
fi
# cgclassify -g cpu,cpuset:Tempus $(pidof SVRDaemon)

