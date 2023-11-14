#!/usr/bin/env bash

export TEST_NAME="DaoTestFixture.backtest_xauusd"
source ../scripts/setup_tempus_env.sh

#while (pidof -csn SVRDaemon); do
#  pkill SVRDaemon
#done

while (pidof -csn SVRDaemon-blackbox-tests); do
  kill $(pidof -csn SVRDaemon-blackbox-tests)
  if (pidof -csn SVRDaemon-blackbox-tests); then kill -9 $(pidof -csn SVRDaemon-blackbox-tests); fi
done
sleep 1

export BACKTEST=1
# rm -f /dev/shm/sem.svrwave_gpu_sem /tmp/SVRDaemon_*
if [[ $1 == "-d" ]]; then
	$DBG --args ./SVRDaemon-blackbox-tests --gtest_filter="${TEST_NAME}" | tee -a ${BACKTEST_OUTPUT} 2>&1
else	
	./SVRDaemon-blackbox-tests --gtest_filter="${TEST_NAME}" >> ${BACKTEST_OUTPUT} 2>&1 &
fi
# cgclassify -g cpu,cpuset:Tempus $(pidof SVRDaemon)

