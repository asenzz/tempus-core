#!/usr/bin/env bash

export BIN=SVRDaemon

source ../scripts/setup_tempus_env.sh

unset SVRWAVE_TEST_WINDOW

REMOVE_OUTPUT=0
DEBUG_SESSION=0
WEB_ONLY=0
LOG_OUTPUT=0
CLEAN_DB=0
for ARG in "$@"
do
    if [[ $ARG == "-c" ]]; then
	      CLEAN_DB=1
    fi
    if [[ $ARG == "-r" ]]; then
	      REMOVE_OUTPUT=1
    fi
    if [[ $ARG == "-d" ]]; then
	      DEBUG_SESSION=1
    fi
    if [[ $ARG == "-w" ]]; then
	      WEB_ONLY=1
    fi
    if [[ $ARG == "-l" ]]; then
	      LOG_OUTPUT=1
    fi
done

if [ $CLEAN_DB -eq 1 ]; then
    echo Cleaning up database.
    PGPASSWORD=svrwave psql -U svrwave -h /var/run/postgresql -d svrwave -c "
      DELETE FROM svr_parameters;
      DELETE FROM dq_scaling_factors;
      DELETE FROM iq_scaling_factors;
      DELETE FROM multival_requests WHERE processed = 't'"
fi

if [ $WEB_ONLY -eq 0 ]; then
  killwait ${BIN}
  rm -f /dev/shm/sem.svrwave_gpu_sem /tmp/${BIN}_*
fi

killwait SVRWeb

if [ $REMOVE_OUTPUT -eq 1 ]; then
  echo Removing debug and log files.
	rm -f $DAEMON_OUTPUT $WEB_OUTPUT
	rm -f *_chunk *_ixs xtest_level_*.txt *_reference_matrix.txt *_learning_matrix.txt *.out
fi


# cgexec -g cpu,cpuset:Tempus ./SVRWeb -a $DAEMON_DIR/../config/app.config -c $DAEMON_DIR/../config/config.json >> ${WEB_OUTPUT} 2>&1 &
./SVRWeb -a $DAEMON_DIR/../config/app.config -c $DAEMON_DIR/../config/config.json >> ${WEB_OUTPUT} 2>&1 &
if [ $WEB_ONLY -eq 1 ]; then
  echo Started only REST interface.
	exit 0
fi

if [ $DEBUG_SESSION -eq 1 ]; then
	$DBG --ex catch\ throw --ex run --ex where --args ./${BIN} -c ${DAEMON_CONFIG} 2>&1 | tee -a ${DAEMON_OUTPUT}.debug
elif [ $LOG_OUTPUT -eq 1 ]; then
  ./${BIN} -c ${DAEMON_CONFIG} >> ${DAEMON_OUTPUT} 2>&1 &
else	
	./${BIN} -c ${DAEMON_CONFIG}
fi
# cgclassify -g cpu,cpuset:Tempus $(pidof ${BIN})
