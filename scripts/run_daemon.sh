#!/usr/bin/env bash
#export SVRWAVE_TEST_WINDOW=660
source ../scripts/setup_tempus_env.sh

REMOVE_OUTPUT=0
DEBUG_SESSION=0
WEB_ONLY=0
LOG_OUTPUT=0
for ARG in "$@"
do
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

if [ $WEB_ONLY -eq 0 ]; then
#     pkill SVRDaemon;
     rm -f /dev/shm/sem.svrwave_gpu_sem /tmp/SVRDaemon_*
fi

while (pidof -csn SVRWeb); do
  kill $(pidof -csn SVRWeb)
  if (pidof -csn SVRWeb); then kill -9 $(pidof -csn SVRWeb); fi
done
sleep 1


if [ $REMOVE_OUTPUT -eq 1 ]; then
	rm -f $DAEMON_OUTPUT $WEB_OUTPUT
	rm -f *_chunk *_ixs xtest_level_*.txt *_reference_matrix.txt *_learning_matrix.txt *.out
fi


#cgexec -g cpu,cpuset:Tempus ./SVRWeb -a $DAEMON_DIR/../config/app.config -c $DAEMON_DIR/../config/config.json >> ${WEB_OUTPUT} 2>&1 &
./SVRWeb -a $DAEMON_DIR/../config/app.config -c $DAEMON_DIR/../config/config.json >> ${WEB_OUTPUT} 2>&1 &
if [ $WEB_ONLY -eq 1 ]; then
	exit 0
fi

if [ $DEBUG_SESSION -eq 1 ]; then
	$DBG --ex catch\ throw --ex run --ex continue --args ./SVRDaemon -c ${DAEMON_CONFIG} 2>&1 | tee -a ${DAEMON_OUTPUT}.debug 2>&1
elif [ $LOG_OUTPUT -eq 1 ]; then
  ./SVRDaemon -c ${DAEMON_CONFIG} >> ${DAEMON_OUTPUT} 2>&1 &
else	
	./SVRDaemon -c ${DAEMON_CONFIG}
fi
# cgclassify -g cpu,cpuset:Tempus $(pidof SVRDaemon)
