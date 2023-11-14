#!/usr/bin/env bash

source ../scripts/setup_tempus_env.sh
cd ${DAEMON_DIR}

while (pidof -csn OnlineSVR); do
  pkill OnlineSVR
done

rm -f /dev/shm/sem.svrwave_gpu_sem

if [[ $1 == "-d" ]]; then
  echo "TBB does exception testing on start, ignore the first exception!"
	$DBG --ex 'catch throw' --ex run --ex continue --args ./OnlineSVR-unit-tests --gtest_filter="$2" | tee -a ${ONLINETEST_OUTPUT} 2>&1
elif [[ $1 == "-v" ]]; then
  /usr/bin/valgrind  --max-threads=$NUM_THREADS --track-origins=yes --error-limit=no --log-file=OnlineSVR-unit-tests.valgrind.log --leak-check=full --tool=memcheck --show-leak-kinds=all --max-stackframe=115062830400 ./OnlineSVR-unit-tests --gtest_filter="$2" # --vgdb=full --vgdb-error=1
else
#	eval `dmalloc -d 0 -l log_file -p log-non-free -p check-fence -p check-funcs`
	./OnlineSVR-unit-tests --gtest_filter="$1" >> ${ONLINETEST_OUTPUT} 2>&1 &
fi
