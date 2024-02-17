#!/usr/bin/env bash
export EXE=./OnlineSVR-test
source ../scripts/setup_tempus_env.sh
cd ${DAEMON_DIR}

while (pidof -csn OnlineSVR); do
  pkill OnlineSVR
done

rm -f /dev/shm/sem.svrwave_gpu_sem

if [[ $1 == "-d" ]]; then
  echo "TBB does exception testing on start, ignore the first exception!"
	${DBG} --ex 'catch throw' --ex run --args ${EXE} --gtest_filter="$2" | tee -a ${ONLINETEST_OUTPUT} 2>&1
elif [[ $1 == "-v" ]]; then
  /usr/bin/valgrind  --max-threads=10000 --track-origins=yes --error-limit=no --log-file=${EXE}.valgrind.log --leak-check=full --tool=memcheck --expensive-definedness-checks=yes --show-leak-kinds=definite --max-stackframe=115062830400 ${EXE} --gtest_filter="$2"  | tee -a ${ONLINETEST_OUTPUT} 2>&1 # Enable to start GDB server on first error --vgdb=full --vgdb-error=1
else
#	eval `dmalloc -d 0 -l log_file -p log-non-free -p check-fence -p check-funcs`
	${EXE} --gtest_filter="$1" >> ${ONLINETEST_OUTPUT} 2>&1 &
fi
