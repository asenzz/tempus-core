#!/usr/bin/env bash
export SVRWAVE_TEST_WINDOW=50
export BIN=OnlineSVR-test
source ../scripts/setup_tempus_env.sh
cd "${DAEMON_DIR}" || exit

pkill -9 "${BIN}"

rm -f /dev/shm/sem.svrwave_gpu_sem

if [[ $1 == "-d" ]]; then
  echo "TBB does exception testing on start, ignore the first exception!"
	${DBG} --ex 'catch throw' --ex run --args ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
elif [[ $1 == "-v" ]]; then
  /usr/bin/valgrind  --max-threads=10000 --track-origins=yes --error-limit=no --log-file=./${BIN}.valgrind.log --leak-check=full --tool=memcheck --expensive-definedness-checks=yes --show-leak-kinds=definite --vgdb=full --vgdb-error=1 --max-stackframe=115062830400 ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}" # Enable to start GDB server on first error --vgdb=full --vgdb-error=1
  # /usr/local/cuda/bin/compute-sanitizer --tool=memcheck ./${BIN} --gtest_filter="$2"
else
#	eval `dmalloc -d 0 -l log_file -p log-non-free -p check-fence -p check-funcs`
	./${BIN} --gtest_filter="$1" >> "${ONLINETEST_OUTPUT}" 2>&1 &
fi
