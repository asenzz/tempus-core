#!/usr/bin/env bash
export SVRWAVE_TEST_WINDOW=460
export BIN=OnlineSVR-test
source ../scripts/setup_tempus_env.sh
cd "${DAEMON_DIR}" || exit

pkill -9 "${BIN}"
rm -f /dev/shm/sem.svrwave_gpu_sem


if [[ $1 == "-d" ]]; then
  echo "TBB does exception testing on start, ignore the first exception!"
	${DBG} --ex 'catch throw' --ex run --directory=${PWD}/../ --se ./${BIN} --args ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
elif [[ $1 == "-v" ]]; then
  export SVRWAVE_TEST_WINDOW=1
  ${VGRIND} --max-threads=100000 --track-origins=yes --error-limit=no --log-file=./${BIN}.valgrind.log --leak-check=full --tool=memcheck --expensive-definedness-checks=yes --show-leak-kinds=definite --vgdb=full --vgdb-error=1 --max-stackframe=115062830400 ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}" # Enable to start GDB server on first error: --vgdb=full --vgdb-error=1
  # /usr/local/cuda/bin/compute-sanitizer --log-file=/tmp/${BIN}.compute-sanitizer.log --check-device-heap=yes --demangle=full --force-blocking-launches --check-warpgroup-mma=yes --tool=memcheck --check-cache-control --require-cuda-init=no --leak-check=full --track-unused-memory --check-api-memory-access=yes --missing-barrier-init-is-fatal=yes ./${BIN} --gtest_filter="$2"
elif [[ $1 == "-p" ]]; then
  PERF_DATA=${BIN}.perf.data
  ${PERF} record -b -e BR_INST_RETIRED.NEAR_TAKEN:uppp,BR_MISP_RETIRED.ALL_BRANCHES:upp -c 1000003 -o $PERF_DATA -- ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
  # sep -start -out unpredictable.tb7 -ec BR_INST_RETIRED.NEAR_TAKEN:PRECISE=YES:SA=1000003:pdir:lbr:USR=YES,BR_MISP_RETIRED.ALL_BRANCHES:PRECISE=YES:SA=1000003:lbr:USR=YES -lbr no_filter:usr -perf-script event,ip,brstack -app ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
  $PROFGEN --format text --output=$PERF_DATA.freq.prof --binary=$BIN --sample-period=1000003 --perf-event=BR_INST_RETIRED.NEAR_TAKEN:uppp --perfdata=$PERF_DATA
  $PROFGEN --format text --output=$PERF_DATA.misp.prof --binary=$BIN --sample-period=1000003 --perf-event=BR_MISP_RETIRED.ALL_BRANCHES:upp --leading-ip-only --perfdata=$PERF_DATA
  chmod a+rw $PERF_DATA.freq.prof $PERF_DATA.misp.prof $PERF_DATA
else
#	eval `dmalloc -d 0 -l log_file -p log-non-free -p check-fence -p check-funcs`
	./${BIN} --gtest_filter="$1" >> "${ONLINETEST_OUTPUT}" 2>&1 &
fi
