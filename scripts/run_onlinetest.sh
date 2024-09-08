#!/usr/bin/env bash

export SVRWAVE_TEST_WINDOW=460
export BIN=OnlineSVR-test

source ../scripts/setup_tempus_env.sh
cd "${DAEMON_DIR}" || exit

pkill -9 "${BIN}"
rm -f /dev/shm/sem.svrwave_gpu_sem


if [[ $1 == "-d" ]]; then # Debug
  echo "TBB does exception testing on start, ignore the first exception!"
	${DBG} --ex 'catch throw' --ex run --directory=${PWD}/../SVRRoot --se ./${BIN} --args ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
elif [[ $1 == "-v" ]]; then # Valgrind
  export SVRWAVE_TEST_WINDOW=1
  ${VGRIND} --max-threads=100000 --track-origins=yes --error-limit=no --log-file=./${BIN}.valgrind.log --leak-check=full --tool=memcheck --expensive-definedness-checks=yes --show-leak-kinds=definite --vgdb=full --vgdb-error=1 --max-stackframe=115062830400 ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}" # Enable to start GDB server on first error: --vgdb=full --vgdb-error=1
  # /usr/local/cuda/bin/compute-sanitizer --max-connections=1000 --target-processes=all --log-file=/tmp/${BIN}.compute-sanitizer.log --check-device-heap=yes --demangle=full --port=16000 --tool=memcheck --require-cuda-init=no --leak-check=full --check-api-memory-access=yes --missing-barrier-init-is-fatal=yes ./${BIN} --gtest_filter="$2" --launch-timeout=0 # --track-unused-memory --force-blocking-launches --check-warpgroup-mma=yes --check-cache-control
elif [[ $1 == "-p" ]]; then # Profile CPU
  PERF_DATA=${BIN}.perf.data
  ${PERF} record -b -e BR_INST_RETIRED.NEAR_TAKEN:uppp,BR_MISP_RETIRED.ALL_BRANCHES:upp -c 1000003 -o $PERF_DATA -- ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
  # sep -start -out unpredictable.tb7 -ec BR_INST_RETIRED.NEAR_TAKEN:PRECISE=YES:SA=1000003:pdir:lbr:USR=YES,BR_MISP_RETIRED.ALL_BRANCHES:PRECISE=YES:SA=1000003:lbr:USR=YES -lbr no_filter:usr -perf-script event,ip,brstack -app ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
  $PROFGEN --format text --output=$PERF_DATA.freq.prof --binary=$BIN --sample-period=1000003 --perf-event=BR_INST_RETIRED.NEAR_TAKEN:uppp --perfdata=$PERF_DATA
  $PROFGEN --format text --output=$PERF_DATA.misp.prof --binary=$BIN --sample-period=1000003 --perf-event=BR_MISP_RETIRED.ALL_BRANCHES:upp --leading-ip-only --perfdata=$PERF_DATA
  chmod a+rw $PERF_DATA.freq.prof $PERF_DATA.misp.prof $PERF_DATA
elif [[ $1 == "-n" ]]; then # Profile NVidia
  nvprof --dependency-analysis --openmp-profiling --cpu-profiling-show-library --cpu-profiling-percentage-threshold 10 --cpu-profiling --cpu-thread-tracing --trace gpu,api --replay-mode disabled --track-memory-allocations --profile-child-processes --cpu-profiling-percentage-threshold=1 -f --metrics all --replay-mode disabled --analysis-metrics --openmp-profiling --metrics achieved_occupancy --export-profile ${BIN}.nvprof ./${BIN} --gtest_filter="$1" >> "${ONLINETEST_OUTPUT}" 2>&1 & # --track-memory-allocations --metrics all --cpu-thread-tracing
  # nsys profile -r cuda,nvtx,osrt,cublas,cusolver,cusparse,openmp -o ${BIN} ./${BIN} --gtest_filter="$1" >> "${ONLINETEST_OUTPUT}" 2>&1 & # No support for Volta
  # nsys analyze ${BIN}.nsys-rep
else # Vanilla
#	eval `dmalloc -d 0 -l log_file -p log-non-free -p check-fence -p check-funcs`
	nice -n ${NICENESS} ./${BIN} --gtest_filter="$1" >> "${ONLINETEST_OUTPUT}" 2>&1 &
fi
