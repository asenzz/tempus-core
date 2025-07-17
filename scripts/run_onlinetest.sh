#!/usr/bin/env bash

if [ -z "$SVRWAVE_TEST_WINDOW" ]; then
  export SVRWAVE_TEST_WINDOW=641 # 115
fi
export BIN=OnlineSVR-test

source ../scripts/setup_tempus_env.sh
cd "${DAEMON_DIR}" || exit

killwait ${BIN}
rm -f /dev/shm/sem.svrwave_gpu_sem

echo Test window is ${}SVRWAVE_TEST_WINDOW}.
if [[ $1 == "-d" ]]; then # Debug
  echo "TBB does exception testing on start, ignore the first exception!"
	${DBG} --ex 'catch throw' --ex run --directory=${PWD}/../SVRRoot --se ./${BIN} --args ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
elif [[ $1 == "-v" ]]; then # Valgrind
  # export MALLOC_CONF="prof:true,prof_active:true,prof_prefix:jeprof.out,lg_prof_interval:30,lg_prof_sample:19" # jemalloc profiling doesn't work?

  # Valgrind thread sanitizer
  $VGRIND --tool=helgrind --free-is-write=yes --history-backtrace-size=64 --max-threads=100000 --error-limit=no --log-file=./${BIN}.valgrind.log --max-stackframe=115062830400 ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"

  # Valgrind memcheck
  # $VGRIND --max-threads=100000 --track-origins=yes --error-limit=no --log-file=./${BIN}.valgrind.log --leak-check=full --tool=memcheck --expensive-definedness-checks=yes --show-leak-kinds=definite --max-stackframe=115062830400 ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}" # Enable to start GDB server on first error: --vgdb=full --vgdb-error=1


  # Valgrind stack audit
  # $VGRIND --max-threads=100000 --error-limit=no --log-file=${BIN}.valgrind.log --massif-out-file=${BIN}.massif.out --tool=massif --depth=10 --threshold=10.0 --peak-inaccuracy=10.0 --detailed-freq=100 --max-stackframe=115062830400 ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}" # Enable to start GDB server on first error: --vgdb=full --vgdb-error=1

  # CUDA Computer sanitizer
  # /usr/local/cuda/bin/compute-sanitizer --max-connections=1000 --target-processes=all --log-file=/tmp/${BIN}.compute-sanitizer.log --check-device-heap=yes --coredump-behavior=exit --demangle=full --port=16000 --tool=memcheck --require-cuda-init=no --leak-check=full --check-api-memory-access=yes --report-api-errors=all --missing-barrier-init-is-fatal=yes --check-cache-control --force-blocking-launches --launch-timeout=0 --track-unused-memory --kernel-name regex='.*G_distances.*' ./${BIN} --gtest_filter="$2"

  # ByteHound similar to Valgrind
  # export MEMORY_PROFILER_LOG=warn
  # LD_PRELOAD=/usr/local/lib/libbytehound.so ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
elif [[ $1 == "-p" ]]; then # Profile CPU
  PERF_DATA=${BIN}.perf.data
  SAMPLE_PERIOD=1000003
  $PERF record -b -e BR_INST_RETIRED.NEAR_TAKEN:uppp,BR_MISP_RETIRED.ALL_BRANCHES:upp -c $SAMPLE_PERIOD -o $PERF_DATA -- ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
  # sep -start -out unpredictable.tb7 -ec BR_INST_RETIRED.NEAR_TAKEN:PRECISE=YES:SA=1000003:pdir:lbr:USR=YES,BR_MISP_RETIRED.ALL_BRANCHES:PRECISE=YES:SA=1000003:lbr:USR=YES -lbr no_filter:usr -perf-script event,ip,brstack -app ./${BIN} --gtest_filter="$2" 2>&1 | tee -a "${ONLINETEST_OUTPUT}"
  $PROFGEN --format text --output=$PERF_DATA.freq.prof --binary=$BIN --sample-period=${SAMPLE_PERIOD} --perf-event=BR_INST_RETIRED.NEAR_TAKEN:uppp --perfdata=$PERF_DATA
  $PROFGEN --format text --output=$PERF_DATA.misp.prof --binary=$BIN --sample-period=${SAMPLE_PERIOD} --perf-event=BR_MISP_RETIRED.ALL_BRANCHES:upp --leading-ip-only --perfdata=$PERF_DATA
  chmod a+rw $PERF_DATA.freq.prof $PERF_DATA.misp.prof $PERF_DATA
elif [[ $1 == "-n" ]]; then # Profile NVidia
  nvprof --dependency-analysis --openmp-profiling --cpu-profiling-show-library --cpu-profiling-percentage-threshold 10 --cpu-profiling --cpu-thread-tracing --trace gpu,api --replay-mode disabled --track-memory-allocations --profile-child-processes --cpu-profiling-percentage-threshold=1 -f --metrics all --replay-mode disabled --analysis-metrics --openmp-profiling --metrics achieved_occupancy --export-profile ${BIN}.nvprof ./${BIN} --gtest_filter="$1" >> "${ONLINETEST_OUTPUT}" 2>&1 & # --track-memory-allocations --metrics all --cpu-thread-tracing
  # nsys profile -r cuda,nvtx,osrt,cublas,cusolver,cusparse,openmp -o ${BIN} ./${BIN} --gtest_filter="$1" >> "${ONLINETEST_OUTPUT}" 2>&1 & # No support for Volta
  # nsys analyze ${BIN}.nsys-rep
elif [[ $1 == "-f" ]]; then # Fork
	$MPIEXEC ./${BIN} --gtest_filter="$2" >> "${ONLINETEST_OUTPUT}" 2>&1 &
  renice -n ${NICENESS} -p $(pidof ${BIN})
else # Vanilla
	$MPIEXEC ./${BIN} --gtest_filter="$1" >> "${ONLINETEST_OUTPUT}" 2>&1
fi
