#!/usr/bin/env bash

if [[ -z "${ONEAPI_ROOT}" ]]; then
  export ONEAPI_ROOT=/opt/intel/oneapi
fi

if [[ -z "${SETVARS_COMPLETED}" ]]; then
  source ${ONEAPI_ROOT}/setvars.sh --include-intel-llvm intel64 lp64
fi

export NICENESS=19
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export ARTELYS_LICENSE=/opt/knitro/licenses/artelys_lic_8817_ASEN_2024-09-11_trial_knitro_97-65-2f-5a-81.txt
# export CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=cublas.log
# export CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=cublasLt.log
export PETSC_MATH_LIB_PRECISION=double
export MPIEXEC=${ONEAPI_ROOT}/mpi/latest/bin/mpiexec

export scriptname="$(basename $0)"

killwait() {
  while (pidof -csn $1); do
    echo "${scriptname}: Stopping $1 . . ."
    pkill $1
    sleep 1
  done
}

export UBSAN_OPTIONS=print_stacktrace=1:print_suppressions=1:use_unaligned=1:report_objects=1:log_path=/tmp/${BIN}.ubsan.log
export LSAN_OPTIONS=log_threads=1:use_unaligned=1:report_objects=1:log_path=/tmp/${BIN}.lsan.log # print_suppressions=1:suppressions=
export ASAN_OPTIONS=detect_container_overflow=true:protect_shadow_gap=0:detect_invalid_pointer_pairs=1:replace_intrin=1:detect_leaks=1:debug=true:check_initialization_order=true:detect_stack_use_after_return=true:strict_string_checks=true:use_odr_indicator=true:log_path=/tmp/${BIN}.asan.log:verbosity=0:log_threads=1:verify_asan_link_order=0,detect_odr_violation=0,alloc_dealloc_mismatch=0
export TSAN_OPTIONS=log_path=/tmp/${BIN}.tsan.log

# export LD_PRELOAD="${LD_PRELOAD}:libduma.so"
# export DUMA_OPTIONS=debug=1,log=/tmp/${BIN}.duma.log

export LD_PRELOAD="${LD_PRELOAD}:${ONEAPI_ROOT}/compiler/latest/lib/libomptarget.sycl.wrap.so"

export VGRIND=/usr/local/bin/valgrind
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/lib"

# Debugger
# export DBG=/usr/bin/gdb # GNU debugger for GCC builds
# export DBG=${ONEAPI_ROOT}/debugger/latest/opt/debugger/bin/gdb-oneapi # Intel debugger for ICPX builds
export DBG=/usr/local/cuda/bin/cuda-gdb # NVidia

# export PERF=${ONEAPI_ROOT}/vtune/2024.1/bin64/amplxe-perf
export PERF=/usr/bin/perf
export PROFGEN=${ONEAPI_ROOT}/compiler/latest/bin/compiler/llvm-profgen

export DAEMON_DIR=${PWD} # /mnt/faststore/repo/tempus-core/build
export DAEMON_CONFIG=$DAEMON_DIR/../config/daemon.config

export LOGDIR=/mnt/slowstore/var/log/
export ONLINETEST_OUTPUT=${LOGDIR}svronline_tests.log
export BUSINESSTEST_OUTPUT=${LOGDIR}svrbusiness_tests.log
export BACKTEST_OUTPUT=${LOGDIR}svrbacktest.log
export WEB_OUTPUT=${LOGDIR}svrweb.log
export DAEMON_OUTPUT=${LOGDIR}svrdaemon.log
export TEST_DB_INIT_SCRIPTS=${DAEMON_DIR}/../../tempus-db/dbscripts/init_db.sh

GR='\033[1;32m'
NC='\033[0m' # No Color


if [ `whoami` != 'root' ]
then
	echo '\n${GR}B-root!\n'
	#exit 1
fi

if [ ! -d "../lib/oemd_fir_masks_xauusd_1s_backtest/" ]; then mkdir "../lib/oemd_fir_masks_xauusd_1s_backtest/"; fi
if [ ! -d "../lib/oemd_fir_masks_xauusd_1s/" ]; then mkdir "../lib/oemd_fir_masks_xauusd_1s/"; fi

NUM_THREADS=$(( 1 * $(grep -c ^processor /proc/cpuinfo) ))
printf "\n\n${GR}Default thread pool size is ${NUM_THREADS} threads.${NC}\n\n"

# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}${ONEAPI_ROOT}/compiler/latest/lib:${ONEAPI_ROOT}/mkl/latest/lib/intel64:${ONEAPI_ROOT}/compiler/latest/linux/compiler/lib/intel64_lin/:${ONEAPI_ROOT}/tbb/latest/lib/intel64/gcc4.8"
export LD_PRELOAD="${LD_PRELOAD}:`/usr/local/bin/jemalloc-config --libdir`/libjemalloc.so.`jemalloc-config --revision`"
# export LSAN_OPTIONS=suppressions=../sanitize-blacklist.txt

export OMP_NESTED=true
# export OMP_CANCELLATION=true # Seems to slow down OMP code
# export OMP_PROC_BIND=spread # TODO Tune
# export OMP_STACKSIZE=16M # TODO Tune
# export KMP_AFFINITY=compact # Kills performance TODO Tune
# export KMP_HW_SUBSET=2s,10c,2t # Slows down noticeably, find better values
export OMP_WAIT_POLICY=PASSIVE                             # Sets spincount to zero
export OMP_SCHEDULE=static,1                               # May disable nesting (bug in OMP?)
export MAX_ACTIVE_LEVELS=10000                             # Nested depth
export OMP_THREAD_LIMIT=10000 # $(( 10 * $NUM_THREADS ))   # Increase with RAM size
export OMP_NUM_THREADS=4 # ${NUM_THREADS}
# export CILK_NWORKERS=${NUM_THREADS}
export MKL_NUM_THREADS=4 # ${NUM_THREADS}
export MAGMA_NUM_THREADS=4 # ${NUM_THREADS}
export MKL_DYNAMIC="FALSE"
# export VISIBLE_GPUS="0" # Disable GPUs
# export CUDA_VISIBLE_DEVICES=VISIBLE_GPUS
export MKL_ENABLE_INSTRUCTIONS=AVX2
export MKL_THREADING_LAYER=INTEL
export MKL_INTERFACE_LAYER=LP64

export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_ASYNC_PROGRESS=1
export MPICH_NEMESIS_ASYNC_PROGRESS=1

ulimit -s 8192
ulimit -i unlimited
ulimit -n 100000
ulimit -c unlimited
# ulimit -v unlimited

if [ `whoami` == 'root' ]
then
  echo 1073741820 > /proc/sys/kernel/threads-max
  echo 1073741820 > /proc/sys/vm/max_map_count
  echo 1073741 > /proc/sys/kernel/pid_max
  echo 0 > /proc/sys/kernel/kptr_restrict
  echo -1 > /proc/sys/kernel/perf_event_paranoid
  #echo 1 > /sys/bus/pci/devices/0000:82:00.0/remove
  #echo 1 > /sys/bus/pci/devices/0000:04:00.0/remove
  echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

  /usr/bin/nvidia-smi -pl 100
  /usr/bin/nvidia-smi -pm 1
  /usr/bin/nvidia-smi -c 0
  /usr/bin/nvidia-smi -am 0
  /usr/bin/nvidia-smi -mig 0
  # /usr/bin/nvidia-smi --auto-boost-default=1
  /usr/bin/nvidia-smi -den 0
fi

/usr/sbin/sysctl --system > /dev/null
# Recommended sysctl config from https://www.xomedia.io/linux-system-tuning/
# kernel.shmmax = 709743345664
# kernel.shmall = 173015040
# kernel.shmmni = 8192
# kernel.sem = 2048 65536 2048 1024
# kernel.msgmni = 4096
# kernel.msgmnb = 512000
# kernel.msgmax = 65535
# kernel.hung_task_timeout_secs = 0
# vm.swappiness = 0
echo 1 > /proc/sys/vm/overcommit_memory # default 2
# vm.overcommit_ratio = 100
# vm.min_free_kbytes = 1048576
# vm.dirty_expire_centisecs = 500
# vm.dirty_ratio = 20
# vm.dirty_background_ratio = 3
# vm.dirty_writeback_centisecs = 100
# net.core.somaxconn = 8192
# net.core.netdev_max_backlog = 300000
# net.core.netdev_budget = 3000
# net.core.optmem_max = 33554432
# net.core.rmem_max = 33554432
# net.core.wmem_max = 33554432
# net.core.rmem_default = 8388608
# net.core.wmem_default = 8388608
# net.ipv4.tcp_mem = 18605184 24806912 37210368
# net.ipv4.tcp_rmem = 4096 87380 33554432
# net.ipv4.tcp_wmem = 4096 65536 33554432
# net.ipv4.tcp_tw_reuse = 1
# net.ipv4.tcp_keepalive_time = 180
# net.ipv4.tcp_keepalive_intvl = 60
# net.ipv4.tcp_keepalive_probes = 9
# net.ipv4.tcp_retries2 = 5
# net.ipv4.tcp_sack = 1
# net.ipv4.tcp_dsack = 0
# net.ipv4.tcp_syn_retries = 1
# net.ipv4.tcp_timestamps = 1
# net.ipv4.tcp_window_scaling = 1
# net.ipv4.tcp_syncookies = 0
# fs.file-max = 6815744
# fs.aio-max-nr = 3145728
# vm.nr_hugepages = 512
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo 1 > /proc/sys/vm/page_lock_unfairness
echo 0 > /proc/sys/kernel/numa_balancing
sysctl vm.vfs_cache_pressure=50
