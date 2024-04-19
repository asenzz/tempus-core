#!/usr/bin/env bash

export DBG="/usr/bin/gdb"
# export DBG=/usr/local/cuda/bin/cuda-gdb


export DAEMON_DIR=${PWD} # /mnt/faststore/repo/tempus-core/build
export DAEMON_CONFIG=$DAEMON_DIR/../config/daemon.config

export LOGDIR=/mnt/slowstore/var/log/
export ONLINETEST_OUTPUT=${LOGDIR}svronline_tests.log
export BUSINESSTEST_OUTPUT=${LOGDIR}svrbusiness_tests.log
export BACKTEST_OUTPUT=${LOGDIR}svrbacktest.log
export WEB_OUTPUT=${LOGDIR}svrweb.log
export DAEMON_OUTPUT=${LOGDIR}svrdaemon_test.log
export TEST_DB_INIT_SCRIPTS=${DAEMON_DIR}/../../tempus-db/dbscripts/init_db.sh

GR='\033[1;32m'
NC='\033[0m' # No Color


if [ `whoami` != 'root' ]
then
	echo '\n${GR}B-root!\n'
	#exit 1
fi

if [ ! -d "../libs/oemd_fir_masks_xauusd_1s_backtest/" ]; then mkdir "../libs/oemd_fir_masks_xauusd_1s_backtest/"; fi
if [ ! -d "../libs/oemd_fir_masks_xauusd_1s/" ]; then mkdir "../libs/oemd_fir_masks_xauusd_1s/"; fi

NUM_THREADS=$(( 1 * $(grep -c ^processor /proc/cpuinfo) ))
printf "\n\n${GR}Default thread pool size is ${NUM_THREADS} threads.${NC}\n\n"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:"/opt/intel/oneapi/compiler/latest/lib":/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/:/opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8
export LD_PRELOAD=`/usr/local/bin/jemalloc-config --libdir`/libjemalloc.so.`jemalloc-config --revision`
export LSAN_OPTIONS=suppressions=../sanitize-blacklist.txt

export OMP_NESTED=true
# export KMP_AFFINITY=compact # Kills performance
# export KMP_HW_SUBSET=2s,10c,2t # Slows down noticeably, find better values
export OMP_WAIT_POLICY=PASSIVE                            # Sets spincount to zero
# export OMP_SCHEDULE=dynamic,1                           # May disable nesting (bug in OMP?)
export MAX_ACTIVE_LEVELS=$(( 10 * $NUM_THREADS ))         # Nested depth
export OMP_THREAD_LIMIT=$(( 20 * $MAX_ACTIVE_LEVELS ))    # Increase with RAM size
# export OMP_NUM_THREADS=${NUM_THREADS}
export CILK_NWORKERS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export MKL_DYNAMIC="FALSE"
# export VISIBLE_GPUS="0" # Disable GPUs
# export CUDA_VISIBLE_DEVICES=VISIBLE_GPUS

ulimit -s 8192
ulimit -i unlimited
ulimit -n 100000

if [ `whoami` == 'root' ]
then
  echo 1073741820 > /proc/sys/kernel/threads-max
  echo 1073741820 > /proc/sys/vm/max_map_count
  echo 1073741 > /proc/sys/kernel/pid_max
fi
