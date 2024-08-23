#!/usr/bin/env bash

export PARAMTUNE_DIR=/home/zarko/repo/tempus-core/build
export PARAMTUNE_OPTIONS=${PARAMTUNE_DIR}/../config/paramtune-zarko.cfg
export PARAMTUNE_CONFIG=${PARAMTUNE_DIR}/../config/paramtune.app.config
export PARAMTUNE_OUTPUT=/var/log/paramtune.log
NUM_THREADS=8

#`grep -c ^processor /proc/cpuinfo`
if [[ `pgrep paramtune` ]]; then echo "Paramtune is already running!"; exit 0; fi
pkill paramtune
pkill SVRDaemon
/usr/bin/python ../lib/paramtune_get_best_params.py $PARAMTUNE_OUTPUT 16 > /tmp/paramtune.best && rm -f "$PARAMTUNE_OUTPUT"
mv /tmp/paramtune.best $PARAMTUNE_OUTPUT
sleep 1
rm -f /dev/shm/sem.svrwave_gpu_sem /tmp/SVRDaemon_*
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/compilers_and_libraries_2019/linux/mkl/lib/intel64
export OMP_NUM_THREADS=${NUM_THREADS}
cd ${PARAMTUNE_DIR}
ulimit -s 1024
ulimit -i unlimited 
ulimit -n 100000
echo 1073741820 > /proc/sys/kernel/threads-max
echo 1073741820 > /proc/sys/vm/max_map_count
echo 1073741 > /proc/sys/kernel/pid_max 
if [[ $1 == "-d" ]]; then
	MKL_NUM_THREADS=${NUM_THREADS} OMP_NUM_THREADS=${NUM_THREADS} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/compilers_and_libraries_2019/linux/mkl/lib/intel64 gdb --args ./paramtune --app-config ${PARAMTUNE_CONFIG} --options ${PARAMTUNE_OPTIONS} | tee -a ${PARAMTUNE_OUTPUT}.debug 2>&1 
else
	MKL_NUM_THREADS=${NUM_THREADS} OMP_NUM_THREADS=${NUM_THREADS} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/compilers_and_libraries_2019/linux/mkl/lib/intel64 ./paramtune --app-config ${PARAMTUNE_CONFIG} --options ${PARAMTUNE_OPTIONS} >> ${PARAMTUNE_OUTPUT} &
fi
cgclassify -g cpu,cpuset:Tempus $(pidof paramtune)
