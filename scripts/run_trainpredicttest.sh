#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
for TEST_START in 1097 982 867 752 641 526 411 296 207 115; do # Starts at every Monday zero hours
  export SVRWAVE_TEST_WINDOW=$TEST_START
  "${SCRIPT_DIR}"/run_onlinetest.sh manifold_tune_train_predict.basic_integration
done
