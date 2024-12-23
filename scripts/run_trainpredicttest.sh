#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# for TEST_START in 115 230 345 460 575 690 805 920 1035 1150 1265 1380; do
for TEST_START in 1265; do
  export SVRWAVE_TEST_WINDOW=$TEST_START
  "${SCRIPT_DIR}"/run_onlinetest.sh manifold_tune_train_predict.basic_integration
done
