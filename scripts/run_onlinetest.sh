#!/usr/bin/env bash

if [ -z "$SVRWAVE_TEST_WINDOW" ]; then
  export SVRWAVE_TEST_WINDOW=250 # 115
fi
export BIN=OnlineSVR-test
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. ${SCRIPT_DIR}/run_test.sh $@
