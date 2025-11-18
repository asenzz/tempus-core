#!/usr/bin/env bash

if [ -z "$SVRWAVE_TEST_WINDOW" ]; then
  export SVRWAVE_TEST_WINDOW=250
fi

export BIN=SVRBusiness-tests
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source ${SCRIPT_DIR}/run_test.sh $@
