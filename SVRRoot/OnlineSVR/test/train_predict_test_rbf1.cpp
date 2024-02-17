#include <sstream>
#include <gtest/gtest.h>

#include "../include/onlinesvr.hpp"
#include "test_harness.hpp"
#include "kernel_basic_integration_test_smo.hpp"


#define MODEL_FILE ("../SVRRoot/OnlineSVR/test/test_data/batch_train_output_saved01.txt")
#define DUMP_FILE ("rbf_train_predict_dump.txt")
#define SAVED_MODEL_FILE ("../SVRRoot/OnlineSVR/test/test_data/online_svr_final_rbf_test.txt")


TEST(rbf_train_predict, basic_integration)
{
//    kernel_basic_integration_test(MODEL_FILE, DUMP_FILE, SAVED_MODEL_FILE);
}
