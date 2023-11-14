/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <sstream>
#include <gtest/gtest.h>

#include "test_harness.hpp"
#include "../include/onlinesvr.hpp"
#include "../include/kernel_factory.hpp"
#include "kernel_basic_integration_test.hpp"


#define MODEL_FILE ("../SVRRoot/OnlineSVR/test/test_data/features_level_0.txt")
#define DUMP_FILE ("global_alignment_2_train_predict_dump.txt")
#define SAVED_MODEL_FILE ("../SVRRoot/OnlineSVR/test/test_data/online_svr_final_rbf2_test.txt")


TEST(gak_train_predict_2, basic_integration)
{
//    kernel_basic_integration_test(MODEL_FILE, DUMP_FILE, SAVED_MODEL_FILE);
}
