#pragma once

/* Global constants and defines */

//#define NO_PARALLEL // Disable parallel macros

#define TUNE_KEEP_PREDS (2)

// #define SYSTEMIC_TUNE

#define INTEGRATION_TEST

// #define OUTPUT_LIBSVM_TRAIN_DATA // To output LibSVM train files for every model
// #define OUTPUT_TRAINING_DATA // Output data in Paramsys formatted files // TODO Move to daemon config

#define MIN_LEVEL_COUNT 8

#define PRIMARY_COLUMN "xauusd_avg_bid" // Ignore tuning or validating other input queue columns in case of aux columns
#define NO_MAIN_DECON
//#define LAST_KNOWN_LABEL
// #define EMO_DIFF


#define QUANTIZE_FIXED      10
//#define QUANTIZE_FIXED_MAX  15
//#define QUANTIZE_FIXED_MIN  1

#ifndef QUANTIZE_FIXED
#define QUANTIZE_FEAT_EXP   1
#define QUANTIZE_FEAT_MUL   (600. /* Max number of samples per feature */ * 1. / MAIN_DECON_QUEUE_RES_SECS) // (1./(double(MAIN_DECON_QUEUE_RES_SECS)/double(OEMD_STRETCH_COEF))) // (1. / (3600. / double(OEMD_STRETCH_COEF))) // 3600 is main to aux resolution ratio // prev QUANTIZE_FEAT_MUL 0.001
#endif

#define OFFSET_PRED_MUL     .1 // Prediction is ahead of last-known by main queue resolution * OFFSET_PRED_MUL

// Tuning
// #define TUNE_IDEAL
#define TUNE_HYBRID

#define FFA_ALPHA       .5   // Starting randomness
#define FFA_BETAMIN     .5   // Attraction
#define FFA_GAMMA       1.   // Visibility
// #define FFA_HARD_WALL

// Default configuration
// TODO Replace the defines below with constants
#define DEFAULT_SQL_PROPERTIES_DIR_KEY ("../SVRRoot/SVRPersist/postgres")
#define DEFAULT_LOG_LEVEL_KEY ("info")
#define DEFAULT_DAO_TYPE_KEY ("postgres")
#define DEFAULT_ONLINE_ITER_LIMIT ("4")
#define DEFAULT_STABILIZE_ITERATIONS_COUNT ("16")
#define DEFAULT_ERROR_TOLERANCE ("1e-7")
#define DEFAULT_ONLINESVR_LOG_FILE ("learn_log.txt")
#define DEFAULT_TUNE_PARAMETERS ("0")
#define DEFAULT_SLIDE_COUNT ("13")
#define DEFAULT_TUNE_RUN_LIMIT ("3600")
#define DEFAULT_SCALING_ALPHA ("0.01")
#define DEFAULT_MULTISTEP_LEN ("1")
#define DEFAULT_HARDWARE_CONCURRENCY (16)
#define DEFAULT_FEATURES_MAX_TIME_GAP (bpt::hours(60))

// Default SVR parameters
#define DEFAULT_SVRPARAM_DECON_LEVEL 0
#define DEFAULT_SVRPARAM_CHUNK_IX 0
#define DEFAULT_SVRPARAM_GRAD_LEVEL 0
#define DEFAULT_SVRPARAM_SVR_COST 0
#define DEFAULT_SVRPARAM_SVR_EPSILON 0
#define DEFAULT_SVRPARAM_KERNEL_PARAM_1 0
#define DEFAULT_SVRPARAM_KERNEL_PARAM_2 0
#define DEFAULT_SVRPARAM_DECREMENT_DISTANCE common::C_default_kernel_max_chunk_size
#define DEFAULT_SVRPARAM_ADJACENT_LEVELS_RATIO 1
#define DEFAULT_SVRPARAM_KERNEL_TYPE svr::datamodel::kernel_type_e::PATH
#define DEFAULT_SVRPARAM_LAG_COUNT 1000

constexpr double C_bad_validation = 1e6;
constexpr unsigned CUDA_BLOCK_SIZE = 1024;
constexpr unsigned TILE_WIDTH = 32; // For Path kernel must be 32x32 == 1024 == Nx_local or 16 x 16 = 256, careful!

#ifdef ARMA_DONT_ZERO_INIT
#undef ARMA_DONT_ZERO_INIT
#endif
