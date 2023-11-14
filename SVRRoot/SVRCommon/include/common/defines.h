#pragma once

#include <stdlib.h>

/* Global constants and defines */

//#define NO_PARALLEL

#define TUNE_KEEP_PREDS (2)

#define NEW_SCALING // Shortest time-frame scaling
#define MODEL_TRAIN_OFFSET 0 // Use for MQL5 backtests, shows how many main queue samples backward to shift the signal

#define MANIFOLD_TEST
#ifdef MANIFOLD_TEST
constexpr unsigned TEST_OFFSET_DEFAULT = 115;
#define MANIFOLD_TEST_VALIDATION_WINDOW (getenv("SVRWAVE_TEST_WINDOW") ? strtoul(getenv("SVRWAVE_TEST_WINDOW"), nullptr, 10) : (TEST_OFFSET_DEFAULT))
#endif

// #define OUTPUT_LIBSVM_TRAIN_DATA // To output LibSVM train files for every model
// #define OUTPUT_TRAINING_DATA // Output data in Paramsys formatted files // TODO Move to daemon config


#define PRIMARY_COLUMN "xauusd_avg_bid" // Ignore tuning or validating other input queue columns in case of aux columns
#define NO_MAIN_DECON
constexpr unsigned MAIN_DECON_QUEUE_RES_SECS = 3600;
//#define LAST_KNOWN_LABEL
#define MULTISTEP_PRED_PERIOD       bpt::seconds(MAIN_DECON_QUEUE_RES_SECS)
constexpr unsigned CHUNK_DECREMENT = 6000;
// #define LAST_QUANT_FEAT // The feature is average of QUANTIZE_FIXED number of features vs the last value of the range
// #define EMO_DIFF
// #define EMO_DIFF_FEATS


#define QUANTIZE_FIXED      10
//#define QUANTIZE_FIXED_MAX  15
//#define QUANTIZE_FIXED_MIN  1

#ifndef QUANTIZE_FIXED
#define QUANTIZE_FEAT_EXP   1
#define QUANTIZE_FEAT_MUL   (600. /* Max number of samples per feature */ * 1. / MAIN_DECON_QUEUE_RES_SECS) // (1./(double(MAIN_DECON_QUEUE_RES_SECS)/double(OEMD_STRETCH_COEF))) // (1. / (3600. /* * double(OMEGA_DIVISOR) */ / double(OEMD_STRETCH_COEF))) // 3600 is main to aux resolution ratio // prev QUANTIZE_FEAT_MUL 0.001
#endif

#define OEMD_STRETCH_COEF   1 // MAIN_DECON_QUEUE_RES_SECS * OMEGA_DIVISOR // 10 * MAIN_DECON_QUEUE_RES_SECS * OMEGA_DIVISOR // Fat OEMD transformer
#define OFFSET_PRED_MUL     .1
#define EVENING_FACTOR      0.

// Tuning

// #define TUNE_IDEAL
#define TUNE_HYBRID

#ifdef TUNE_HYBRID
// #define TUNE_HYBRID_IDEAL
// #define HYBRID_QUANTILES
#ifndef TUNE_HYBRID_IDEAL
constexpr unsigned EMO_SLIDE_SKIP = 8; // 2
constexpr unsigned EMO_MAX_J = 13; // 18 // Slides count
constexpr unsigned DEFAULT_EMO_SLIDE_LEN = 270; // 65 // TODO test 5 * 16 + 190 / epsco 20
constexpr unsigned EMO_SLIDE_LEN = DEFAULT_EMO_SLIDE_LEN; // = EMO_TUNE_TEST_SIZE
constexpr unsigned EMO_TUNE_VALIDATION_WINDOW = EMO_SLIDE_LEN - EMO_SLIDE_SKIP * EMO_MAX_J; // 200 seems to be the best validation window for predicting the next 120 hours, setting the window to 120 gives worse results for predicting the next 120 hours
constexpr unsigned EMO_TUNE_TEST_SIZE = EMO_SLIDE_SKIP * EMO_MAX_J + EMO_TUNE_VALIDATION_WINDOW; // = EMO_SLIDE_LEN
#endif
#else
#define TUNE_GAMMA_MAX          1e7 // Max gamma should be below 4e8 when features and labels are in the -1..1 range
#define TUNE_GAMMA_MIN          1 // Min gamma is determined by the input data, lower is better, scaling factors determine min gamma, therefore data scaling factors should provide the lowest min gamma possible
#define EMO_TUNE_VALIDATION_WINDOW 1000
constexpr unsigned EMO_TUNE_TEST_SIZE = 0;
#endif


//#define MAX_TUNE_CHUNKS 1 // Less than num_chunks

#define FFA_ALPHA       .2   // Starting randomness
#define FFA_BETAMIN     .5   // Attraction
#define FFA_GAMMA       1.   // Visibility
// #define FFA_HARD_WALL
#define TUNE_COST_MIN           5e-2
#define TUNE_COST_MAX           1e15
#define DEFAULT_TAU_TUNE        .25
#define DEFAULT_KERNEL_TYPE     DEFAULT_SVRPARAM_KERNEL_TYPE
#define PSO_TOPOLOGY            svr::optimizer::PsoTopology::global
#define PSO_SAVE_STATE_PATH     svr::common::formatter() << "/tmp/pso_state_" << PRIMARY_COLUMN << "_" << _p_svr_parameters->get_decon_level() << ".sav"


#define MIN_COST_EXP 6 // Used in unit tests
#define MAX_COST_EXP 7

#define SAVE_DECON
#define SAVE_OUTPUT_LOCATION    ("/mnt/slowstore/var/tmp/")


// #define COMPRESS_LABEL_TIME 6


// Default configuration
// TODO Replace the defines below with constants
#define DEFAULT_SQL_PROPERTIES_DIR_KEY ("../SVRRoot/SVRPersist/postgres")
#define DEFAULT_LOG_LEVEL_KEY ("info")
#define DEFAULT_DAO_TYPE_KEY ("postgres")
#define DEFAULT_DONT_UPDATE_R_MATRIX ("0")
#define DEFAULT_MAIN_COLUMNS_AUX ("1")
#define DEFAULT_MAX_SMO_ITERATIONS ("300000")
#define DEFAULT_CASCADE_REDUCE_RATIO ("0.6")
#define DEFAULT_CASCADE_MAX_SEGMENT_SIZE ("5000")
#define DEFAULT_CASCADE_BRANCHES_COUNT ("2")
#define DEFAULT_DISABLE_CASCADED_SVM ("0")
#define DEFAULT_ONLINE_ITERS_LIMIT_MULT ("2")
#define DEFAULT_LEARN_ITER_LIMIT ("500")
#define DEFAULT_SMO_EPSILON_DIVISOR ("100.0")
#define DEFAULT_SMO_COST_DIVISOR ("1e+4")
#define DEFAULT_STABILIZE_ITERATIONS_COUNT ("10000")
#define DEFAULT_DEFAULT_NUMBER_VARIATIONS ("50")
#define DEFAULT_ERROR_TOLERANCE ("1e-7")
#define DEFAULT_ONLINESVR_LOG_FILE ("learn_log.txt")
#define DEFAULT_MAX_VARIATIONS ("100")
#define DEFAULT_ENABLE_COMB_VALIDATE ("1")
#define DEFAULT_TUNE_PARAMETERS ("0")
#define DEFAULT_FUTURE_PREDICT_COUNT ("1000")
#define DEFAULT_SLIDE_COUNT ("10")
#define DEFAULT_TUNE_RUN_LIMIT ("1")
#define DEFAULT_SCALING_ALPHA ("0.01")
#define DEFAULT_ALL_AUX_LEVELS ("0")
#define DEFAULT_OEMD_FIND_FIR_COEFFICIENTS ("0")
#define DEFAULT_COMB_TRAIN_COUNT ("7")
#define DEFAULT_COMB_VALIDATE_COUNT ("3")
#define DEFAULT_COMB_VALIDATE_LIMIT ("50")
#define MAX_ITER_SMO (2147483647UL)
#define DEFAULT_MULTISTEP_LEN ("1")
#define DEFAULT_SVR_PARAMTUNE_COLUMN ("ALL")
#define DEFAULT_SVR_PARAMTUNE_LEVEL ("ALL")
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
#define DEFAULT_SVRPARAM_DECREMENT_DISTANCE CHUNK_DECREMENT
#define DEFAULT_SVRPARAM_ADJACENT_LEVELS_RATIO 1
#define DEFAULT_SVRPARAM_KERNEL_TYPE (svr::datamodel::kernel_type_e::PATH)
#define DEFAULT_SVRPARAM_LAG_COUNT 800


#define POSTGRES_DATETIME_FORMAT_STRING ("%Y.%m.%d,%H:%M")

#define PSO_INDEX_COST (0)
#define PSO_INDEX_EPSILON (1)
#define PSO_INDEX_KERNEL_PARAM1 (2)
#define PSO_INDEX_KERNEL_PARAM2 (3)
#define PSO_INDEX_DECREMENTAL_DISTANCE (4)
#define PSO_INDEX_ADJACENT_LEVELS_RATIO (5)
#define PSO_INDEX_LAG_COUNT (6)


#ifndef MANIFOLD_TEST
#define MANIFOLD_TEST_VALIDATION_WINDOW 0
#ifdef MODEL_TRAIN_OFFSET
#undef MODEL_TRAIN_OFFSET
#define MODEL_TRAIN_OFFSET 0
#endif
#endif

constexpr double BAD_VALIDATION = 1000;

constexpr unsigned CUDA_BLOCK_SIZE = 256;
constexpr unsigned TILE_WIDTH = 16; // For Path kernel must be 32x32 == 1024 == Nx_local or 16 x 16 = 256, careful!
