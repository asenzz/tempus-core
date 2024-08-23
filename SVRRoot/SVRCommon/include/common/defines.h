#pragma once

/* Global constants and defines */

// #define NO_PARALLEL // Disable parallel macros

#define INTEGRATION_TEST
#define DO_SHIFT
// #define OUTPUT_TRAINING_DATA // Output data in Paramsys formatted files // TODO Move to daemon config

#define MIN_LEVEL_COUNT 8 // Don't touch

#define NO_MAIN_DECON // Don't decompose main input queue
// #define LAST_KNOWN_LABEL
#define EMO_DIFF

// Tuning
#define TUNE_HYBRID

// #define FFA_HARD_WALL

#define DEFAULT_ONLINE_ITER_LIMIT 5
#define DEFAULT_STABILIZE_ITERATIONS_COUNT 40






#define __TOSTR(X) #X
#define TOSTR(X) __TOSTR(X)
