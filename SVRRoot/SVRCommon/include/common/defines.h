#pragma once

/* Global constants and defines */

// #define NO_PARALLEL // Disable parallelisation macros

#define INTEGRATION_TEST
// #define OUTPUT_TRAINING_DATA // Output data in Paramsys formatted files // TODO Move to daemon config
#define NO_ONLINE_TRAINING // Disable online training, buggy and resource-hungry
#define CTX_PER_GPU 2 // Default streams per GPU

// #define LAST_KNOWN_LABEL
#define EMO_DIFF
// #define INSTANCE_WEIGHTS // TODO Test should it multiply labels too?

// #define VMD_ONLY
#define EMD_ONLY
// #define NEW_PATH

#if defined(VMD_ONLY) && defined(EMD_ONLY)
#pragma error "Both VMD_ONLY and EMD_ONLY are defined"
#endif

#ifdef VMD_ONLY
#define LEVEL_STEP 2
#define MIN_LEVEL_COUNT 4
#elif defined(EMD_ONLY)
#define LEVEL_STEP 1
#define MIN_LEVEL_COUNT 2
#else
#define LEVEL_STEP 2
#define MIN_LEVEL_COUNT 8 // Don't touch
#endif

// Tuning
#define TUNE_HYBRID

// #define FFA_HARD_WALL

#define DEFAULT_ONLINE_ITER_LIMIT 5
#define DEFAULT_STABILIZE_ITERATIONS_COUNT 40

#define __TOSTR(X) #X
#define TOSTR(X) __TOSTR(X)

#define SANITIZE_THREAD THREAD