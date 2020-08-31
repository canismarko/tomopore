#define TRUE 1
#define FALSE 0

/* A global config struct that can be shared amongst files. One .c
   file should contain ``#define EXTERN`` to define the variable. */

typedef struct {
  int n_threads;
  int verbose;
  int quiet;
} Config;

#ifdef TP_CONFIG_EXTERN
extern Config config
#endif

#ifndef TP_CONFIG_EXTERN
Config config;
#define TP_CONFIG_EXTERN
#endif
