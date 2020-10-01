#define TRUE 1
#define FALSE 0

/* A global config struct that can be shared amongst files. One .c
   file should contain ``#define EXTERN`` to define the variable. */

typedef struct {
  int n_threads;
  int verbose;
  int quiet;
} Config;

extern Config config;
