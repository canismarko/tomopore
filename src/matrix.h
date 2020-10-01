#include <stdint.h>
#include <cuda_runtime.h>

#ifndef TP_MATRIX

// Type definitions
typedef uint64_t DIM;
typedef uint64_t DDIM;
typedef double VEC;
typedef float DTYPE;
typedef struct {
  DIM nslices;
  DIM nrows;
  DIM ncolumns;
  DTYPE arr[];
} Matrix3D;
typedef struct {
  DIM nrows;
  DIM ncolumns;
  DTYPE arr[];
} Matrix2D;
typedef struct {
  DIM z;
  DIM y;
  DIM x;
} Point;
typedef struct {
  VEC z;
  VEC y;
  VEC x;
} Vector;

#define TP_MATRIX
#endif

/* Function defintions */
/* =================== */
DDIM tp_indices(Matrix3D *vol, DIM islice, DIM irow, DIM icolumn);
__host__ __device__ DDIM tp_indices2d(Matrix2D *vol, DIM irow, DIM icolumn);
