#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

// Type definitions
typedef uint16_t DIM;
typedef float VEC;
typedef float DTYPE;
typedef struct {
  DIM nslices;
  DIM nrows;
  DIM ncolumns;
  DTYPE arr[];
} Matrix3D;
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

// Function declarations
float tp_apply_kernel(Matrix3D *arr, Matrix3D *kernel, uint16_t islc, uint16_t irow, uint16_t icol);

Matrix3D *tp_matrixmalloc(DIM n_slices, DIM n_rows, DIM n_columns);

uint16_t tp_num_iters(float *arr, uint16_t dimension);

void tp_normalize_kernel(Matrix3D *kernel);

void tp_ellipsoid(Matrix3D *kernel);

void tp_box(Matrix3D *kernel);

uint64_t tp_indices(Matrix3D *vol, DIM islice, DIM irow, DIM icolumn);
