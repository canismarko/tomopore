#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

// Type definitions
typedef uint16_t DIM;
typedef uint64_t DDIM;
typedef float VEC;
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
typedef struct ThreadPayload {
  DIM row_start;
  DIM row_end;
  DIM n_cols;
  DIM new_islc;
  Matrix2D *pore_slice_buffer;
  Matrix3D *working_buffer;
  Matrix3D *kernel;
  void (*filter_func)(DTYPE volume_val, DTYPE kernel_val, double *running_total, DDIM *running_count, char *is_first_round);
} ThreadPayload;

// Function declarations
float tp_apply_kernel(Matrix3D *subvolume, Matrix3D *kernel, DIM islc, DIM irow, DIM icol,
		      void (*filter_func)(DTYPE volume_val, DTYPE kernel_val, double *running_total, DDIM *running_count, char *is_first_round));

char tp_apply_filter(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel,
		     void (*filter_func)(DTYPE volume_val, DTYPE kernel_val, double *running_total, DDIM *running_count, char *is_first_round));
char tp_extract_pores(hid_t volume_ds, hid_t pores_ds, hid_t h5fp, DIM min_pore_size, DIM max_pore_size);

Matrix3D *tp_matrixmalloc(DIM n_slices, DIM n_rows, DIM n_columns);

Matrix2D *tp_matrixmalloc2d(DIM n_rows, DIM n_columns);

uint16_t tp_num_iters(float *arr, uint16_t dimension);

void tp_normalize_kernel(Matrix3D *kernel);

void tp_ellipsoid(Matrix3D *kernel);

void tp_box(Matrix3D *kernel);

DDIM tp_indices(Matrix3D *vol, DIM islice, DIM irow, DIM icolumn);
DDIM tp_indices2d(Matrix2D *vol, DIM irow, DIM icolumn);

void roll_buffer(Matrix3D *buffer);

DIM min_d(DIM x, DIM y);
DIM max_d(DIM x, DIM y);

/* Applied kernel functions */
/* ======================== */

/* Each one of these functions can be given to ``tp_apply_kernel`` to */
/* calculate the metric for that pixel and apply to it to the running */
/* total. They must all have matching signatures, though do no */
/* necessarily have to use all the values that are given. *running_count* */
/* is intended to hold the number of inputs that have been processed and */
/* can be useful when calculating means, etc. *is_first_round* can be */
/* set by the function to keep track of whether initialization is */
/* needed. */

void tp_apply_max(DTYPE volume_val, DTYPE kernel_val, double *running_total, DDIM *running_count, char *is_first_round);
void tp_apply_min(DTYPE volume_val, DTYPE kernel_val, double *running_total, DDIM *running_count, char *is_first_round);
