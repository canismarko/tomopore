#include <hdf5.h>

#include "matrix.h"

#ifndef TP_FILTERS
enum operation {Min, Max};

// Type definitions
typedef struct ThreadPayload {
  DIM row_start;
  DIM row_end;
  DIM n_cols;
  DIM new_islc;
  Matrix2D *pore_slice_buffer;
  Matrix3D *working_buffer;
  Matrix3D *kernel;
  enum operation op;
} ThreadPayload;

#define TP_FILTERS
#endif

/* Function defintions */
/* =================== */
static float tp_apply_kernel(Matrix3D *subvolume, Matrix3D *kernel, DIM islc, DIM irow, DIM icol, enum operation op);

char tp_apply_filter(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel, enum operation op);

void print_progress(DIM current, DIM total, const char* desc);

static char tp_subtract_datasets(hid_t src_ds1, hid_t src_ds2, hid_t dest_ds);

static void *tp_apply_kernel_thread(void *args);

char tp_extract_pores(hid_t volume_ds, hid_t pores_ds, hid_t h5fp, char *name, DIM min_pore_size, DIM max_pore_size);

char tp_extract_lead(hid_t volume_ds, hid_t pores_ds, hid_t h5fp, char *name, DIM min_pore_size, DIM max_pore_size);

char tp_apply_black_tophat(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel);
char tp_apply_white_tophat(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel);

char tp_apply_closing(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel);
char tp_apply_opening(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel);

char tp_apply_dilation(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel);
char tp_apply_erosion(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel);

void tp_normalize_kernel(Matrix3D *kernel);

void tp_ellipsoid(Matrix3D *kernel);
void tp_box(Matrix3D *kernel);

static void roll_buffer(Matrix3D *buffer);

static DIM min_d(DIM x, DIM y);
static DIM max_d(DIM x, DIM y);
