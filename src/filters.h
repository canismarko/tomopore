#include <hdf5.h>

#include "matrix.h"


namespace tomopore {

  #ifndef TP_FILTERS

  enum operation {Min, Max};

  // Type definitions
  typedef struct ThreadPayload {
    DIM row_start;
    DIM row_end;
    DIM n_cols;
    DIM new_islc;
    tomopore::Matrix2D *pore_slice_buffer;
    tomopore::Matrix3D *working_buffer;
    tomopore::Matrix3D *kernel;
    enum operation op;
  } ThreadPayload;

  #define TP_FILTERS
  #endif

  /* Function defintions */
  /* =================== */
  static float apply_kernel(tomopore::Matrix3D *subvolume, tomopore::Matrix3D *kernel, DIM islc, DIM irow, DIM icol, enum operation op);

  char apply_filter(hid_t src_ds, hid_t dest_ds, tomopore::Matrix3D *kernel, enum operation op);

  void print_progress(DIM current, DIM total, const char* desc);

  static char subtract_datasets(hid_t src_ds1, hid_t src_ds2, hid_t dest_ds);

  static void *apply_kernel_thread(void *args);

  char extract_pores(hid_t volume_ds, hid_t pores_ds, hid_t h5fp, char *name, DIM min_pore_size, DIM max_pore_size);

  char extract_lead(hid_t volume_ds, hid_t pores_ds, hid_t h5fp, char *name, DIM min_pore_size, DIM max_pore_size);

  char apply_black_tophat(hid_t src_ds, hid_t dest_ds, tomopore::Matrix3D *kernel);
  char apply_white_tophat(hid_t src_ds, hid_t dest_ds, tomopore::Matrix3D *kernel);

  char apply_closing(hid_t src_ds, hid_t dest_ds, tomopore::Matrix3D *kernel);
  char apply_opening(hid_t src_ds, hid_t dest_ds, tomopore::Matrix3D *kernel);

  char apply_dilation(hid_t src_ds, hid_t dest_ds, tomopore::Matrix3D *kernel);
  char apply_erosion(hid_t src_ds, hid_t dest_ds, tomopore::Matrix3D *kernel);

  void normalize_kernel(tomopore::Matrix3D *kernel);

  void ellipsoid(tomopore::Matrix3D *kernel);
  void box(tomopore::Matrix3D *kernel);

  static void roll_buffer(tomopore::Matrix3D *buffer);

  static DIM min_d(DIM x, DIM y);
  static DIM max_d(DIM x, DIM y);

}
