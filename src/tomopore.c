#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

// Global constants
#define DS_SRC_NAME "volume"
#define DS_DST_NAME "pores"
#define DS_SIZE 262144
#define PORE_MIN_SIZE 5
#define PORE_MAX_SIZE 31

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
Matrix3D *tp_matrixmalloc(DIM n_slices, DIM n_rows, DIM n_columns);
uint16_t tp_num_iters(float *arr, uint16_t dimension);
void tp_normalize_kernel(Matrix3D *kernel);
void tp_ellipsoid(Matrix3D *kernel);
void tp_box(Matrix3D *kernel);
uint64_t tp_indices(Matrix3D *vol, DIM islice, DIM irow, DIM icolumn);


/* Apply a series of filters to the volume using 3D kernels This is done */
/* with a 3D kernel so that we don't lose data, but that means lots more */
/* memory. To avoid running out of memory, intermediate arrays are saved */
/* in HDF5 datasets. */


Matrix3D *tp_matrixmalloc(DIM n_slices, DIM n_rows, DIM n_columns) {
  // Allocated memory for the 3D matrix
  Matrix3D *new_matrix = malloc(sizeof(Matrix3D) + n_slices * n_rows * n_columns * sizeof(DTYPE));
  if (new_matrix == NULL) {
    fprintf(stderr, "Unable to allocate memory for (%u, %u, %u) array.", n_slices, n_rows, n_columns);
  } 
  // Store the size of the array
  new_matrix->nslices = n_slices;
  new_matrix->nrows = n_rows;
  new_matrix->ncolumns = n_columns;
  return new_matrix;
}


// Take a 3D volume of tomography data and isolate the pore structure using morphology filters
char tp_extract_pores(hid_t volume_ds, hid_t pores_ds) {
  // Create a kernel for the black tophat filters
  Matrix3D *kernelmax = tp_matrixmalloc(PORE_MAX_SIZE, PORE_MAX_SIZE, PORE_MAX_SIZE);
  tp_ellipsoid(kernelmax);
  Matrix3D *kernelmin = tp_matrixmalloc(PORE_MIN_SIZE, PORE_MIN_SIZE, PORE_MIN_SIZE);
  tp_box(kernelmin);
  // Retrieve metadata about the dataset
  hid_t volume_dsp = H5Dget_space(volume_ds);
  int ndims = H5Sget_simple_extent_ndims(volume_dsp);
  if (ndims != 3) {
    fprintf(stderr, "Error: Input dataset has %u dimensions instead of 3.\n", ndims);
  }
  hsize_t shape[3];
  H5Sget_simple_extent_dims(volume_dsp, shape, NULL);
  DIM n_slcs = PORE_MAX_SIZE;
  hsize_t n_rows = shape[1];
  hsize_t n_cols = shape[2];
  // Create an array to hold the data as it's loaded from disk
  float volume_buffer[n_slcs][n_rows][n_cols];
  float pore_slice_buffer[n_rows][n_cols];
  /* // Load the first kernel-height slices worth of data from disk to memory */
  /* hid_t dtype = H5Dget_type(volume); */
  /* herr_t error = H5Dread(volume,        // dataset_id */
  /* 			 dtype,         // mem_type_id */
  /* 			 H5S_ALL,       // mem_space_id */
  /* 			 H5S_ALL,       // file_space_id */
  /* 			 H5P_DEFAULT,   // xfer_plist_id */
  /* 			 volume_buffer  // buffer to hold the data */
  /* 			 ); */
  /* if (error < 0) { */
  /*   fprintf(stderr, "Failed to read dataset: %d\n", error); */
  /* }   */
  /* // Loop through the slices and apply the 3D filter */
  /* for (uint16_t islc=0; islc < n_slcs; islc++) { */
  /*   // Update the in-memory buffer with a new slice if necessary */
    
  /*   // Step through each pixel in the row and apply the kernel */
  /*   for (DIM irow=0; irow < n_rows; irow++) { */
  /*     for (DIM icol=0; icol < n_cols; icol++) { */
  /* 	pore_slice_buffer[irow][icol] = tp_apply_kernel(volume_buffer, kernelmin, islc, irow, icol); */
  /*     } */
  /*   } */
  /*   // Write the slice to the HDF5 dataset */
  /* } */
  // Free up memory and return
  free(kernelmax);
  return 0;
}


int main(int argc, char *argv[]) {
  // Open the HDF5 file
  hid_t h5fp = H5Fopen("data/phantom3d.h5", // File name to be opened
		       H5F_ACC_RDWR,        // file access mode
		       H5P_DEFAULT          // file access properties list
		       );
  // Open the source datasets
  hid_t src_ds, dst_ds;
  src_ds = H5Dopen2(h5fp, DS_SRC_NAME, H5P_DEFAULT);
  if (src_ds < 0) {
    if (!H5Lexists(h5fp, DS_SRC_NAME, H5P_DEFAULT)) {
      fprintf(stderr, "Source dataset '%s' not found.\n", DS_SRC_NAME);
    } else {
      fprintf(stderr, "Error opening source dataset '%s': %ld\n", DS_SRC_NAME, src_ds);
    }
    return -1;
  }
  // Open (or create) the destination dataset
  if (H5Lexists(h5fp, DS_DST_NAME, H5P_DEFAULT)) {
    dst_ds = H5Dopen2(h5fp, DS_DST_NAME, H5P_DEFAULT);
  } else {
    // Create the destination dataset if it didn't exist
    hid_t src_space = H5Dget_space(src_ds);
    dst_ds = H5Dcreate(h5fp,             // loc_id
		       DS_DST_NAME,      // name
		       H5T_NATIVE_FLOAT, // Datatype identifier
		       src_space,        // Dataspace identifier
		       H5P_DEFAULT,      // Link property list
		       H5P_DEFAULT,      // Creation property list
		       H5P_DEFAULT       // access property list
		       );
  }
  // Apply the morpohology filters to extract the pore structure
  char result = tp_extract_pores(src_ds, dst_ds);
  if (result < 0) {
    fprintf(stderr, "Failed to extract pores %s: %d\n", DS_SRC_NAME, result);
    return -1;
  }
  // TODO: Close the source and destination datasets
  return 0;
}


// Normalize the kernel so that the sum of all pixels equal 1
void tp_normalize_kernel(Matrix3D *kernel) {
  // Find out what unnormalized sum total is
  double total;
  for (uint16_t k=0; k < kernel->nslices; k++) {
    for (uint16_t j=0; j < kernel->nrows; j++) {
      for (uint16_t i=0; i < kernel->ncolumns; i++) {
	/* total += kernel[k][j][i]; */
      }
    }
  }
  // Divide every entry in the array by the sum total
  for (uint16_t k=0; k < kernel->nslices; k++) {
    for (uint16_t j=0; j < kernel->nrows; j++) {
      for (uint16_t i=0; i < kernel->ncolumns; i++) {
	/* kernel[k][j][i] = kernel[k][j][i] / total; */
      }
    }
  }
}


// Determine how many iterations exist in an array along the given dimension
// Eg. ``tp_num_iters(arr, 1)`` on a (16, 32, 64) array will return 32
/* uint16_t tp_num_iters(float *arr, uint16_t dimension) { */
/*   while (dimension > 0) { */
/*     arr = &arr[0]; */
/*   } */
/*   return sizeof(arr) / sizeof(arr[0]); */
/* } */


// Take a kernel of given dimensions and make it an ellipsoid of 1's surrounded by zeroes
void tp_ellipsoid(Matrix3D *kernel) {
  // Determine the center and radius along each axis
  Vector r_vec;
  uint64_t r_total;
  Vector center = {.z = (float) (kernel->nslices - 1) / 2.,
		   .y = (float) (kernel->nrows - 1) / 2.,
		   .x = (float) (kernel->ncolumns - 1) / 2.};
  Vector R_max = {.z = center.z*center.z, .y = center.y*center.y, .x = center.x * center.x};
  // Iterate over the array and set the value depending on if it's in the ellipsoid
  uint16_t i, j, k;
  for (i=0; i < kernel->nslices; i++) {
    for (j=0; j < kernel->nrows; j++) {
      for (k=0; k < kernel->ncolumns; k++) {
	// Calculate how far this point is from the center
	r_vec.z = center.z - (float) i;
	r_vec.y = center.y - (float) j;
	r_vec.x = center.x - (float) k;
	float r_total = (r_vec.x * r_vec.x / R_max.x) +
	  (r_vec.y * r_vec.y / R_max.y) + (r_vec.z * r_vec.z / R_max.z);
	if (r_total <= 1.) {
	  kernel->arr[tp_indices(kernel, i, j, k)] = 1.;
	} else {
	  kernel->arr[tp_indices(kernel, i, j, k)] = 0;
	}
      }
    }
  }
  // Normalize the kernel
  tp_normalize_kernel(kernel);
}


uint64_t tp_indices(Matrix3D *vol, DIM islice, DIM irow, DIM icolumn) {
  return islice * vol->nrows * vol->ncolumns + irow * vol->ncolumns + icolumn;
}


// Take a kernel of pre-determined dimensions and fill it with 1's
void tp_box(Matrix3D *kernel) {
  // Iterate through the kernel and make each element 1. for a box
  for (uint16_t i=0; i<kernel->nslices; i++) {
    for (uint16_t j=0; j<kernel->ncolumns; j++) {
      for (uint16_t k=0; k<kernel->nrows; k++) {
	kernel->arr[tp_indices(kernel, i, j, k)] = 1.;
      }
    }
  }
  // Normalize the kernel
  tp_normalize_kernel(kernel);
}


// Take a buffer of data, and apply the kernel to each row/column pixel for the given slice index
float tp_apply_kernel(float *arr, float *kernel, uint16_t islc, uint16_t irow, uint16_t icol) {
  
}
