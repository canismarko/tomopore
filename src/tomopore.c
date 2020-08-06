#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include "tomopore.h"

// Global constants
#define DS_SRC_NAME "volume"
#define DS_DST_NAME "pores"
#define DS_SIZE 262144
#define PORE_MIN_SIZE 5
#define PORE_MAX_SIZE 31
#define RANK 3



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


char tp_apply_filter
(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel,
 void (*filter_func)(DTYPE volume_val, DTYPE kernel_val, double *running_total, uint64_t *running_count)
 )
// Apply a given element-wise filter function to using the provided
// kernel to *src_ds* HDF5 dataset and save it to *dest_ds* HDF5
// dataset. *src_ds* and *dest_ds* can be the save dataset, in which
// case the operation is done in place. Return 0 on success, else -1
// on error.
{
  // Retrieve metadata about the dataset
  hid_t src_dsp = H5Dget_space(src_ds);
  int ndims = H5Sget_simple_extent_ndims(src_dsp);
  if (ndims != 3) {
    fprintf(stderr, "Error: Input dataset has %u dimensions instead of 3.\n", ndims);
    return -1;
  }
  hsize_t shape[3];
  H5Sget_simple_extent_dims(src_dsp, shape, NULL);
  DIM n_slcs = PORE_MIN_SIZE;
  hsize_t n_rows = shape[1];
  hsize_t n_cols = shape[2];
  // Create an array to hold the data as it's loaded from disk
  Matrix3D *working_buffer = tp_matrixmalloc(n_slcs, n_rows, n_cols);
  DTYPE pore_slice_buffer[n_rows][n_cols];
  // Prepare HDF5 dataspaces for reading and writing
  hsize_t strides[3] = {1, 1, 1};
  hsize_t counts[3] = {1, 1, 1};
  hid_t src_filespace_id = H5Dget_space(src_ds);
  hid_t pores_filespace_id = H5Dget_space(dest_ds);
  hsize_t write_blocks[3] = {1, n_rows, n_cols};
  hsize_t write_blocks_mem[2] = {n_rows, n_cols};
  hid_t src_memspace_id;
  hid_t pores_memspace_id = H5Screate_simple(RANK-1, write_blocks_mem, NULL);
  // Load the first kernel-height slices worth of data from disk to memory
  hsize_t read_starts[3] = {0, 0, 0}; // Gets update as new rows are read
  hsize_t write_starts[3] = {0, 0, 0}; // Gets update as new rows are written
  hsize_t read_blocks[3] = {n_slcs, n_rows, n_cols}; // Gets changed to {1, ...} after first read
  hsize_t read_extent[3] = {n_slcs, n_rows, n_cols};
  src_memspace_id = H5Screate_simple(RANK, read_blocks, NULL);
  H5Sselect_hyperslab(src_filespace_id,   // space_id,
		      H5S_SELECT_SET, // op,
		      read_starts,         // start
		      strides,        // stride
		      counts,         // count
		      read_blocks     // block
		      );
  herr_t error = H5Dread(src_ds,         // dataset_id
			 H5T_NATIVE_FLOAT,  // mem_type_id
			 src_memspace_id,  // mem_space_id
			 src_filespace_id,      // file_space_id
			 H5P_DEFAULT,       // xfer_plist_id
			 working_buffer->arr // buffer to hold the data
			 );
  if (error < 0) {
    fprintf(stderr, "Failed to read dataset: %d\n", error);
    return -1;
  }
  // Update the memory space so we can read one slice at a time
  read_blocks[0] = 1; // So we only get one slice at a time going forward
  hsize_t read_mem_starts[3] = {kernel->nslices - 1, 0, 0}; // Gets update as new rows are read
  src_memspace_id = H5Screate_simple(RANK, read_extent, NULL);
  H5Sselect_hyperslab(src_memspace_id, // space_id,
		      H5S_SELECT_SET,  // op,
		      read_mem_starts,     // start
		      strides,         // stride
		      counts,          // count
		      read_blocks      // block
		      );
  // Loop through the slices and apply the 3D filter
  uint8_t in_head = 0;
  uint8_t in_tail = 0;
  uint8_t update_needed = 0;
  DIM dL = (kernel->nslices - 1) / 2;
  DIM new_buffslc;
  DIM new_diskslc;
  DIM new_islc;
  hsize_t read_file_starts[3] = {0, 0, 0}; // Gets update as new rows are read
  for (DIM islc=0; islc < shape[0]; islc++) {
    // Update the in-memory buffer with a new slice if necessary
    in_head = (islc <= dL);
    in_tail = (islc >= (shape[0] - dL));
    update_needed = (!in_head && !in_tail);
    if (update_needed) {
      // Drop off the old slice and put the next one on top
      new_diskslc = islc + dL;
      new_islc = dL;
      printf("islc: %02d, diskslc: %02d, buffslc: %02lld, new_islc: %02d\n", islc, new_diskslc, read_mem_starts[0], new_islc);
      // Move each slice down
      for (DIM i=0; i<(kernel->nslices-1); i++) {
	for (DIM j=0; j<working_buffer->nrows; j++) {
	  for (DIM k=0; k<working_buffer->ncolumns; k++) {
	    working_buffer->arr[i,j,k] = working_buffer->arr[i+1,j,k];
	  }
	}
      }
      // Get a new last slice
      read_file_starts[0] = new_diskslc;
      H5Sselect_hyperslab(src_filespace_id,   // space_id,
			  H5S_SELECT_SET, // op,
			  read_file_starts,         // start
			  strides,        // stride
			  counts,         // count
			  read_blocks     // block
			  );
      herr_t error = H5Dread(src_ds,         // dataset_id
			     H5T_NATIVE_FLOAT,  // mem_type_id
			     src_memspace_id,  // mem_space_id
			     src_filespace_id,      // file_space_id
			     H5P_DEFAULT,       // xfer_plist_id
			     working_buffer->arr // buffer to hold the data
			     );
      if (error < 0) {
	fprintf(stderr, "Failed to read dataset: %d\n", error);
	return -1;
      }
    } else if (in_head) {
      new_islc = islc;
      printf("islc: --, diskslc: --, buffslc: --, new_islc: %02d\n", new_islc);
    } else if (in_tail) {
      new_islc = islc - shape[0] + kernel->nrows;
      printf("islc: --, diskslc: --, buffslc: --, new_islc: %02d\n", new_islc);
    }
    // Step through each pixel in the row and apply the kernel */
    for (DIM irow=0; irow < n_rows; irow++) {
      for (DIM icol=0; icol < n_cols; icol++) {
	pore_slice_buffer[irow][icol] = tp_apply_kernel(working_buffer,
							kernel,
							new_islc,
							irow,
							icol,
							filter_func);
      }
    }
    // Write the slice to the HDF5 dataset
    write_starts[0] = islc;
    H5Sselect_hyperslab(pores_filespace_id, // space_id,
			H5S_SELECT_SET,     // op,
			write_starts,       // start
			strides,            // stride
			counts,             // count
			write_blocks        // block
			);
    herr_t error = H5Dwrite(dest_ds,           // dataset_id
			    H5T_NATIVE_FLOAT,   // mem_type_id
			    pores_memspace_id,  // mem_space_id
			    pores_filespace_id, // file_space_id
			    H5P_DEFAULT,        // xfer_plist_id
			    pore_slice_buffer   // *buf
			    );
    if (error < 0) {
      fprintf(stderr, "Failed to write dataset: %d\n", error);
      return -1;
    }
  }
  
  return 0;
}


// Take a 3D volume of tomography data and isolate the pore structure using morphology filters
char tp_extract_pores(hid_t volume_ds, hid_t pores_ds) {
  // Create a kernel for the black tophat filters
  Matrix3D *kernelmax = tp_matrixmalloc(PORE_MAX_SIZE, PORE_MAX_SIZE, PORE_MAX_SIZE);
  tp_ellipsoid(kernelmax);
  Matrix3D *kernelmin = tp_matrixmalloc(PORE_MIN_SIZE, PORE_MIN_SIZE, PORE_MIN_SIZE);
  tp_box(kernelmin);

  char result = tp_apply_filter(volume_ds, pores_ds, kernelmin, tp_apply_max);
  if (result < 0) {
    fprintf(stderr, "Error: Could not apply max filter.\n");
  }
  // Free up memory and return
  free(kernelmax);
  free(kernelmin);
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
  src_ds = H5Dopen(h5fp, DS_SRC_NAME, H5P_DEFAULT);
  if (src_ds < 0) {
    if (!H5Lexists(h5fp, DS_SRC_NAME, H5P_DEFAULT)) {
      fprintf(stderr, "Source dataset '%s' not found.\n", DS_SRC_NAME);
    } else {
      fprintf(stderr, "Error opening source dataset '%s': %ld\n", DS_SRC_NAME, src_ds);
    }
    return -1;
  }
  // Open (or create) the destination dataset
  hid_t src_space = 0;
  if (H5Lexists(h5fp, DS_DST_NAME, H5P_DEFAULT)) {
    dst_ds = H5Dopen(h5fp, DS_DST_NAME, H5P_DEFAULT);
  } else {
    // Create the destination dataset if it didn't exist
    src_space = H5Dget_space(src_ds);
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
  // Close all the datasets, dataspaces, etc
  if (src_space > 0) {
    H5Sclose(src_space);
  }
  H5Dclose(src_ds);
  H5Dclose(dst_ds);
  H5Fclose(h5fp);
  // Check if the pore structure extraction finished successfully
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
  double total = 0;
  for (uint16_t k=0; k < kernel->nslices; k++) {
    for (uint16_t j=0; j < kernel->nrows; j++) {
      for (uint16_t i=0; i < kernel->ncolumns; i++) {
	total += kernel->arr[tp_indices(kernel, k, j, i)];
      }
    }
  }
  // Divide every entry in the array by the sum total
  for (uint16_t k=0; k < kernel->nslices; k++) {
    for (uint16_t j=0; j < kernel->nrows; j++) {
      for (uint16_t i=0; i < kernel->ncolumns; i++) {
	uint64_t idx = tp_indices(kernel, k, j, i);
	kernel->arr[idx] = kernel->arr[idx] / total;
      }
    }
  }
  total = 0;
  for (uint16_t k=0; k < kernel->nslices; k++) {
    for (uint16_t j=0; j < kernel->nrows; j++) {
      for (uint16_t i=0; i < kernel->ncolumns; i++) {
	total += kernel->arr[tp_indices(kernel, k, j, i)];
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


// Take a volume and return a flattened index for a given slice, row and column
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


/* Take a buffer of data, and apply the kernel to each row/column pixel for the given slice index
     *islc*, *irow*, *icol* give the current position in the *subvolume* buffer */
float tp_apply_kernel(Matrix3D *subvolume, Matrix3D *kernel, DIM islc, DIM irow, DIM icol,
		      void (*filter_func)(DTYPE volume_val, DTYPE kernel_val, double *running_total, uint64_t *running_count))
// i, j, k -> indices of subvolume
// l, m, n -> indices of kernel
// dL, dM, dN -> reach of the kernel, so for a 3x3x3 kernel, each is (3-1)/2 = 1
// Calculate some values to relate between the kernel and the subvolume
{
  DIM dL = (kernel->nslices - 1) / 2;
  DIM dM = (kernel->nrows - 1) / 2;
  DIM dN = (kernel->ncolumns - 1) / 2;
  DIM i, j, k;
  DTYPE kernel_val, volume_val;
  int is_in_bounds;
  double running_total = 0;
  uint64_t running_count = 0;
  // Iterate over the kernel dimensions, then apply them to the main arr
  for (DIM l=0; l < kernel->nslices; l++) {
    for (DIM m=0; m < kernel->nrows; m++) {
      for (DIM n=0; n < kernel->ncolumns; n++) {
	// Calculate relative coordinates in the arr matrix
	i = islc + (l - dL);
	j = irow + (m - dM);
	k = icol + (n - dN);
	// Determine if the new coordinates are in bounds for the subvolume
	is_in_bounds = ((i >= 0) && (i < subvolume->nslices) &&
			(j >= 0) && (j < subvolume->nrows) &&
			(k >= 0) && (k < subvolume->ncolumns));
	// Retrieve the values from arrays and perform the actual
	// operation
	if (is_in_bounds) {
	  volume_val = subvolume->arr[tp_indices(subvolume, i, j, k)];
	  kernel_val = kernel->arr[tp_indices(kernel, l, m, n)];
	  // TODO: Do a thing
	  (*filter_func)(volume_val, kernel_val, &running_total, &running_count);
	}
      }
    }
  }
  return (float) running_total;
}


void tp_apply_max(DTYPE volume_val, DTYPE kernel_val, double *running_total, uint64_t *running_count)
{
  if (kernel_val > 0) {
    // Save this value as the new maximum if it's bigger than the old one
    if (volume_val > *running_total) {
      *running_total = volume_val;
    }
    // Increment the counter, even though it doesn't really matter for calculating maxima
    *running_count++;
  }
}
