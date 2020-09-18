#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <hdf5.h>
#include <pthread.h>

#include "filters.h"
#include "hdfhelpers.h"
// #include "tomopore.h"
#define EXTERN
#include "config.h"

#define RANK 3

/* Take a buffer of data, and apply the kernel to each row/column pixel for the given slice index
     *islc*, *irow*, *icol* give the current position in the *subvolume* buffer */
float tp_apply_kernel(Matrix3D *subvolume, Matrix3D *kernel, DIM islc, DIM irow, DIM icol, enum operation op)
// i, j, k -> indices of subvolume
// l, m, n -> indices of kernel
// dL, dM, dN -> reach of the kernel, so for a 3x3x3 kernel, each is (3-1)/2 = 1
// Calculate some values to relate between the kernel and the subvolume
{
  DIM dL = (kernel->nslices - 1) / 2;
  DIM dM = (kernel->nrows - 1) / 2;
  DIM dN = (kernel->ncolumns - 1) / 2;
  DIM i, j, k;
  DDIM subvolume_slice_idx = 0;
  DDIM subvolume_idx = 0;
  DDIM kernel_idx = 0;
  DTYPE kernel_val, volume_val;
  int is_in_bounds;
  char is_first_round = TRUE;
  double running_total = 0;
  char replace_value = FALSE;
  // Iterate over the kernel dimensions, then apply them to the main arr
  // i, j, k are in the buffer space
  // l, m, n are in the kernel space
  for (DIM l=0; l < kernel->nslices; l++) {
    i = islc + (l - dL);
    // Check if this is a valid slice in the subvolume
    is_in_bounds = (i >= 0) && (i < subvolume->nslices);
    if (!is_in_bounds) continue;
    // Keep track of where we are in the subvolume
    subvolume_slice_idx = i * subvolume->nrows * subvolume->ncolumns;
    for (DIM m=0; m < kernel->nrows; m++) {
      j = irow + (m - dM);
      // Check if this is a valid row in the subvolume
      is_in_bounds = (j >= 0) && (j < subvolume->nrows);
      if (!is_in_bounds) continue;
      // Keep track of where we are in the subvolume
      subvolume_idx = subvolume_slice_idx + j * subvolume->ncolumns;
      for (DIM n=0; n < kernel->ncolumns; n++) {
	// Calculate relative coordinates in the arr matrix
	k = icol + (n - dN);
	// Check if this is a valid column in the subvolume
	is_in_bounds = (k >= 0) && (k < subvolume->ncolumns);
	if (!is_in_bounds) continue;
	// Retrieve the values from arrays and perform the actual
	// operation
	if (is_in_bounds) {
	  volume_val = subvolume->arr[subvolume_idx + k];
	  kernel_val = kernel->arr[kernel_idx];
	  // Apply the actual kernel filter function
	  if (kernel_val > 0) {
	    // Set the beginning value if one hasn't been set yet
	    if (is_first_round) {
	      running_total = volume_val;
	      is_first_round = FALSE;
	    }
	    // Save this value as the new minimum/maximum if it's bigger than the old one
	    replace_value = (volume_val < running_total) && (op == Min);
	    replace_value = replace_value || ((volume_val > running_total) && (op == Max));
	    if (replace_value) {
	      running_total = volume_val;
	    }
	  }	  
	}
	kernel_idx++;
      }
    }
  }
  return (float) running_total;
}


char tp_apply_filter
(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel, enum operation op)
// Apply a given element-wise filter function to using the provided
// kernel to *src_ds* HDF5 dataset and save it to *dest_ds* HDF5
// dataset. *src_ds* and *dest_ds* can be the same dataset, in which
// case the operation is done in place. Return 0 on success, else -1
// on error.
{
  // Retrieve metadata about the dataset
  hid_t src_filespace_id = H5Dget_space(src_ds);
  int ndims = H5Sget_simple_extent_ndims(src_filespace_id);
  if (ndims != RANK) {
    fprintf(stderr, "Error: Input dataset has %u dimensions instead of 3.\n", ndims);
    return -1;
  }
  hsize_t shape[3];
  H5Sget_simple_extent_dims(src_filespace_id, shape, NULL);
  DIM n_slcs = kernel->nslices;
  hsize_t n_rows = shape[1];
  hsize_t n_cols = shape[2];
  // Create an array to hold the data as it's loaded from disk
  Matrix3D *working_buffer = tp_matrixmalloc(n_slcs, n_rows, n_cols);
  // DTYPE pore_slice_buffer[n_rows][n_cols];
  Matrix2D *pore_slice_buffer = tp_matrixmalloc2d(n_rows, n_cols);
  hid_t dest_filespace_id = H5Dget_space(dest_ds);
  // Prepare HDF5 dataspaces for reading and writing
  hsize_t strides[3] = {1, 1, 1};
  hsize_t counts[3] = {1, 1, 1};
  hsize_t write_blocks[3] = {1, n_rows, n_cols};
  hsize_t write_blocks_mem[2] = {n_rows, n_cols};
  hid_t dest_memspace_id = H5Screate_simple(RANK-1, write_blocks_mem, NULL);
  // Load the first kernel-height slices worth of data from disk to memory
  hsize_t read_starts[3] = {0, 0, 0}; // Gets update as new rows are read
  hsize_t write_starts[3] = {0, 0, 0}; // Gets update as new rows are written
  hsize_t read_blocks[3] = {n_slcs, n_rows, n_cols}; // Gets changed to {1, ...} after first read
  hsize_t read_extent[3] = {n_slcs, n_rows, n_cols};
  hid_t src_memspace_id = H5Screate_simple(RANK, read_blocks, NULL);
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
		      read_mem_starts, // start
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
      // Move each slice down
      roll_buffer(working_buffer);
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
    } else if (in_tail) {
      new_islc = islc - shape[0] + kernel->nrows;
    }
    // Step through each pixel in the row and apply the kernel */
    DIM rows_per_thread = (DIM) ceil((double) n_rows / (double) config.n_threads);
    DIM next_row = 0;
    pthread_t tids[config.n_threads];
    ThreadPayload *payload;
    for (DIM tidx=0; tidx < config.n_threads; tidx++) {
      if (next_row < n_rows) {
	payload = (ThreadPayload *) malloc(sizeof(ThreadPayload));
	payload->row_start = next_row;
	next_row += rows_per_thread;
	payload->row_end = min_d(next_row, n_rows);
	payload->n_cols = n_cols;
	payload->new_islc = new_islc;
	payload->pore_slice_buffer = pore_slice_buffer;
	payload->working_buffer = working_buffer;
	payload->kernel = kernel;
	payload->op = op;
	pthread_create(&tids[tidx], NULL, tp_apply_kernel_thread, payload);
      } else {
	tids[tidx] = 0;
      }
    }
    // Wait for threads to finish
    for (pthread_t tidx=0; tidx < config.n_threads; tidx++) {
      if (tids[tidx] > 0) {
	pthread_join(tids[tidx], NULL);
      }
    }
    // Write the slice to the HDF5 dataset
    write_starts[0] = islc;
    H5Sselect_hyperslab(dest_filespace_id, // space_id,
			H5S_SELECT_SET,    // op,
			write_starts,      // start
			strides,           // stride
			write_blocks,      // count
			counts             // block
			);
    herr_t error = H5Dwrite(dest_ds,               // dataset_id
			    H5T_NATIVE_FLOAT,      // mem_type_id
			    dest_memspace_id,      // mem_space_id
			    dest_filespace_id,     // file_space_id
			    H5P_DEFAULT,           // xfer_plist_id
			    pore_slice_buffer->arr // *buf
			    );
    if (error < 0) {
      fprintf(stderr, "Failed to write dataset: %d\n", error);
      return -1;
    }
    print_progress(islc, shape[0]-1, "Filtering slices");
  }
  
  return 0;
}


void print_progress(DIM current, DIM total, const char* desc)
{
  if (config.quiet)
    return;
  float ratio = ((float) current / (float) total);
  // Prepare a progress bar
  char bar_length = 30;
  char bar[bar_length+1];
  bar[bar_length] = '\0';
  float bar_ratio;
  for (char i=0; i<bar_length; i++) {
    bar_ratio = ((float) i / (float) bar_length);
    if (bar_ratio < ratio) {
      bar[i] = '#';
    } else {
      bar[i] = ' ';
    }
  }
  // Do the printing
  printf("\r%s: |%s| %lu/%lu (%.1f%%)", desc, bar, current, total, ratio * 100.);
  if (current == total) {
    // We're finished, so print a newline
    printf("\n");
  } else {
    fflush(stdout);
  }
}


char tp_subtract_datasets(hid_t src_ds1, hid_t src_ds2, hid_t dest_ds)
// Perform element-wise subtraction on one dataset from another, and
// save the result.
// 
// For each element *i*, dest_ds[i] = src_ds1[i] - src_ds2
// 
// *dest_ds* may be the same as either *src_ds1* or *src_ds2*, in
// which case the opeartion is done in place.
{
  // Retrieve metadata about the dataset
  hid_t src_filespace1 = H5Dget_space(src_ds1);
  hid_t src_filespace2 = H5Dget_space(src_ds2);
  hid_t dest_filespace = H5Dget_space(dest_ds);
  int ndims = H5Sget_simple_extent_ndims(src_filespace1);
  // Check if the sources and destination are equally sized
  if (!H5Sextent_equal(src_filespace1, src_filespace2)) {
    printf("Could not subtract datasets, source extents are not equal.\n");
    return -1;
  }
  if (!H5Sextent_equal(src_filespace1, dest_filespace)) {
    printf("Could not subtract datasets, source and destination extents are not equal.\n");
    return -1;
  }
  if (ndims != RANK) {
    fprintf(stderr, "Error: Input dataset has %u dimensions instead of 3.\n", ndims);
    return -1;
  }
  hsize_t shape[3];
  H5Sget_simple_extent_dims(src_filespace1, shape, NULL);
  hsize_t n_slcs = shape[0];
  hsize_t n_rows = shape[1];
  hsize_t n_cols = shape[2];
  // Create arrays to hold the data as it's loaded from disk
  Matrix2D *input_buffer1 = tp_matrixmalloc2d(n_rows, n_cols);
  Matrix2D *input_buffer2 = tp_matrixmalloc2d(n_rows, n_cols);
  Matrix2D *output_buffer = tp_matrixmalloc2d(n_rows, n_cols);
  // Prepare HDF5 dataspaces for reading and writing
  hsize_t strides[3] = {1, 1, 1};
  hsize_t counts[3] = {1, 1, 1};
  hsize_t blocks_mem[2] = {n_rows, n_cols};
  hsize_t blocks_file[3] = {1, n_rows, n_cols};
  hsize_t starts[3] = {0, 0, 0}; // Gets update as new rows are read
  hid_t memspace = H5Screate_simple(RANK-1, blocks_mem, NULL);
  // Iterate through each slice and do the subtraction
  for (DIM islc=0; islc<n_slcs; islc++) {
    starts[0] = islc;
    // Load the data from disk into the buffers
    H5Sselect_hyperslab(src_filespace1, // space_id,
			H5S_SELECT_SET, // op,
			starts,         // start
			strides,        // stride
			counts,         // count
			blocks_file      // block
			);
    herr_t error;
    error = H5Dread(src_ds1,           // dataset_id
		    H5T_NATIVE_FLOAT,  // mem_type_id
		    memspace,          // mem_space_id
		    src_filespace1,    // file_space_id
		    H5P_DEFAULT,       // xfer_plist_id
		    input_buffer1->arr // buffer to hold the data
		    );
    if (error < 0) {
      printf("Read failed\n");
      return -1;
    }
    H5Sselect_hyperslab(src_filespace2, // space_id,
			H5S_SELECT_SET, // op,
			starts,         // start
			strides,        // stride
			counts,         // count
			blocks_file      // block
			);
    error = H5Dread(src_ds2,           // dataset_id
		    H5T_NATIVE_FLOAT,  // mem_type_id
		    memspace,          // mem_space_id
		    src_filespace2,    // file_space_id
		    H5P_DEFAULT,       // xfer_plist_id
		    input_buffer2->arr // buffer to hold the data
		    );
    if (error < 0) {
      printf("Read failed\n");
      return -1;
    }
    // Do the subtraction elemenet-wise
    for (DIM irow=0; irow<n_rows; irow++) {
      for (DIM icol=0; icol<n_cols; icol++) {
	DDIM this_i = tp_indices2d(output_buffer, irow, icol);
	output_buffer->arr[this_i] = input_buffer1->arr[this_i] - input_buffer2->arr[this_i];
      }
    }
    // Write this slice back to disk
    H5Sselect_hyperslab(dest_filespace, // space_id,
			H5S_SELECT_SET, // op,
			starts,         // start
			strides,        // stride
			counts,         // count
			blocks_file     // block
			);
    error = H5Dwrite(dest_ds,            // dataset_id
		     H5T_NATIVE_FLOAT,   // mem_type_id
		     memspace,           // mem_space_id
		     dest_filespace,     // file_space_id
		     H5P_DEFAULT,        // xfer_plist_id
		     output_buffer->arr  // *buf
		     );
    if (error < 0) {
      fprintf(stderr, "Failed to write dataset: %d\n", error);
      return -1;
    }
    if (config.verbose)
      print_progress(islc, shape[0]-1, "Subtracting");
  }
  return 0;
}


void *tp_apply_kernel_thread(void *args)
{
  // Unpack the payload
  ThreadPayload *payload = (ThreadPayload *) args;
  Matrix2D *pore_slice_buffer = payload->pore_slice_buffer;
  Matrix3D *working_buffer = payload->working_buffer;
  Matrix3D *kernel = payload->kernel;
  // Now process the row
  DDIM this_idx;
  for (DIM irow=payload->row_start; irow < payload->row_end; irow++) {
    for (DIM icol=0; icol < payload->n_cols; icol++) {
      this_idx = tp_indices2d(pore_slice_buffer, irow, icol);
      pore_slice_buffer->arr[this_idx] = tp_apply_kernel(working_buffer,
							 kernel,
							 payload->new_islc,
							 irow,
							 icol,
							 payload->op);
    }
  }
  return 0;
}


// Take a 3D volume of tomography data and isolate the pore structure using morphology filters
char tp_extract_pores(hid_t volume_ds, hid_t pores_ds, hid_t h5fp, char *name, DIM min_pore_size, DIM max_pore_size) {
  if (max_pore_size <= 1) {
    if (!config.quiet)
      printf("Skipping pore extraction since max_pore_size <= 1\n");
    return 0;
  } else {
    if (!config.quiet)
    printf("\nPore extraction (4 passes)\n"
	   "==========================\n");
  }

  // Create a kernel for the black tophat filters
  Matrix3D *kernelmax = tp_matrixmalloc(max_pore_size, max_pore_size, max_pore_size);
  tp_ellipsoid(kernelmax);
  Matrix3D *kernelmin = tp_matrixmalloc(min_pore_size, min_pore_size, min_pore_size);
  tp_ellipsoid(kernelmin);

  // Prepare a temporary dataset to hold the intermediate data
  hid_t src_space = H5Dget_space(volume_ds);
  hid_t src_type = H5Dget_type(volume_ds);
  hid_t temporary_ds = tp_replace_dataset(strcat(strdup(name), "_tomopore_temp"), h5fp, src_space, src_type);

  // Apply small black tophat filter
  char result;
  if (min_pore_size > 0) {
    // A minimum pore size was requested, so do it in multiple steps
    result = tp_apply_black_tophat(volume_ds, temporary_ds, kernelmin);
  
    // Apply large black tophat filter
    result = tp_apply_black_tophat(volume_ds, pores_ds, kernelmax);
  
    // Subtract the two
    result = tp_subtract_datasets(pores_ds, temporary_ds, pores_ds);
  } else {
    // No minimum pore size, so single-step extraction
    result = tp_apply_black_tophat(volume_ds, pores_ds, kernelmax);
  }

  // Free up memory and return
  H5Dclose(temporary_ds);
  /* H5Dclose(temporary_ds2);  */
  free(kernelmax);
  free(kernelmin);
  return result;
}


// Take a 3D volume of tomography data and isolate the lead structure using morphology filters
char tp_extract_lead(hid_t volume_ds, hid_t lead_ds, hid_t h5fp, char *name, DIM min_lead_size, DIM max_lead_size) {
  if (max_lead_size <= 1) {
    if (!config.quiet)
      printf("Skipping free lead extraction since max_lead_size <= 1\n");
    return 0;
  } else {
    if (!config.quiet)
      printf("\nFree lead extraction (4 passes)\n"
	     "===============================\n");
  }
  // Create a kernel for the black tophat filters
  Matrix3D *kernelmax = tp_matrixmalloc(max_lead_size, max_lead_size, max_lead_size);
  tp_ellipsoid(kernelmax);
  Matrix3D *kernelmin = tp_matrixmalloc(min_lead_size, min_lead_size, min_lead_size);
  tp_ellipsoid(kernelmin);

  // Prepare a temporary dataset to hold the intermediate data
  hid_t src_space = H5Dget_space(volume_ds);
  hid_t src_type = H5Dget_type(volume_ds);
  hid_t temporary_ds = tp_replace_dataset(strcat(strdup(name), "_tomopore_temp"), h5fp, src_space, src_type);

  // Apply small black tophat filter
  char result;
  result = tp_apply_white_tophat(volume_ds, temporary_ds, kernelmin);
  
  // Apply large black tophat filter
  result = tp_apply_white_tophat(volume_ds, lead_ds, kernelmax);
  
  // Subtract the two
  result = tp_subtract_datasets(lead_ds, temporary_ds, lead_ds);

  // Free up memory and return
  H5Dclose(temporary_ds);
  free(kernelmax);
  free(kernelmin);
  return 0;

}


char tp_apply_erosion(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel)
{
  if (config.verbose)
    printf("Applying erosion filter: (%lu, %lu, %lu) kernel.\n", kernel->nslices, kernel->nrows, kernel->ncolumns);
  return tp_apply_filter(src_ds, dest_ds, kernel, Min);
}

char tp_apply_dilation(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel)
{
  if (config.verbose)
    printf("Applying dilation filter: (%lu, %lu, %lu) kernel.\n", kernel->nslices, kernel->nrows, kernel->ncolumns);
  return tp_apply_filter(src_ds, dest_ds, kernel, Max);
}

char tp_apply_opening(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel)
{
  // Morphological opening on an image is defined as an erosion
  // followed by a dilation.
  char result;
  // Erosion filter
  result = tp_apply_erosion(src_ds, dest_ds, kernel);
  if (result < 0) {
    fprintf(stderr, "Error: Could not apply min filter.\n");
  }
  // Dilation filter
  result = tp_apply_dilation(dest_ds, dest_ds, kernel);
  if (result < 0) {
    fprintf(stderr, "Error: Could not apply max filter.\n");
  }
  return 0;
}


char tp_apply_closing(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel)
{
  // Morphological closing on an image is defined as a dilation
  // followed by an erosion. Closing can remove small dark spots
  // (i.e. “pepper”) and connect small bright cracks.
  char result;
  // Dilation filter
  result = tp_apply_dilation(src_ds, dest_ds, kernel);
  if (result < 0) {
    fprintf(stderr, "Error: Could not apply max filter.\n");
  }
  // Erosion filter
  result = tp_apply_erosion(dest_ds, dest_ds, kernel);
  if (result < 0) {
    fprintf(stderr, "Error: Could not apply min filter.\n");
  }
  return result;
}


char tp_apply_white_tophat(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel)
{
  // The white_tophat of an image is defined as the image minus its
  // morphological opening. This operation returns the bright spots of
  // the image that are smaller than the structuring element.

  // Apply the opening filter
  char result;
  result = tp_apply_opening(src_ds, dest_ds, kernel);

  // Subtract the opening image from the original
  if (result == 0) {
    result = tp_subtract_datasets(src_ds, dest_ds, dest_ds);
  } else {
    fprintf(stderr, "Failed to apply morphological opening for white tophat: %d\n", result);
  }

  return result;
}
  

char tp_apply_black_tophat(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel)
{
  // The black_tophat of an image is defined as its morphological
  // closing minus the original image. This operation returns the dark
  // spots of the image that are smaller than the structuring element.

  // Apply the closing filter
  char result;
  result = tp_apply_closing(src_ds, dest_ds, kernel);

  // Subtract the original image from the closing
  if (result == 0) {
    result = tp_subtract_datasets(dest_ds, src_ds, dest_ds);
  }

  return result;
}


Matrix3D *tp_matrixmalloc(DIM n_slices, DIM n_rows, DIM n_columns) {
  // Allocated memory for the 3D matrix
  Matrix3D *new_matrix = (Matrix3D *) malloc(sizeof(Matrix3D) + n_slices * n_rows * n_columns * sizeof(DTYPE));
  if (new_matrix == NULL) {
    fprintf(stderr, "Unable to allocate memory for (%lu, %lu, %lu) array.", n_slices, n_rows, n_columns);
  } 
  // Store the size of the array
  new_matrix->nslices = n_slices;
  new_matrix->nrows = n_rows;
  new_matrix->ncolumns = n_columns;
  return new_matrix;
}


Matrix2D *tp_matrixmalloc2d(DIM n_rows, DIM n_columns) {
  // Allocated memory for the 2D matrix
  Matrix2D *new_matrix = (Matrix2D *) malloc(sizeof(Matrix2D) + n_rows * n_columns * sizeof(DTYPE));
  if (new_matrix == NULL) {
    fprintf(stderr, "Unable to allocate memory for (%lu, %lu) array.", n_rows, n_columns);
  } 
  // Store the size of the array
  new_matrix->nrows = n_rows;
  new_matrix->ncolumns = n_columns;
  return new_matrix;
}


void tp_normalize_kernel(Matrix3D *kernel)
// Normalize the kernel so that the sum of all pixels equal 1
{
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
	DDIM idx = tp_indices(kernel, k, j, i);
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


void tp_ellipsoid(Matrix3D *kernel)
// Take a kernel of given dimensions and make it an ellipsoid of 1's
// surrounded by zeroes
{
  // Determine the center and radius along each axis
  Vector r_vec;
  uint64_t r_total;
  Vector center = {
    .z = (kernel->nslices - 1) / 2.,
    .y = (kernel->nrows - 1) / 2.,
    .x = (kernel->ncolumns - 1) / 2.
  };
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


void tp_box(Matrix3D *kernel)
// Take a kernel of pre-determined dimensions and fill it with 1's
{
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


static void roll_buffer(Matrix3D *buffer)
// Take a 3D matrix and move everything down by 1 line (slice) along
// the first axis. The last line (slice) along the first axis is then
// a duplicate of the second to last line (slice).
{
  uint64_t old_idx, new_idx;
  for (DIM i=0; i < (buffer->nslices-1); i++) {
    for (DIM j=0; j < buffer->nrows; j++) {
      for (DIM k=0; k < buffer->ncolumns; k++) {
	old_idx = tp_indices(buffer, i, j, k);
	new_idx = tp_indices(buffer, i+1, j, k);
	buffer->arr[old_idx] = buffer->arr[new_idx];
      }
    }
  }
}


static DIM min_d(DIM x, DIM y) 
{
  return (x < y) ? x : y;
}


static DIM max_d(DIM x, DIM y) 
{
  return (x > y) ? x : y;
}
