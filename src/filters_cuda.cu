#include <cuda_profiler_api.h>

#include "config.h"
#include "filters.h"
#include "filters_cuda.h"


// Local (cuda-only) function declarations
__global__ void tp_apply_kernel_cuda_kernel
(Matrix3D *working_buffer, Matrix3D *kernel, Matrix2D *pore_slice_buffer,
 enum operation op, DIM row_start, DIM row_end, DIM islc);
__device__ float tp_apply_kernel(Matrix3D *subvolume, Matrix3D *kernel, DIM islc, DIM irow, DIM icol, enum operation op);


// Function definitions
/* Take a buffer of data, and apply the kernel to each row/column pixel for the given slice index
 *islc*, *irow*, *icol* give the current position in the *subvolume* buffer */
__device__ float tp_apply_kernel(Matrix3D *subvolume, Matrix3D *kernel, DIM islc, DIM irow, DIM icol, enum operation op)
// i, j, k -> indices of subvolume
// l, m, n -> indices of kernel
// dL, dM, dN -> reach of the kernel, so for a 3x3x3 kernel, each is (3-1)/2 = 1
// Calculate some values to relate between the kernel and the subvolume
{
  DIM dL = (kernel->nslices - 1) / 2;
  DIM dM = (kernel->nrows - 1) / 2;
  DIM dN = (kernel->ncolumns - 1) / 2;
  int64_t i, j, k;
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


__global__ void tp_apply_kernel_cuda_kernel
(Matrix3D *working_buffer, Matrix3D *kernel, Matrix2D *pore_slice_buffer,
 enum operation op, DIM islc)
{
  DIM n_rows = working_buffer->nrows;
  DIM n_cols = working_buffer->ncolumns;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (DIM irow=index; irow < n_rows; irow+=stride) {
    for (DIM icol=0; icol < n_cols; icol++) {
      DDIM this_idx = irow * pore_slice_buffer->ncolumns + icol;
      pore_slice_buffer->arr[this_idx] = tp_apply_kernel(working_buffer,
							 kernel,
							 islc,
							 irow,
							 icol,
							 op);
    }
  }
    // DIM rows_per_thread = (DIM) ceil((double) n_rows / (double) config.n_threads);
    // DIM next_row = 0;
    // pthread_t tids[config.n_threads];
    // ThreadPayload *payload;
    // for (DIM tidx=0; tidx < config.n_threads; tidx++) {
    //   if (next_row < n_rows) {
    // 	payload = (ThreadPayload *) malloc(sizeof(ThreadPayload));
    // 	payload->row_start = next_row;
    // 	next_row += rows_per_thread;
    // 	payload->row_end = min_d(next_row, n_rows);
    // 	payload->n_cols = n_cols;
    // 	payload->new_islc = new_islc;
    // 	pthread_create(&tids[tidx], NULL, tp_apply_kernel_thread, payload);
    //   } else {
    // 	tids[tidx] = 0;
    //   }
    // }
    // // Wait for threads to finish
    // for (pthread_t tidx=0; tidx < config.n_threads; tidx++) {
    //   if (tids[tidx] > 0) {
    // 	pthread_join(tids[tidx], NULL);
    //   }
    // }
}



char tp_apply_kernel_cuda
(Matrix3D *working_buffer, Matrix3D *kernel, Matrix2D *pore_slice_buffer, enum operation op, DIM islc)
{
  // printf("Before: %f\n", working_buffer->arr[0]);
  cudaProfilerStart();
  tp_apply_kernel_cuda_kernel<<<1, 512>>>(working_buffer, kernel, pore_slice_buffer, op, islc);
  cudaDeviceSynchronize();
  cudaProfilerStop();
  // printf("After: %f\n", working_buffer->arr[0]);
  return 0;
}
