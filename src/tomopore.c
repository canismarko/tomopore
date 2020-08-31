#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>
#include <pthread.h>
#include <argp.h>
#include <sys/sysinfo.h>
#include <time.h>

#include "tomopore.h"

// Global constants and default values
#define RANK 3
#define TRUE 1
#define FALSE 0
#define PORE_MIN_SIZE 5
#define PORE_MAX_SIZE 51
#define LEAD_MIN_SIZE 5
#define LEAD_MAX_SIZE 35
#define SOURCE_NAME "volume"
#define DEST_NAME_PORES "pores"
#define DEST_NAME_LEAD "lead"
const char *argp_program_version =
  "tomopore 0.1";
const char *argp_program_bug_address =
  "<wolfman@anl.gov>";

// Program documentation
static char doc[] =
  "Tomopore -- Memory efficient 3D extraction of pores from tomography data";

// A description of the arguments we accept
static char args_doc[] =
  "H5_FILE";

// Global variables
static struct {
  int n_threads, verbose, quiet;
} config;

#define OPT_MIN_PORE_SIZE 1
#define OPT_MAX_PORE_SIZE 2
#define OPT_MIN_LEAD_SIZE 3
#define OPT_MAX_LEAD_SIZE 4
#define OPT_DEST_PORES 5
#define OPT_DEST_LEAD 6
#define OPT_NO_PORES 7
#define OPT_NO_LEAD 8

static struct argp_option options[] = {
  {"threads", 'j', "NUM_THREADS", 0, "Number of parallel threads, defaults to using all cores", 0},
  {"verbose",  'v', 0,      0,  "Produce verbose output", 0},
  {"quiet",    'q', 0,      0,  "Don't produce any output", 0},
  
  {"source", 's', "DATASET", 0, "Path to the source dataset containing float volume data", 1},
  {"dest-pores", OPT_DEST_PORES, "DATASET", 0, "Path to the dataset that will receive segmented pores", 1},
  {"dest-lead", OPT_DEST_LEAD, "DATASET", 0, "Path to the dataset that will receive segmented free lead", 1},

  {"no-pores", OPT_NO_PORES, 0, 0, "Skip segmentation of pores", 2},
  {"no-lead", OPT_NO_LEAD, 0, 0, "Skip segmentation of free lead", 2},

  {"min-pore-size", OPT_MIN_PORE_SIZE, "SIZE", 0, "Minimum size of pores (in pixels)", 3},
  {"max-pore-size", OPT_MAX_PORE_SIZE, "SIZE", 0, "Maximum size of pores (in pixels)", 3},
  {"min-lead-size", OPT_MIN_LEAD_SIZE, "SIZE", 0, "Minimum size of lead (in pixels)", 3},
  {"max-lead-size", OPT_MAX_LEAD_SIZE, "SIZE", 0, "Maximum size of lead (in pixels)", 3},
  
  { 0 }
};

/* Used by main to communicate with parse_opt. */
struct arguments
{
  char *hdf_filename, *source, *dest_lead, *dest_pores;
  DIM min_pore_size, max_pore_size, min_lead_size, max_lead_size;
  int n_threads, quiet, verbose, no_lead, no_pores;
};

/* Parse a single option. */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
  struct arguments *arguments = state->input;

  switch (key)
    {
    case 'j':
      arguments->n_threads = atoi(arg);
      break;

    case 'q':
      arguments->quiet = TRUE;
      break;

    case 'v':
      arguments->verbose = TRUE;
      break;

    case OPT_NO_PORES:
      arguments->no_pores = TRUE;
      break;

    case OPT_NO_LEAD:
      arguments->no_lead = TRUE;
      break;      

    case OPT_MIN_PORE_SIZE:
      arguments->min_pore_size = atoi(arg);
      break;

    case OPT_MAX_PORE_SIZE:
      arguments->max_pore_size = atoi(arg);
      break;

    case OPT_MIN_LEAD_SIZE:
      arguments->min_lead_size = atoi(arg);
      break;

    case OPT_MAX_LEAD_SIZE:
      arguments->max_lead_size = atoi(arg);
      break;

    case 's':
      arguments->source = arg;
      break;

    case OPT_DEST_PORES:
      arguments->dest_pores = arg;
      break;

    case OPT_DEST_LEAD:
      arguments->dest_lead = arg;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 1)
        /* Too many arguments. */
        argp_usage (state);
      arguments->hdf_filename = arg;
      break;

    case ARGP_KEY_END:
      if (state->arg_num < 1)
        /* Not enough arguments. */
        argp_usage (state);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, doc };


/* Apply a series of filters to the volume using 3D kernels This is done */
/* with a 3D kernel so that we don't lose data, but that means lots more */
/* memory. To avoid running out of memory, intermediate arrays are saved */
/* in HDF5 datasets. */

hid_t tp_replace_dataset
(char *dataset_name, hid_t h5fp, hid_t dataspace)
// Replace an existing dataset with a new one specified in *dataspace*
{
  hid_t new_dataset_id;
  if (H5Lexists(h5fp, dataset_name, H5P_DEFAULT)) {
    // Unlink the old dataset
    if (config.verbose)
      printf("Removing existing dataset: %s\n", dataset_name);
    herr_t error = H5Ldelete(h5fp,         // loc_id
			     dataset_name, // *name
			     H5P_DEFAULT   // access property list
			     );
  }
  // Now create a new dataset
  new_dataset_id = tp_require_dataset(dataset_name, h5fp, dataspace);
  return new_dataset_id;
}

hid_t tp_require_dataset
(char *dataset_name, hid_t h5fp, hid_t dataspace)
// Open an existing dataset, or create a new one if one doesn't exist
{
  hid_t new_dataset_id;
  if (H5Lexists(h5fp, dataset_name, H5P_DEFAULT)) {
    // Compare extents to make sure they match
    new_dataset_id = H5Dopen(h5fp, dataset_name, H5P_DEFAULT);
  } else {
    // Create new dataset if it didn't already exist
    if (config.verbose)
      printf("Creating new dataset: %s\n", dataset_name);
    new_dataset_id = H5Dcreate(h5fp,             // loc_id
		       dataset_name,      // name
		       H5T_NATIVE_FLOAT, // Datatype identifier
		       dataspace,        // Dataspace identifier
		       H5P_DEFAULT,      // Link property list
		       H5P_DEFAULT,      // Creation property list
		       H5P_DEFAULT       // access property list
		       );
    if (new_dataset_id < 0) {
      fprintf(stderr, "Error: Failed to create new data '%s': %ld\n", dataset_name, new_dataset_id);
      return -1;
    }
  }
  return new_dataset_id;
}


void print_progress(DIM current, DIM total, char desc[])
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
  printf("\r%s: |%s| %d/%d (%.1f%%)", desc, bar, current, total, ratio * 100.);
  if (current == total) {
    // We're finished, so print a newline
    printf("\n");
  } else {
    fflush(stdout);
  }
}


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


Matrix2D *tp_matrixmalloc2d(DIM n_rows, DIM n_columns) {
  // Allocated memory for the 2D matrix
  Matrix2D *new_matrix = malloc(sizeof(Matrix2D) + n_rows * n_columns * sizeof(DTYPE));
  if (new_matrix == NULL) {
    fprintf(stderr, "Unable to allocate memory for (%u, %u) array.", n_rows, n_columns);
  } 
  // Store the size of the array
  new_matrix->nrows = n_rows;
  new_matrix->ncolumns = n_columns;
  return new_matrix;
}


void roll_buffer(Matrix3D *buffer)
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
    /* DIM this_idx; */
    /* for (DIM irow=0; irow < n_rows; irow++) { */
    /*   for (DIM icol=0; icol < n_cols; icol++) { */
    /* 	this_idx = tp_indices2d(pore_slice_buffer, irow, icol); */
    /* 	pore_slice_buffer->arr[this_idx] = tp_apply_kernel(working_buffer, */
    /* 							   kernel, */
    /* 							   new_islc, */
    /* 							   irow, */
    /* 							   icol, */
    /* 							   op); */
    /*   } */
    /* } */
    DIM rows_per_thread = (DIM) ceil((double) n_rows / (double) config.n_threads);
    DIM next_row = 0;
    pthread_t tids[config.n_threads];
    ThreadPayload *payload;
    for (DIM tidx=0; tidx < config.n_threads; tidx++) {
      /* for (DIM irow=0; irow < n_rows; irow++) { */
      if (next_row < n_rows) {
	payload = malloc(sizeof(ThreadPayload));
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
}


char tp_apply_erosion(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel)
{
  if (config.verbose)
    printf("Applying erosion filter: (%d, %d, %d) kernel.\n", kernel->nslices, kernel->nrows, kernel->ncolumns);
  return tp_apply_filter(src_ds, dest_ds, kernel, Min);
}

char tp_apply_dilation(hid_t src_ds, hid_t dest_ds, Matrix3D *kernel)
{
  if (config.verbose)
    printf("Applying dilation filter: (%d, %d, %d) kernel.\n", kernel->nslices, kernel->nrows, kernel->ncolumns);
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
  hid_t temporary_ds = tp_replace_dataset(strcat(strdup(name), "_tomopore_temp"), h5fp, src_space);

  // Apply small black tophat filter
  char result;
  result = tp_apply_black_tophat(volume_ds, temporary_ds, kernelmin);
  
  // Apply large black tophat filter
  result = tp_apply_black_tophat(volume_ds, pores_ds, kernelmax);
  
  // Subtract the two
  result = tp_subtract_datasets(pores_ds, temporary_ds, pores_ds);

  // Free up memory and return
  H5Dclose(temporary_ds);
  /* H5Dclose(temporary_ds2);  */
  free(kernelmax);
  free(kernelmin);
  return 0;
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
  hid_t temporary_ds = tp_replace_dataset(strcat(strdup(name), "_tomopore_temp"), h5fp, src_space);

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


int main(int argc, char *argv[]) {
  // Save start time to measure total execution
  time_t start_time = time(NULL);
  
  struct arguments arguments;
  /* Default option values. */
  arguments.min_pore_size = PORE_MIN_SIZE;
  arguments.max_pore_size = PORE_MAX_SIZE;
  arguments.min_lead_size = LEAD_MIN_SIZE;
  arguments.max_lead_size = LEAD_MAX_SIZE;
  arguments.n_threads = get_nprocs() * 2;
  arguments.quiet = FALSE;
  arguments.verbose = FALSE;
  arguments.no_lead = FALSE;
  arguments.no_pores = FALSE;
  arguments.source = SOURCE_NAME;
  arguments.dest_pores = DEST_NAME_PORES;
  arguments.dest_lead = DEST_NAME_LEAD;

  /* Parse our arguments; every option seen by parse_opt will
     be reflected in arguments. */
  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  // Apply global options to the global variables
  config.n_threads = arguments.n_threads;
  config.quiet = arguments.quiet;
  config.verbose = arguments.verbose;

  // Print the selected arguments
  if (!config.quiet) {
    printf("Filename: %s\n", arguments.hdf_filename);
    if (config.verbose) {
      printf("Number of threads: %d\n", config.n_threads);
      printf("Quiet: %d\n", config.quiet);
      printf("Verbose: %d\n", config.verbose);
      printf("Skip lead: %d\n", arguments.no_lead);
      printf("Skip pores: %d\n", arguments.no_pores);
    }
    printf("Source dataset: %s\n", arguments.source);
    if ((!arguments.no_pores) || config.verbose) {
      printf("Pores destination dataset: %s\n", arguments.dest_pores);
      printf("Min pore size: %d\n", arguments.min_pore_size);
      printf("Max pore size: %d\n", arguments.max_pore_size);
    }
    if ((!arguments.no_lead) || config.verbose) {
      printf("Lead destination dataset: %s\n", arguments.dest_lead);
      printf("Min lead size: %d\n", arguments.min_lead_size);
      printf("Max lead size: %d\n", arguments.max_lead_size);
    }
  }

  // Validate the supplied options  
  if (!arguments.no_pores) {
    // Make sure min and max sizes are in the right order
    if (arguments.min_pore_size >= arguments.max_pore_size) {
      printf("Error: Max pore size (%d) must be larger than min pore size (%d).\n",
	     arguments.max_pore_size, arguments.min_pore_size);
      return -1;
    }
    // Check that kernel sizes are odd
    if (!(arguments.min_pore_size % 2) && (!config.quiet))
      printf("Warning: Min pore size (%d) should be an odd number.\n", arguments.min_pore_size);
    if (!(arguments.max_pore_size % 2) && (!config.quiet))
      printf("Warning: Max pore size (%d) should be an odd number.\n", arguments.max_pore_size);
  }
  if (!arguments.no_lead) {
    // Validate the supplied options
    if (arguments.min_lead_size >= arguments.max_lead_size) {
      printf("Error: Max lead size (%d) must be larger than min lead size (%d).\n",
	     arguments.max_lead_size, arguments.min_lead_size);
      return -1;
    }
    // Check that kernel sizes are odd
    if (!(arguments.min_lead_size % 2) && (!config.quiet))
      printf("Warning: Min lead size (%d) should be an odd number.\n", arguments.min_lead_size);
    if (!(arguments.max_lead_size % 2) && (!config.quiet))
      printf("Warning: Max lead size (%d) should be an odd number.\n", arguments.max_lead_size);
  }
  
  // Open the HDF5 file
  hid_t h5fp = H5Fopen(arguments.hdf_filename, // File name to be opened
		       H5F_ACC_RDWR,        // file access mode
		       H5P_DEFAULT          // file access properties list
		       );
  if (h5fp < 0) {
    fprintf(stderr, "Error: Could not open file %s\n", arguments.hdf_filename);
  }
  // Open the source datasets
  hid_t src_ds;
  src_ds = H5Dopen(h5fp, arguments.source, H5P_DEFAULT);
  if (src_ds < 0) {
    if (!H5Lexists(h5fp, arguments.source, H5P_DEFAULT)) {
      fprintf(stderr, "Source dataset '%s' not found.\n", arguments.source);
    } else {
      fprintf(stderr, "Error opening source dataset '%s': %ld\n", arguments.source, src_ds);
    }
    return -1;
  }
  hid_t src_space = H5Dget_space(src_ds);
  
  // Prepare and segment the pores
  char return_val = 0;
  if (!arguments.no_pores) {
    hid_t dst_ds_pores;
    char result_pores = 0;
    // Create a new destination dataset
    dst_ds_pores = tp_replace_dataset(arguments.dest_pores, h5fp, src_space);
    // Apply the morpohology filters to extract the pore and lead structures
    result_pores = tp_extract_pores(src_ds, dst_ds_pores, h5fp, arguments.dest_pores,
				    arguments.min_pore_size, arguments.max_pore_size);
    // Close dataset
    H5Dclose(dst_ds_pores);
    // Check if the pore structure extraction finished successfully
    if (result_pores < 0) {
      fprintf(stderr, "Failed to extract pores %s: %d\n", arguments.source, result_pores);
      return_val = -1;
    }
  }
  if (!arguments.no_lead) {
    hid_t dst_ds_lead;
    char result_lead = 0;
    // Create a new destination dataset
    dst_ds_lead = tp_replace_dataset(arguments.dest_lead, h5fp, src_space);
    // Apply the morpohology filters to extract the pore and lead structures
    result_lead = tp_extract_lead(src_ds, dst_ds_lead, h5fp, arguments.dest_lead,
				  arguments.min_lead_size, arguments.max_lead_size);
    // Close destination dataset
    H5Dclose(dst_ds_lead);
    // Check if the free-lead extraction finished successfully
    if (result_lead < 0) {
      fprintf(stderr, "Failed to extract lead %s: %d\n", arguments.source, result_lead);
      return_val = -1;
    }
  }
  // Close all the common datasets, dataspaces, etc
  H5Dclose(src_ds);
  H5Fclose(h5fp);

  // Report the total amount of time used
  if (!config.quiet) {
    time_t end_time = time(NULL);
    printf("\nFinished in %ld seconds.\n", end_time - start_time + 1);
  }
  return return_val;
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
DDIM tp_indices(Matrix3D *vol, DIM islice, DIM irow, DIM icolumn) {
  return islice * vol->nrows * vol->ncolumns + irow * vol->ncolumns + icolumn;
}

// Take a slice and return a flattened index for a given row and column
DDIM tp_indices2d(Matrix2D *vol, DIM irow, DIM icolumn) {
  return irow * vol->ncolumns + icolumn;
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


DIM min_d(DIM x, DIM y) 
{
  return (x < y) ? x : y;
}


DIM max_d(DIM x, DIM y) 
{
  return (x > y) ? x : y;
}
