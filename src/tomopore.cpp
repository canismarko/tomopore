/* Apply a series of filters to the volume using 3D kernels This is done */
/* with a 3D kernel so that we don't lose data, but that means lots more */
/* memory. To avoid running out of memory, intermediate arrays are saved */
/* in HDF5 datasets. */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <hdf5.h>
#include <argp.h>
#include <sys/sysinfo.h>
#include <time.h>

#include "config.h"
#include "hdfhelpers.h"
#include "filters.h"

// Default values
const DIM PORE_MIN_SIZE = 5;
const DIM PORE_MAX_SIZE = 51;
const DIM LEAD_MIN_SIZE = 5;
const DIM LEAD_MAX_SIZE = 35;
const char *SOURCE_NAME = "volume";
const char *DEST_NAME_PORES = "pores";

// Documentation for the ``--help`` argument
const char *argp_program_version =
  "tomopore 0.1";
const char *argp_program_bug_address =
  "<wolfman@anl.gov>";
// Program description
static char doc[] =
  "Tomopore -- Memory efficient 3D extraction of pores from tomography data";
// A description of the arguments we accept
static char args_doc[] =
  "H5_FILE";

enum tophat_direction {black, white};

// Global variables
/* static struct { */
/*   int n_threads, verbose, quiet; */
/* } config; */

#define OPT_MIN_PORE_SIZE 1
#define OPT_MAX_PORE_SIZE 2
#define OPT_MIN_LEAD_SIZE 3
#define OPT_MAX_LEAD_SIZE 4
#define OPT_DEST 'd'
#define OPT_NO_PORES 5
#define OPT_NO_LEAD 6

static struct argp_option options[] = {
  {"threads", 'j', "NUM_THREADS", 0, "Number of parallel threads, defaults to using all cores", 0},
  {"verbose",  'v', 0,      0,  "Produce verbose output", 0},
  {"quiet",    'q', 0,      0,  "Don't produce any output", 0},
  
  {"source", 's', "DATASET", 0, "Path to the source dataset containing float volume data", 1},
  {"dest", OPT_DEST, "DATASET", 0, "Path to the dataset that will receive segmented pores", 1},

  {"min-pore-size", OPT_MIN_PORE_SIZE, "SIZE", 0, "Minimum size of pores (in pixels)", 3},
  {"max-pore-size", OPT_MAX_PORE_SIZE, "SIZE", 0, "Maximum size of pores (in pixels)", 3},
  {"white-tophat", 'w', 0, 0, "Identify pores using bright regions, instead of dark.", 3},
  
  { 0 }
};

/* Used by main to communicate with parse_opt. */
struct arguments
{
  char *hdf_filename, *source, *dest;
  DIM min_pore_size, max_pore_size;
  int n_threads, quiet, verbose, white_tophat;
};

/* Parse a single option. */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
  struct arguments *arguments = (struct arguments *) state->input;

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

    case OPT_MIN_PORE_SIZE:
      arguments->min_pore_size = atoi(arg);
      break;

    case OPT_MAX_PORE_SIZE:
      arguments->max_pore_size = atoi(arg);
      break;

    case 's':
      arguments->source = arg;
      break;

    case 'w':
      arguments->white_tophat = TRUE;
      break;      

    case OPT_DEST:
      arguments->dest = arg;
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


int main(int argc, char *argv[]) {
  // Save start time to measure total execution
  time_t start_time = time(NULL);
  
  struct arguments arguments;
  /* Default option values. */
  arguments.min_pore_size = PORE_MIN_SIZE;
  arguments.max_pore_size = PORE_MAX_SIZE;
  arguments.n_threads = get_nprocs() * 2;
  arguments.quiet = FALSE;
  arguments.verbose = FALSE;
  arguments.source = strdup(SOURCE_NAME);
  arguments.dest = strdup(DEST_NAME_PORES);
  arguments.white_tophat = FALSE;

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
    }
    printf("Source dataset: %s\n", arguments.source);
    printf("Tophat color: %s\n", arguments.white_tophat ? "white" : "black");
    if (config.verbose) {
      printf("Pores destination dataset: %s\n", arguments.dest);
      printf("Min pore size: %lu\n", arguments.min_pore_size);
      printf("Max pore size: %lu\n", arguments.max_pore_size);
    }
  }

  // Validate the supplied options  
  // Make sure min and max sizes are in the right order
  if (arguments.min_pore_size >= arguments.max_pore_size) {
    printf("Error: Max pore size (%ld) must be larger than min pore size (%lu).\n",
	   arguments.max_pore_size, arguments.min_pore_size);
    return -1;
  }
  // Check that kernel sizes are odd
  if (!(arguments.min_pore_size % 2) && (!config.quiet))
    printf("Warning: Min pore size (%lu) should be an odd number.\n", arguments.min_pore_size);
  if (!(arguments.max_pore_size % 2) && (!config.quiet))
    printf("Warning: Max pore size (%lu) should be an odd number.\n", arguments.max_pore_size);
 
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
  hid_t src_type = H5Dget_type(src_ds);

  // Validate the input datatype
  size_t sz = H5Tget_size(src_type);;
  switch (H5Tget_class(src_type)) {
  case H5T_INTEGER:
    if (config.verbose) {
      printf("Input datatype: ");
      // Determine signed vs unsigned
      H5T_sign_t sgn = H5Tget_sign(src_type);
      switch (sgn) {
      case H5T_SGN_NONE:
	printf("u");
	break;
      }
      // Print integer bit depth
      printf("int%d ", (int) (sz * sizeof(H5T_NATIVE_INT)));
      // Print endianism
      H5T_order_t ord = H5Tget_order(src_type);
      switch (ord) {
      case H5T_ORDER_LE:
	printf("(LE)");
	break;
      case H5T_ORDER_BE:
	printf("(BE)");
	break;
      }
      printf("\n");
    }
    if (!config.quiet) {
      printf("Warning: Possible precision loss converting integers on disk to floats in memory.\n");
    }
    break;
  case H5T_FLOAT:
    if (config.verbose) {
      printf("Input datatype: float%d\n", (int) (sz * sizeof(H5T_NATIVE_INT)));
    }
    break;
  case H5T_STRING:
    printf("Error: cannot segment datatype H5T_STRING\n");
    exit(-1);
    break;
  case H5T_BITFIELD:
    printf("Error: cannot segment datatype H5T_BITFIELD\n");
    exit(-1);
    break;
  case H5T_OPAQUE:
    printf("Error: cannot segment datatype H5T_OPAQUE\n");
    exit(-1);
    break;
  case H5T_COMPOUND:
    printf("Error: cannot segment datatype H5T_COMPOUND\n");
    exit(-1);
    break;
  case H5T_REFERENCE:
    printf("Error: cannot segment datatype H5T_REFERENCE\n");
    exit(-1);
    break;
  case H5T_ENUM:
    printf("Error: cannot segment datatype H5T_ENUM\n");
    exit(-1);
    break;
  case H5T_VLEN:
    printf("Error: cannot segment datatype H5T_VLEN\n");
    exit(-1);
    break;
  case H5T_ARRAY:
    printf("Error: cannot segment datatype H5T_ARRAY\n");
    exit(-1);
    break;
  default:
    printf("Error: unknown datatype %ld\n", src_type);
    exit(-1);
    break;
  }


  // Prepare and segment the pores
  char return_val = 0;
  hid_t dst_ds_pores;
  char result_pores = 0;
  // Create a new destination dataset
  dst_ds_pores = tomopore::replace_dataset(arguments.dest, h5fp, src_space, src_type);
  // Apply the morpohology filters to extract the pore and lead structures
  result_pores = tomopore::extract_pores(src_ds, dst_ds_pores, h5fp, arguments.dest,
					 arguments.min_pore_size, arguments.max_pore_size,
					 arguments.white_tophat);
  // Close dataset
  H5Dclose(dst_ds_pores);
  // Check if the pore structure extraction finished successfully
  if (result_pores < 0) {
    fprintf(stderr, "Failed to extract pores %s: %d\n", arguments.source, result_pores);
    return_val = -1;
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
