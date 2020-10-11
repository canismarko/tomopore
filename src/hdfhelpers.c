#include "hdfhelpers.h"
#include "config.h"

namespace tomopore {

  hid_t replace_dataset
    (char *dataset_name, hid_t h5fp, hid_t dataspace, hid_t datatype)
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
    new_dataset_id = tomopore::require_dataset(dataset_name, h5fp, dataspace, datatype);
    return new_dataset_id;
  }


  hid_t require_dataset
    (char *dataset_name, hid_t h5fp, hid_t dataspace, hid_t datatype)
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
				 datatype, // Datatype identifier
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

}
