#include <hdf5.h>

namespace tomopore {

  hid_t replace_dataset(char *dataset_name, hid_t h5fp, hid_t dataspace, hid_t datatype);
  hid_t require_dataset(char *dataset_name, hid_t h5fp, hid_t dataspace, hid_t datatype);

}
