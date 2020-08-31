#include <hdf5.h>

hid_t tp_replace_dataset(char *dataset_name, hid_t h5fp, hid_t dataspace);
hid_t tp_require_dataset(char *dataset_name, hid_t h5fp, hid_t dataspace);
