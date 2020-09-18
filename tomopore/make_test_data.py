#!/bin/env python

import sys
from pathlib import Path

import h5py
import numpy as np
from tomopy.misc import phantom
import matplotlib.pyplot as plt


h5file = (Path(__file__)/'../../data/phantom3d.h5').resolve()
vol_size = (64, 64, 64)


def main():
    with h5py.File(h5file, mode='a') as h5fp:
        # vol = phantom.shepp3d(vol_size, 'int')
        combos = [
            ('volume', 'float32', 1),
            ('volume_int', 'int', 16),
            ('volume_uint', 'uint', 16),
            ('volume_bool', 'bool', 1.),
        ]
        vol = phantom.shepp3d(vol_size)
        for (name, dtype, mult) in combos:
            # Delete existing datasets
            if name in h5fp.keys():
                del h5fp[name]
            # Scale the dataset to match the datatype
            new_vol = (vol * mult).astype(dtype)
            # Save to disk
            h5fp.create_dataset(name, data=new_vol)
    return 0


if __name__ == '__main__':
    sys.exit(main())
