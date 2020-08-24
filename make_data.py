import h5py
from tomopy.misc import phantom

def main():
    vol = phantom.shepp3d(256)
    with h5py.File('data/phantom3d_big.h5', mode='a') as h5fp:
        h5fp.create_dataset('volume', data=vol)


if __name__ == "__main__":
    main()
