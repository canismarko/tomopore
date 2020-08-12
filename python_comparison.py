from skimage import morphology
import h5py


def main():
    selem = morphology.cube(5)
    with h5py.File("data/phantom3d.h5", mode="r+") as h5fp:
        # Delete existing datasets
        for ds_name in ['python_opening_5pxbox', 'python_closing_5pxbox',
                        'python_erosion_5pxbox', 'python_dilation_5pxbox']:
            if ds_name in h5fp.keys():
                del h5fp[ds_name]
        # Do the filtering
        volume_ds = h5fp['volume']
        dilation_out = morphology.dilation(volume_ds, selem)
        h5fp.create_dataset('python_dilation_5pxbox', data=dilation_out)
        erosion_out = morphology.erosion(volume_ds, selem)
        h5fp.create_dataset('python_erosion_5pxbox', data=erosion_out)
        opening_out = morphology.opening(volume_ds, selem)
        h5fp.create_dataset('python_opening_5pxbox', data=opening_out)
        closing_out = morphology.closing(volume_ds, selem)
        h5fp.create_dataset('python_closing_5pxbox', data=closing_out)
        black_tophat_out = morphology.black_tophat(volume_ds, selem)
        h5fp.create_dataset('python_black_tophat_5pxbox', data=black_tophat_out)
        white_tophat_out = morphology.white_tophat(volume_ds, selem)
        h5fp.create_dataset('python_white_tophat_5pxbox', data=white_tophat_out)


if __name__ == '__main__':
    main()
