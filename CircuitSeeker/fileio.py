import SimpleITK as sitk
import h5py
from glob import glob
from os import path
import numpy as np

import dask.bag as db
import dask.array as da


from dask.delayed import delayed

def testPathExtensionForHDF5(image_path):
    """
    Returns true if `image_path` has an hdf5 like extension
    Currently: `['.h5', '.hdf5']
    """

    hdf5_extensions = ['.h5', '.hdf5']
    if image_path in hdf5_extensions:
        return True
    else:
        ext = path.splitext(image_path)[-1]
        return ext in hdf5_extensions


def globPaths(folder, prefix, suffix):
    """
    Returns sorted list of all absolue paths matching `folder/prefix*suffix`
    If no such paths are found, throws AsserionError
    """

    folder = path.abspath(folder)
    images = sorted(glob(path.join(folder, prefix) + '*' + suffix))
    assert (len(images) > 0), f"No images in {folder} matching {prefix}*{suffix}"
    return images


def daskBagOfFilePaths(folder, prefix, suffix, npartitions=None, slicer=None):
    """
    Returns dask.bag of absolute paths matching `folder/prefix*suffix`
    Specify bag partitions with `npartitions`; default None
    if `npartitions == None` then each filepath is its own partition
    """
    images = globPaths(folder, prefix, suffix)
    if slicer is None: slicer = slice(len(images))
    images = images[slicer]
    if npartitions is None: npartitions = len(images)
    return db.from_sequence(images, npartitions=npartitions)


def daskArrayBackedByHDF5(folder, prefix, suffix, dataset_path):
    """
    Returns dask.array backed by HDF5 files matching absolute path `folder/prefix*suffix`
    You must specify hdf5 dataset path with `dataset_path`
    """

    error_message = "daskArrayBackedByHDF5 requires hdf5 files with .h5 or .hdf5 extension"
    assert (testPathExtensionForHDF5(suffix)), error_message
    images = globPaths(folder, prefix, suffix)

    ex = readHDF5(images[0], dataset_path)
    readHDF5_d = lambda img: da.from_delayed(
        delayed(readHDF5)(img, dataset_path), ex.shape, ex.dtype
    )
    arrays = [readHDF5_d(image) for image in images]

    return da.stack(arrays, axis=0)


def readHDF5(image_path, dataset_path):
    """
    Returns (lazy) h5py array to dataset at `image_path[dataset_path]`
    """

    error_message = "readHDF5 requires hdf5 files with .h5 or .hdf5 extension"
    assert (testPathExtensionForHDF5(image_path)), error_message
    return h5py.File(image_path, 'r')[dataset_path][:]


def return_file_name(path):
    import ntpath
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def return_dir(path):
    import os
    return os.path.dirname(path)

def format_integer_to_zebrascope_standard(integer, CM):
    """convert an interger string into a stack file indexing format
    """

    return f'TM{str(integer).zfill(7)}_CM{CM}_CHN00'

def readImage(image_path, dataset_path=None):
    """
    Returns array like object for dataset at `image_path`
    If `image_path` is an hdf5 file, you must specify `dataset_path`
    If `image_path` is an hdf5 file, the return array is a lazy h5py File object
    Else, `image_path` is read by SimpleITK and an in memory numpy array is returned
    File format must be either hdf5 or supported by SimpleITK file readers
    """

    try:
        if testPathExtensionForHDF5(image_path):
            assert(dataset_path is not None), "Must provide dataset_path for .h5/.hdf5 files"
            return readHDF5(image_path, dataset_path)
        else:
            return sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    except:

        import shutil
        import os


        filename = return_file_name(image_path)
        directory = return_dir(image_path)
        timepoint = int(filename[2:9])
        CM = int(filename[12:13])

        if timepoint > 0:
            replacement_timepoint = timepoint - 1
        else:
            replacement_timepoint = timepoint + 1

        replacement_filename = format_integer_to_zebrascope_standard(replacement_timepoint, CM)+ '.h5'
        replacement_path = os.path.join(directory, replacement_filename)

        ## make a note about offending file
        pre, ext = os.path.splitext(image_path)
        txt_path = pre + '.txt'

        ## if txt_path exists, h5 file was already replaced once
        if os.path.exists(txt_path):
            raise
        else:
            with open(txt_path, 'w') as f:
                f.write(f"Cannot be read. Replaced by: {replacement_path}")

        os.rename(image_path, image_path.replace('/TM', '/original_TM'))
        shutil.copy(replacement_path, image_path)

        return readImage(image_path, dataset_path=dataset_path)

def writeHDF5(image_path, dataset_path, array):
    """
    Writes `array` to `image_path[dataset_path]` as an hdf5 file
    """

    with h5py.File(image_path, 'w') as f:
        dset = f.create_dataset(dataset_path, array.shape, array.dtype)
        dset[...] = array


def writeImage(image_path, array, spacing=None, axis_order='zyx'):
    """
    Writes `array` to `image_path`
    `image_path` extension determines format - must be supported by SimpleITK image writers
    Many formats support voxel spacing in the metadata, to specify set `spacing`
    If format supports voxel spacing in meta data, you can set with `spacing`
    Use `axis_order` to specify current axis ordering; default: 'zyx'
    """

    # argsort axis_order
    transpose_order = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(axis_order))]
    array = array.transpose(transpose_order[::-1])  # sitk inverts axis order
    img = sitk.GetImageFromArray(array)
    if spacing is not None:
        spacing = spacing[transpose_order]
        img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, image_path)


def ensureArray(reference, dataset_path):
    """
    """

    if not isinstance(reference, np.ndarray):
        if not isinstance(reference, str):
            raise ValueError("image references must be ndarrays or filepaths")
        reference = readImage(reference, dataset_path)[...]  # hdf5 arrays are lazy
    return reference

