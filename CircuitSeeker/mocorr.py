from glob import glob
import numpy as np
import os

import CircuitSeeker.fileio as csio
import CircuitSeeker.distributed as csd
import dask.array as da
import dask.bag as db
import dask.delayed as delayed

import SimpleITK as sitk
import zarr
from numcodecs import Blosc

def ensureArray(reference, dataset_path=None):
    """
    """

    import dask.array as da
    if isinstance(reference, da.Array):
        reference = reference.compute()
    elif not isinstance(reference, np.ndarray):
        if not isinstance(reference, str):
            raise ValueError("image references must be ndarrays or filepaths")
        reference = csio.readImage(reference, dataset_path)[...]  # hdf5 arrays are lazy
    return reference


def rigidAlign(
        fixed, moving,
        fixed_vox, moving_vox,
        dataset_path=None,
        metric_sample_percentage=0.1,
        shrink_factors=[2,1],
        smooth_sigmas=[1,0],
        minStep=0.1,
        learningRate=1.0,
        numberOfIterations=50,
        target_spacing=2.0,
        pad_fixed=False,
):
    """
    Returns rigid transform parameters aligning `fixed` coords to `moving` coords
    `fixed` and `moving` must be numpy arrays or file paths
    `fixed_vox` and `moving_vox` must be fixed and moving image voxel spacings as numpy arrays
    if `fixed` and/or `moving` are hdf5 filepaths, you must specify `dataset_path`
    remaining arguments adjust the rigid registration algorithm

    Images are skip sub-sampled before registration. The skip stride is determined by
    `target_spacing` which is the target voxel spacing after skip sub-sampling.
    Images are never up-sampled so axes with spacing greater than `target_spacing` are
    not skip sub-sampled.
    """

    # get moving/fixed images as ndarrays
    fixed = ensureArray(fixed, dataset_path)
    if pad_fixed:
        pad_nvoxels = np.round(pad_fixed / fixed_vox).astype('int')
        fixed = np.pad(fixed, pad_width=((pad_nvoxels[0], pad_nvoxels[0]),
                                         (pad_nvoxels[1], pad_nvoxels[1]),
                                         (pad_nvoxels[2], pad_nvoxels[2])))
    moving = ensureArray(moving, dataset_path)

    # determine skip sample factors
    fss = np.maximum(np.round(target_spacing / fixed_vox), 1).astype(np.int)
    mss = np.maximum(np.round(target_spacing / moving_vox), 1).astype(np.int)

    # skip sample the images
    fixed = fixed[::fss[0], ::fss[1], ::fss[2]]
    moving = moving[::mss[0], ::mss[1], ::mss[2]]
    fixed_vox = fixed_vox * fss
    moving_vox = moving_vox * mss

    # convert to sitk images, set spacing
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)
    fixed.SetSpacing(fixed_vox[::-1])  # numpy z,y,x --> itk x,y,z
    moving.SetSpacing(moving_vox[::-1])

    # set up registration object
    irm = sitk.ImageRegistrationMethod()
    try:
        ncores = int(os.environ["LSB_DJOB_NUMPROC"])  # LSF specific!
    except:
        import multiprocessing
        ncores = multiprocessing.cpu_count()

    irm.SetNumberOfThreads(2*ncores)
    irm.SetInterpolator(sitk.sitkLinear)

    # metric, built for speed
    irm.SetMetricAsMeanSquares()
    irm.SetMetricSamplingStrategy(irm.RANDOM)
    irm.SetMetricSamplingPercentage(metric_sample_percentage)

    # optimizer, built for simplicity
    max_step = np.min(fixed_vox)
    irm.SetOptimizerAsRegularStepGradientDescent(
        minStep=minStep, learningRate=learningRate,
        numberOfIterations=numberOfIterations,
        maximumStepSizeInPhysicalUnits=max_step
    )
    irm.SetOptimizerScalesFromPhysicalShift()

    # pyramid
    irm.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    irm.SetSmoothingSigmasPerLevel(smoothingSigmas=smooth_sigmas)
    irm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # initialize
    irm.SetInitialTransform(sitk.Euler3DTransform())

    # execute, convert to numpy and return
    transform = irm.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                            sitk.Cast(moving, sitk.sitkFloat32),
    )
    return transform.GetParameters()

def rigidAlignAndSave(
        fixed, moving,
        fixed_vox, moving_vox,
        dataset_path=None,
        metric_sample_percentage=0.1,
        shrink_factors=[2,1],
        smooth_sigmas=[1,0],
        minStep=0.1,
        learningRate=1.0,
        numberOfIterations=50,
        target_spacing=2.0,
        pad_fixed=False,
        savepath=''
):
    transform = rigidAlign(fixed, moving, fixed_vox, moving_vox,
                           dataset_path=dataset_path,
                           metric_sample_percentage=metric_sample_percentage,
                           shrink_factors=shrink_factors,
                           smooth_sigmas=smooth_sigmas,
                           minStep=minStep,
                           learningRate=learningRate,
                           numberOfIterations=numberOfIterations,
                           pad_fixed=pad_fixed,
                           target_spacing=target_spacing)

    if savepath: np.save(savepath, transform)

    return transform


def planes_with_some_data(arr, percentage):

    nonzero_z, nonzero_y, nonzero_x = np.nonzero(arr)  # 2.78 s for arr of shape (70, 1152, 2048)

    plane_npixels = arr.shape[1] * arr.shape[2]

    unique, counts = np.unique(nonzero_z, return_counts=True) # 2.63 s  for arr of shape (70, 1152, 2048)
    ratio_counts = counts / plane_npixels
    plane_inds = unique[ratio_counts > percentage]

    return plane_inds


def applyTransform(
        moving,
        moving_vox,
        params,
        fixed=None,
        fixed_vox=None,
        resampled_slice=None,
        return_resampled_slice=False,
        dataset_path=None,
        pad_fixed=False
):
    """
    """

    params = ensureArray(params)
    transform = _parametersToEuler3DTransform(params)

    # get the moving image as a numpy array
    moving = ensureArray(moving, dataset_path)
    moving_nplanes = moving.shape[0]
    original_moving_dtype = moving.dtype  ## take note of original dtype before changing

    ## function does not accept float16 dtype:
    ## https://github.com/SimpleITK/SimpleITK/commit/88e2b1326ecfcf06309cb47d01371f95841170af#diff-a53c412a58e7abe996888298507037d7ac451eb3fce3956bd7a6fca35fc29245R667
    if original_moving_dtype == np.float16: moving = moving.astype(np.float32)

    # use sitk transform and interpolation to apply transform
    moving_im = sitk.GetImageFromArray(moving)
    moving_im.SetSpacing(moving_vox[::-1])  # numpy z,y,x --> itk x,y,z

    del moving  ## attempt to save memory

    if fixed is not None and fixed_vox is not None:

        fixed = ensureArray(fixed, dataset_path)

        if pad_fixed:
            pad_nvoxels = np.round(pad_fixed / fixed_vox).astype('int')
            fixed = np.pad(fixed, pad_width=((pad_nvoxels[0], pad_nvoxels[0]),
                                             (pad_nvoxels[1], pad_nvoxels[1]),
                                             (pad_nvoxels[2], pad_nvoxels[2])))

        fixed = fixed.astype(np.float32)
        align_to_fixed = True
        fixed_im = sitk.GetImageFromArray(fixed)
        fixed_im.SetSpacing(fixed_vox[::-1])

        del fixed  ## attempt to save memory

        transformed_im = sitk.Resample(moving_im, fixed_im, transform,
                                    sitk.sitkNearestNeighbor, 0.0, moving_im.GetPixelID()
                                    )
    else:
        transformed_im = sitk.Resample(moving_im, moving_im, transform,
                                    sitk.sitkNearestNeighbor, 0.0, moving_im.GetPixelID()
                                    )

    transformed = sitk.GetArrayFromImage(transformed_im)   ## float32

    # downsample in z because there's only so much data (e.g. from nearest neighbour)
    if align_to_fixed:

        from scipy.ndimage import zoom
        zoom_factor = fixed_vox/moving_vox
        zoomed = zoom(transformed, zoom_factor, order=0)  ## zoom every 5um since that's the voxel spacing of the functional data we collected

        del transformed  ## attempt to save memory

        if resampled_slice is None:
            # take the middle elements from zoomed
            plane_inds = planes_with_some_data(zoomed, 0.4)
            resampled_slice = slice(plane_inds[0],plane_inds[0]+moving_nplanes)

        zoomed = zoomed[resampled_slice]

    # return a numpy array
    if original_moving_dtype == np.float16:
        zoomed = zoomed.astype(np.float16)

    if return_resampled_slice:
        return zoomed, resampled_slice
    else:
        return zoomed

def applyTransformAndSave(moving, moving_vox, params, fixed=None, fixed_vox=None,
                          resampled_slice=None, return_resampled_slice=False, dataset_path=None, pad_fixed=False,
                          save_path='', chunks=(1,100,100)):
    """Apply the transform with an option to save if `savepath` is provided
    moving [np.ndarray or str]: the moving image, usually the functional image
    moving_vox [np.ndarray]: voxel spacing
    params [np.ndarray]: 6x1 array
    dataset_path [str]: if moving is provided as a h5 path, then the h5 reader needs to know where the data is stored
    save_path [str]: optionally, save the transformed image
    """

    if return_resampled_slice:
        transformed, resampled_slice = applyTransform(moving, moving_vox, params, fixed=fixed, fixed_vox=fixed_vox,
                                                      resampled_slice=resampled_slice, return_resampled_slice=return_resampled_slice,
                                                      dataset_path=dataset_path, pad_fixed=pad_fixed)
    else:
         transformed = applyTransform(moving, moving_vox, params, fixed=fixed, fixed_vox=fixed_vox,
                                      resampled_slice=resampled_slice, return_resampled_slice=return_resampled_slice,
                                      dataset_path=dataset_path, pad_fixed=pad_fixed)


    from analysis_toolbox.fileio import saveas_h5
    if save_path:
        saveas_h5(save_path, data=[transformed], dset_names=[dataset_path], chunks=chunks)
        return save_path

    else:
        return transformed

# useful format conversions for rigid transforms
def _euler3DTransformToParameters(euler):
    """
    """
    return np.array(( euler.GetAngleX(),
                      euler.GetAngleY(),
                      euler.GetAngleZ() ) +
                      euler.GetTranslation()
                   )
def _parametersToEuler3DTransform(params):
    """
    """
    transform = sitk.Euler3DTransform()
    transform.SetRotation(*params[:3])
    transform.SetTranslation(params[3:])
    return transform
def _parametersToRigidMatrix(params):
    """
    """
    transform = _parametersToEuler3DTransform(params)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[:3, -1] = np.array(transform.GetTranslation())
    return matrix


def motionCorrect(
        folder, prefix, suffix,
        fixed, fixed_vox, moving_vox,
        write_path=None, dataset_path=None,
        distributed_state=None, sigma=7,
        transforms_dir=None, folder_slicer=None, pad_fixed=False,
        params=None, t_chunksize=False, force_not_chunk=False,
        correct_another=None,
        t_indices=None,
        slice_transformed=(slice(None), slice(None), slice(None)),
        resume=False,
        **kwargs,
):
    """Perform motion correction on functional imaging data
    folder [str]: path to folder with functional imaging data files
    prefix [str]: prefix of .h5 files (e.g. 'TM')
    suffix [str]: suffix of .h5 files (e.g. '.h5')
    fixed [str or np.ndarray]: the reference image
    fixed_vox [np.ndarray]: fixed voxel spacing
    moving_vox [np.ndarray]: moving voxel spacing
    write_path [str]: where the transforms are written
    dataset_path [str]: the location of the dataset in a h5 file
    distributed_state [distributed_state, None, False]:
        distribute_state:
        None: if None is supplied, then one will be created for you
        False: none is supplied and  you don't want one to be created for you
    sigma [int]:
    transforms_dir [str]: path to where the transformed images are saved
    folder_slicer [slice]: if only a portion of frames in folders are of interest
    params [np.ndarray]: transformation params
    t_chunksize [bool/int]: size of chunks determining whether or not to chunk the arrays. chunking reduces the size of the task graph
    force_not_chunk [bool]: force not to chunk. takes precedence over 't_chunksize'
    correct_another [da.Array]: dask array of another set of frames that the transformations will be applied
    t_indices [np.ndarray]: only transform select frames (only works when chunking and starting fresh)
    slice_transformed [tuple]: slice the transformed image as early as possible, right after it was transformed in an attempt to save memory
    """

    # set up the distributed environment
    ds = distributed_state
    if distributed_state is None:
        ds = csd.distributedState()
        # writing large compressed chunks locks GIL for a long time
        ds.modifyConfig({'distributed.comm.timeouts.connect':'60s',
                         'distributed.comm.timeouts.tcp':'180s',}
        )
        ds.initializeLSFCluster(
            # job_extra=["-P scicompsoft"]
        )
        ds.initializeClient()

    # create (lazy) dask bag from all frames
    frames = csio.daskBagOfFilePaths(folder, prefix, suffix, slicer=folder_slicer)
    nframes = frames.npartitions

    # scale cluster carefully
    if distributed_state is None:
        if 'max_workers' in kwargs.keys():
            max_workers = kwargs['max_workers']
        else:
            max_workers = 1250
        ds.scaleCluster(njobs=min(nframes, max_workers))

    # align all
    dfixed = delayed(fixed)
    dfixed_vox = delayed(fixed_vox)
    dmoving_vox = delayed(moving_vox)
    ddataset_path = delayed(dataset_path)

    if params is None:
        params = alignFramesToReference(frames, dfixed, dfixed_vox, dmoving_vox,
                                        sigma, ddataset_path,
                                        transforms_dir=transforms_dir, resume=resume, pad_fixed=pad_fixed)


    # transform frames with params
    chunksize = 5000

    ## SOME INCONSISTENCIES between applying to chunks and not applying
    ## 1. Chunking saves to h5 while not saves to zarr
    ## 2. Resuming is not enabled on not chunking
    if (nframes > chunksize or num_tchunks) and not force_not_chunk:

        print("Applying over chunks of frames...")

        if correct_another is not None:
            frames_to_correct = correct_another
        else:
            frames_to_correct = folder  # str of the folder of the motion corrected .h5 files
            # TODO: this is unequal to the not_chunk, which takes in a dask bag of file paths

        # work on chunks of frames -- reduce size of task graph
        if not t_chunksize: t_chunksize = 10


        from analysis_toolbox.utils import find_files
        actual_write_paths = find_files(write_path + '/', ext='h5', compute_path=True)['path']

        if resume:

            from tqdm.notebook import tqdm

            print("Resuming application of transforms...")
            from analysis_toolbox.dataset_helper import format_integer_to_zebrascope_standard
            expected_write_paths = [write_path+f'/{format_integer_to_zebrascope_standard(index)}.h5' for index in np.arange(len(params))]
            missing_write_paths = np.setdiff1d(expected_write_paths, actual_write_paths)
            missing_indices = np.array([np.where(np.array(expected_write_paths)==missing_write_path)[0][0] for missing_write_path in tqdm(missing_write_paths)])

            if len(actual_write_paths) == 0:

                print("nothing computed yet, start fresh!")
                if t_indices is None:
                    t_indices = np.arange(nframes)
                npartitions = len(t_indices)// t_chunksize
                indices = db.from_sequence(t_indices, npartitions=npartitions)
                computing = True

            elif len(missing_indices) > 0:

                print(f"{len(missing_indices)}/{len(expected_write_paths)} missing")

                npartitions = len(missing_indices) // t_chunksize
                indices = db.from_sequence(missing_indices, npartitions=npartitions)
                computing = True

            elif len(missing_indices) == 0:

               computing = False

        else:

            print("Starting fresh!")

            if t_indices is None:
                t_indices = np.arange(nframes)

            npartitions = len(t_indices) // t_chunksize
            indices = db.from_sequence(t_indices, npartitions=npartitions)
            computing = True



        if computing:

            if isinstance(frames_to_correct, str):
                filename, fileext = os.path.splitext(frames_to_correct)
                if fileext == '.zarr':
                    import zarr
                    frame_paths = zarr.open(frames_to_correct, mode='r')
                else:
                    frame_paths = np.array(csio.globPaths(frame_to_correct, suffix='.h5', prefix='TM'))  ## convert to array for array indexing
            elif isinstance(frame_to_correct, da.Array):
                fileext = None
                pass

            resampled_slice_ref_index = 0
            _, resampled_slice = applyTransform(frame_paths[resampled_slice_ref_index], moving_vox, params[resampled_slice_ref_index],
                                                fixed=fixed, fixed_vox=fixed_vox, return_resampled_slice=True, dataset_path=dataset_path)
            ## TODO: replace applyTransformToChunksOfFrames with applyTransformToAChunkOfFrames and use the former wrap everything in this block
            transformed = indices.map_partitions(applyTransformToChunksOfFrames,
                                                 frame_dir=frames_to_correct, params_path=transforms_dir + '/params.npy',
                                                 moving_vox=moving_vox, dataset_path=dataset_path,
                                                 resampled_slice=resampled_slice,
                                                 slice_transformed=slice_transformed, write_path=write_path,
                                                 fixed=dfixed, fixed_vox=dfixed_vox).to_delayed()

            if write_path is None:
                indices_len = list(indices.map_partitions(lambda x: len(x)).compute())

                ## use the original frames pre-motion correction as an example
                if isinstance(frames_to_correct, str):
                    example_image = csio.readImage(frames.compute()[0], dataset_path=dataset_path)[...]
                elif isinstance(frames_to_correct, da.Array):
                    example_image = frames_to_correct[0]

                shape = example_image.shape
                dtype = example_image.dtype

                arrays = [da.from_delayed(t, shape=(ilen, *shape), dtype=dtype) for t, ilen in zip(transformed,indices_len)]  ## if resuming, this won't be the full array
                transformed = da.concatenate(arrays, axis=0)

        else:

            from fish.util.fileio import to_dask
            transformed = to_dask(actual_write_paths)

    else:

        ## TODO: resume not enabled here!
        print("Applying over frames...")

        if correct_another is not None:
            if isinstance(correct_another, str):
                correct_another = da.from_zarr(correct_another)
            ## expecting dask array
            frames_to_correct = db.from_sequence(correct_another, npartitions=correct_another.shape[0])  # takes a really long time
        else:
            frames_to_correct = frames  # dask bag of file paths

        # work on each frame separately -- better for parallelism
        transformed = applyTransformToFrames(frames_to_correct, params, dmoving_vox, ddataset_path,
                                             slice_transformed=slice_transformed, write_path=write_path,
                                             fixed=dfixed, fixed_vox=dfixed_vox)

    # release resources
    if distributed_state is None:
        ds.closeClient()

    # return reference to data
    return params, transformed

def runAlignFramesToReference(
    folder, prefix, suffix,
    fixed, fixed_vox, moving_vox,
    dataset_path=None,
    distributed_state=None, sigma=7,
        transforms_dir=None, folder_slicer=None, pad_fixed=False,
    **kwargs,
):
    """
    """

    # set up the distributed environment
    ds = distributed_state
    if distributed_state is None:
        ds = csd.distributedState()
        # writing large compressed chunks locks GIL for a long time
        ds.modifyConfig({'distributed.comm.timeouts.connect':'60s',
                         'distributed.comm.timeouts.tcp':'180s',}
        )
        ds.initializeLSFCluster(
            # job_extra=["-P scicompsoft"]
        )
        ds.initializeClient()

    if folder_slicer is None: folder_slicer = slice(len(files))
    # create (lazy) dask bag from all frames
    frames = csio.daskBagOfFilePaths(folder, prefix, suffix, slicer=folder_slicer)
    nframes = frames.npartitions

    # scale cluster carefully
    if distributed_state is None:
        if 'max_workers' in kwargs.keys():
            max_workers = kwargs['max_workers']
        else:
            max_workers = 1250
        ds.scaleCluster(njobs=min(nframes, max_workers))

    # align all
    dfixed = delayed(fixed)
    dfixed_vox = delayed(fixed_vox)
    dmoving_vox = delayed(moving_vox)
    ddataset_path = delayed(dataset_path)

    params = alignFramesToReference(frames, dfixed, dfixed_vox, dmoving_vox,
                                    sigma, ddataset_path,
                                    transforms_dir=transforms_dir, pad_fixed=pad_fixed)

    # release resources
    if distributed_state is None:
        ds.closeClient()

    # return reference to data
    return params


def alignFramesToReference(frames, dfixed, dfixed_vox, dmoving_vox,
                           sigma, ddataset_path,
                           resume=True, transforms_dir=None, pad_fixed=False):
    """
    frames [db.Bag]: dask bag of file paths
    """

    from scipy.ndimage import percentile_filter, gaussian_filter1d

    paths = list(frames)
    expected_param_savepaths = [os.path.join(transforms_dir, os.path.splitext(os.path.basename(path))[0] + '_rigid.npy') for path in paths]

    if resume:
        from analysis_toolbox.utils import find_files
        from tqdm.notebook import tqdm

        actual_param_savepaths = find_files(transforms_dir + '/', grep='TM', ext='npy', compute_path=True)['path']

        if len(actual_param_savepaths) == 0:

            resume = False  ## nothing to resume
            savepaths = db.from_sequence(expected_param_savepaths, npartitions=len(expected_param_savepaths))

        else:

            missing_param_savepaths = np.setdiff1d(expected_param_savepaths, actual_param_savepaths)
            savepaths = db.from_sequence(missing_param_savepaths, npartitions=len(missing_param_savepaths))

            missing_indices = np.array([np.where(np.array(expected_param_savepaths)==missing_param_savepath)[0][0] for missing_param_savepath in tqdm(missing_param_savepaths)])
            paths = np.array(paths)[missing_indices]
            frames = db.from_sequence(paths, npartitions=len(paths))

    else:

        savepaths = db.from_sequence(expected_param_savepaths)

    params = frames.map(lambda b,c,d,w,x,y,z: rigidAlignAndSave(w,b,x,y, dataset_path=z, pad_fixed=d, savepath=c),
                        w=dfixed, x=dfixed_vox, y=dmoving_vox, z=ddataset_path, d=pad_fixed, c=savepaths,
    ).compute()

    if not resume:
        params = np.array(list(params))
    else:
        ## reload from files
        params = np.stack([np.load(expected_param_savepath, allow_pickle=True) for expected_param_savepath in tqdm(expected_param_savepaths)])

    # (weak) outlier removal and smoothing
    params = percentile_filter(params, 50, footprint=np.ones((3,1)))
    params = gaussian_filter1d(params, sigma, axis=0)

    # write transforms as matrices
    if transforms_dir is not None:
        if not os.path.exists(transforms_dir): os.makedirs(transforms_dir)
        np.save(transforms_dir + '/params.npy', params)

        for ind, p in enumerate(params):
            transform = _parametersToRigidMatrix(p)
            basename = os.path.splitext(os.path.basename(paths[ind]))[0]
            path = os.path.join(transforms_dir, basename) + '_rigid.mat'
            np.savetxt(path, transform)

    return params


def applyTransformToChunksOfFrames(indices, frame_dir, params_path, moving_vox, dataset_path,
                                   resampled_slice=None,
                                   slice_transformed=(slice(None), slice(None), slice(None)),
                                   index_first=False, write_path='',
                                   fixed=None, fixed_vox=None, pad_fixed=False):
    """Apply transform to a few frames instead of one at a time
    Relies on a for loop at the end which accepts a single frame and param set.

    indices [np.ndarray]: indices to index both frame_dir and params_path
    frame_dir [str or da.Array]: either a string pointing to the directory with files to work on or a dask array of the frames to work on
    params_path [str]: path with params.npy
    moving_vox [np.ndarray]: moving image voxel spacing
    dataset_path [str]: .h5 data path
    """

    ## determine frame_paths and load params
    if isinstance(frame_dir, str):
        filename, fileext = os.path.splitext(frame_dir)
        if fileext == '.zarr':
            import zarr
            # frame_paths = da.from_zarr(frame_dir, inline_array=True)
            frame_paths = zarr.open(frame_dir, mode='r')
        else:
            frame_paths = np.array(csio.globPaths(frame_dir, suffix='.h5', prefix='TM'))  ## convert to array for array indexing
    elif isinstance(frame_dir, da.Array):
        fileext = None
        pass

    params = np.load(params_path)

    if write_path:
        if not os.path.exists(write_path):
            from pathlib import Path
            Path(write_path).mkdir(parents=True, exist_ok=True)

        ## inconsistent between the two conditions
        if fileext == '':
            from functools import partial
            change_root_dir_in_path_for_transformed = partial(change_root_dir_in_path, replace_with=write_path)
            save_paths = np.vectorize(change_root_dir_in_path_for_transformed)(frame_paths)

        elif fileext == '.zarr' or fileext == '.h5':
            from analysis_toolbox.dataset_helper import format_integer_to_zebrascope_standard
            save_paths = [write_path+f'/{format_integer_to_zebrascope_standard(index)}.h5' for index in np.arange(len(params))]

    else:
        save_paths = np.full((len(params),), '')

    if isinstance(indices, list): indices = np.array(indices)
    indices = indices.astype('int')

    if index_first:
        ## index into frame_paths and params
        indexed_frame_paths = frame_paths[indices]
        indexed_params = params[indices]
        indexed_save_pahts = save_paths[indices]

        # simplify getting item by computing once, here
        # if isinstance(indexed_frame_paths, da.Array):
        #     indexed_frame_paths = indexed_frame_paths.compute()

        out  = [applyTransformAndSave(frame_path, moving_vox, param,
                                      dataset_path=dataset_path, save_path=save_path,
                                      fixed=fixed, fixed_vox=fixed_vox, resampled_slice=resampled_slice, pad_fixed=pad_fixed)  \
                for frame_path, param, save_path in zip(indexed_frame_paths, indexed_params, indexed_save_paths)]

        del params, indices, indexed_frame_paths, indexed_params  ## atttmpt to save memeory

    else:
        out = [applyTransformAndSave(frame_paths[index], moving_vox, params[index],
                                     dataset_path=dataset_path, save_path=save_paths[index],
                                     fixed=fixed, fixed_vox=fixed_vox, resampled_slice=resampled_slice, pad_fixed=pad_fixed) \
               for index in indices]

        del params, indices

    return out


def applyTransformToFrames(frames, params, dmoving_vox, ddataset_path,
                           slice_transformed=(slice(None), slice(None), slice(None)), write_path=None,
                           fixed=None, fixed_vox=None, pad_fixed=False):
    """
    frames [db.Bag]: a dask bag either containing paths or a tzyx dask array
    params [np.ndarray]: params corresponding to the frames
    dmoving_vox [delayed]: voxel spacing of the moving image
    ddataset_path [delayed]: h5 dataset path
    """

    nframes = frames.npartitions
    npartitions = nframes

    # apply transforms to all images
    params = db.from_sequence(params, npartitions=npartitions)
    transformed = frames.map(lambda b,x,y,z,p,q,r: applyTransform(b,x,y, dataset_path=z, fixed=p, fixed_vox=q, pad_fixed=r),
                             x=dmoving_vox, y=params, z=ddataset_path, p=fixed, q=fixed_vox, r=pad_fixed,
    ).to_delayed()

    # convert to a (lazy) 4D dask array
    sh = transformed[0][0].shape.compute()
    dd = transformed[0][0].dtype.compute()
    arrays = [da.from_delayed(t[0][slice_transformed], sh, dtype=dd) for t in transformed]
    transformed = da.stack(arrays, axis=0)

    if write_path is not None:
        if not os.path.exists(write_path): os.makedirs(write_path)
        # write in parallel as 4D array to zarr file
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
        transformed_disk = zarr.open(write_path, 'w',
            shape=transformed.shape, chunks=(256, 10, 256, 256),
            dtype=transformed.dtype, compressor=compressor
        )
        da.to_zarr(transformed, transformed_disk)

    return transformed


def distributedImageMean(
    folder, prefix, suffix, dataset_path=None,
    distributed_state=None, write_path=None,
):
    """
    Returns mean over images matching `folder/prefix*suffix`
    If images are hdf5 you must specify `dataset_path`
    To additionally write the mean image to disk, specify `write_path`
    Computations are distributed, to supply your own dask scheduler and cluster set
        `distributed_state` to an existing `CircuitSeeker.distribued.distributedState` object
        otherwise a new cluster will be created
    """

    # set up the distributed environment
    ds = distributed_state
    if distributed_state is None:
        ds = csd.distributedState()
        ds.initializeLSFCluster(job_extra=["-P scicompsoft"])
        ds.initializeClient()

    # hdf5 files use dask.array
    if csio.testPathExtensionForHDF5(suffix):
        frames = csio.daskArrayBackedByHDF5(folder, prefix, suffix, dataset_path)
        nframes = frames.shape[0]
        if distributed_state is None: ds.scaleCluster(njobs=nframes)
        frames_mean = frames.mean(axis=0).compute()
        frames_mean = np.round(frames_mean).astype(frames[0].dtype)
    # other types use dask.bag
    else:
        frames = csio.daskBagOfFilePaths(folder, prefix, suffix)
        nframes = frames.npartitions
        if distributed_state is None: ds.scaleCluster(njobs=nframes)
        frames_mean = frames.map(csio.readImage).reduction(sum, sum).compute()
        dtype = frames_mean.dtype
        frames_mean = np.round(frames_mean/np.float(nframes)).astype(dtype)

    # release resources
    if distributed_state is None:
        ds.closeClient()

    # write result
    if write_path is not None:
        if csio.testPathExtensionForHDF5(write_path):
            csio.writeHDF5(write_path, dataset_path, frames_mean)
        else:
            csio.writeImage(write_path, frames_mean)

    # return reference to mean image
    return frames_mean

