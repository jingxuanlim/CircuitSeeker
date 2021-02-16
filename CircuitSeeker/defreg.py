import numpy as np
import os

import CircuitSeeker.fileio as csio
import CircuitSeeker.distributed as csd
import dask.array as da

import SimpleITK as sitk

# TODO: TEMP
import sys
from numpy.random import normal


def skipSample(
    fixed, moving,
    fixed_vox, moving_vox,
    target_spacing
    ):
    """
    """

    # determine skip sample factors
    fss = np.maximum(np.round(target_spacing / fixed_vox), 1).astype(np.int)
    mss = np.maximum(np.round(target_spacing / moving_vox), 1).astype(np.int)

    # skip sample the images
    fixed = fixed[::fss[0], ::fss[1], ::fss[2]]
    moving = moving[::mss[0], ::mss[1], ::mss[2]]
    fixed_vox = fixed_vox * fss
    moving_vox = moving_vox * mss
    return fixed, moving, fixed_vox, moving_vox


def numpyToSITK(
    fixed, moving,
    fixed_vox, moving_vox,
    ):
    """
    """

    fixed = sitk.GetImageFromArray(fixed.copy().astype(np.float32))
    moving = sitk.GetImageFromArray(moving.copy().astype(np.float32))
    fixed.SetSpacing(fixed_vox[::-1])
    moving.SetSpacing(moving_vox[::-1])
    return fixed, moving


def affineTransformToMatrix(transform):
    """
    """

    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape((3,3))
    matrix[:3, -1] = np.array(transform.GetTranslation())
    return matrix


def matrixToAffineTransform(matrix):
    """
    """

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix[:3, :3].flatten())
    transform.SetTranslation(matrix[:3, -1].squeeze())
    return transform


def matchHistograms(fixed, moving, bins=1024):
    """
    """

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(bins)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving, fixed)


def speckle(image, scale=0.001):
    """
    """

    mn, mx = np.percentile(image, [1, 99])
    stddev = (mx - mn) * scale
    return image + normal(scale=stddev, size=image.shape)


def getLinearRegistrationModel(
    fixed_vox,
    learning_rate,
    iterations,
    number_of_histogram_bins,
    metric_sampling_percentage,
    shrink_factors,
    smooth_sigmas,
    ):
    """
    """

    # set up registration object
    # ncores = int(os.environ["LSB_DJOB_NUMPROC"]) * 2  # LSF specific!
    ncores = len(os.sched_getaffinity(0))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(ncores)
    irm = sitk.ImageRegistrationMethod()
    irm.SetNumberOfThreads(ncores)
    irm.SetInterpolator(sitk.sitkLinear)

    # metric
    irm.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=number_of_histogram_bins,
    )
    irm.SetMetricSamplingStrategy(irm.RANDOM)
    irm.SetMetricSamplingPercentage(metric_sampling_percentage)

    # optimizer
    irm.SetOptimizerAsGradientDescent(
        numberOfIterations=iterations,
        learningRate=learning_rate,
    )
    irm.SetOptimizerScalesFromPhysicalShift()

    # pyramid
    irm.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    irm.SetSmoothingSigmasPerLevel(smoothingSigmas=smooth_sigmas)
    irm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # callback
    def callback(irm):
        level = irm.GetCurrentLevel()
        iteration = irm.GetOptimizerIteration()
        metric = irm.GetMetricValue()
        print("LEVEL: ", level, " ITERATION: ", iteration, " METRIC: ", metric)
    irm.AddCommand(sitk.sitkIterationEvent, lambda: callback(irm))
    return irm


def getDeformableRegistrationModel(
    fixed_vox,
    learning_rate,
    iterations,
    shrink_factors,
    smooth_sigmas,
    ncc_radius,
        callback_savename=''
    ):
    """
    """

    # set up registration object
    # ncores = int(os.environ["LSB_DJOB_NUMPROC"]) * 2 # LSF specific!
    ncores = len(os.sched_getaffinity(0))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(ncores)
    irm = sitk.ImageRegistrationMethod()
    irm.SetNumberOfThreads(ncores)
    irm.SetInterpolator(sitk.sitkLinear)

    # metric
    irm.SetMetricAsANTSNeighborhoodCorrelation(ncc_radius)  ## might take a list
    irm.MetricUseFixedImageGradientFilterOff()

    # optimizer
    max_step = np.min(fixed_vox)
    irm.SetOptimizerAsGradientDescent(
        numberOfIterations=iterations,
        learningRate=learning_rate,
        maximumStepSizeInPhysicalUnits=max_step,
    )
    irm.SetOptimizerScalesFromPhysicalShift()

#    irm.SetOptimizerAsLBFGS2(
#        numberOfIterations=iterations,
#        lineSearchMinimumStep=1e-5,
#        lineSearchMaximumStep=1e5,
#        lineSearchMaximumEvaluations=10,
#    )
##    irm.SetOptimizerScalesFromPhysicalShift()
 
    # pyramid
    irm.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    irm.SetSmoothingSigmasPerLevel(smoothingSigmas=smooth_sigmas)
    irm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # callback
    def callback(irm):
        
        level = irm.GetCurrentLevel()
        iteration = irm.GetOptimizerIteration()
        metric = irm.GetMetricValue()
        
        print("LEVEL: ", level, " ITERATION: ", iteration, " METRIC: ", metric)

        if callback_savename:
            with open(callback_savename, 'a') as fp:
                fp.write(f'LEVEL: {level} ITERATION: {iteration} METRIC: {metric}')
        
    irm.AddCommand(sitk.sitkIterationEvent, lambda: callback(irm))
    return irm


def rigidAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    number_of_histogram_bins=128,
    metric_sampling_percentage=0.25,
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    learning_rate=1.0,
    number_of_iterations=250,
    target_spacing=2.0):
    """
    """

    # skip sample
    if target_spacing is not None:
        fixed, moving, fixed_vox, moving_vox = skipSample(
            fixed, moving, fixed_vox, moving_vox, target_spacing
        )

    # convert to sitk images, set spacing
    fixed, moving = numpyToSITK(fixed, moving, fixed_vox, moving_vox)

    # set up registration object, initialize
    irm = getLinearRegistrationModel(
        fixed_vox,
        learning_rate,
        number_of_iterations,
        number_of_histogram_bins,
        metric_sampling_percentage,
        shrink_factors,
        smooth_sigmas,
    )
    irm.SetInitialTransform(sitk.Euler3DTransform())

    # execute, return as ndarray
    transform = irm.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )
    etransform = sitk.Euler3DTransform()
    etransform.SetParameters(transform.GetParameters())

    del fixed, moving, irm, transform
    
    return affineTransformToMatrix(etransform)


def affineAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    rigid_matrix=None,
    number_of_histogram_bins=128,
    metric_sampling_percentage=0.25,
    shrink_factors=[2,1],
    smooth_sigmas=[1,0],
    learning_rate=1.0,
    number_of_iterations=250,
    target_spacing=2.0):
    """
    """

    # skip sample
    if target_spacing is not None:
        fixed, moving, fixed_vox, moving_vox = skipSample(
            fixed, moving, fixed_vox, moving_vox, target_spacing
        )

    # convert to sitk images, set spacing
    fixed, moving = numpyToSITK(fixed, moving, fixed_vox, moving_vox)
    rigid = matrixToAffineTransform(rigid_matrix)

    # set up registration object
    irm = getLinearRegistrationModel(
        fixed_vox,
        learning_rate,
        number_of_iterations,
        number_of_histogram_bins,
        metric_sampling_percentage,
        shrink_factors,
        smooth_sigmas,
    )

    # initialize
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(rigid.GetMatrix())
    affine.SetTranslation(rigid.GetTranslation())
    affine.SetCenter(rigid.GetCenter())
    irm.SetInitialTransform(affine)

    # execute, return as ndarray
    transform = irm.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                            sitk.Cast(moving, sitk.sitkFloat32),
    )
    atransform = sitk.AffineTransform(3)
    atransform.SetParameters(transform.GetParameters())

    del fixed, moving, rigid, irm, affine, transform
    return affineTransformToMatrix(atransform)


def denseDeformableAlign(
        fixed, moving,
        fixed_vox=None,
        moving_vox=None,
        affine_matrix=None,
        ncc_radius=8,
        gradient_smoothing=1.0,
        field_smoothing=0.5,
        shrink_factors=[2,],
        smooth_sigmas=[2,],
        learning_rate=1.0,
        number_of_iterations=250,
        callback_savename=''
    ):
    """
    """

    # convert to sitk images, set spacing
    fixed, moving = numpyToSITK(fixed, moving, fixed_vox, moving_vox)
    affine = matrixToAffineTransform(affine_matrix)

    # set up registration object (image registration method)
    irm = getDeformableRegistrationModel(
        fixed_vox,
        learning_rate,
        number_of_iterations,
        shrink_factors,
        smooth_sigmas,
        ncc_radius,
        callback_savename=callback_savename
    )

    # initialize
    df = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
    df.CopyInformation(fixed)
    dft = sitk.DisplacementFieldTransform(df)

    del df

    dft.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=gradient_smoothing,
        varianceForTotalField=field_smoothing,
    )
    
    irm.SetMovingInitialTransform(affine)
    irm.SetInitialTransform(dft, inPlace=True)

    # execute
    deformation = irm.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                              sitk.Cast(moving, sitk.sitkFloat32),
   )

    del dft
    
    # convert transform to displacement field
    tdff = sitk.TransformToDisplacementFieldFilter()
    tdff.SetReferenceImage(fixed)    
    tdff.SetOutputPixelType(sitk.sitkVectorFloat32)

    del fixed, moving, affine, irm
    
    return sitk.GetArrayFromImage(tdff.Execute(deformation))    

    
    
def deformableAlign(
    fixed, moving,
    fixed_vox=None,
    moving_vox=None,
    affine_matrix=None,
    ncc_radius=8,
    gradient_smoothing=1.0,
    field_smoothing=0.5,
    shrink_factors=[2,],
    smooth_sigmas=[2,],
    learning_rate=1.0,
    number_of_iterations=250,
    ):
    """
    """

    # convert to sitk images, set spacing
    fixed, moving = numpyToSITK(fixed, moving, fixed_vox, moving_vox)
    affine = matrixToAffineTransform(affine_matrix)

    # set up registration object (image registration method)
    irm = getDeformableRegistrationModel(
        fixed_vox,
        learning_rate,
        number_of_iterations,
        shrink_factors,
        smooth_sigmas,
        ncc_radius,
    )

    # initialize
    tdff = sitk.TransformToDisplacementFieldFilter()
    tdff.SetReferenceImage(fixed)

    # no problem with applying sitk.GetArrayFromImage(df) to fixed image
    # comes out exactly like applying the coordinate transform directly
    df = tdff.Execute(affine)  ## [image] generate a displacement field from a coordinate transform

    # no problem with also applying resampler.SetTrasform(dft) to fixed image
    # comes out exactly like applying the coordinate transform directly
    dft = sitk.DisplacementFieldTransform(df) ## consume an image to construct a displacement field transform
    dft.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=gradient_smoothing,
        varianceForTotalField=field_smoothing,
    )
    ## perform the registration in-place so that the dft, the initial transform, is modified
    irm.SetInitialTransform(dft, inPlace=True)

#    splines = sitk.BSplineTransformInitializer(
#        image1=fixed,
#        transformDomainMeshSize=[2, 2, 2],
#        order=3,
#    )
#    irm.SetInitialTransformAsBSpline(
#        splines,
#        inPlace=False,
#        scaleFactors=[1,],
#    )

    # execute
    deformation = irm.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                              sitk.Cast(moving, sitk.sitkFloat32),
   )
    ## when applied resampler.SetTrasform(deformation), it already
    # gives a weird result

    # convert to displacement vector field and return as ndarray
    tdff.SetOutputPixelType(sitk.sitkVectorFloat32)
    return sitk.GetArrayFromImage(tdff.Execute(deformation))


def distributedDeformableAlign(
    fixed, moving,
    fixed_vox, moving_vox,
    affine_matrix,
    block_size=[112,112,112],
    overlap=8, 
    distributed_state=None,
    **kwargs
    ):
    """
    """

    # reasmple moving image with affine
    moving_res = applyTransformToImage(
        fixed, moving,
        fixed_vox, moving_vox,
        matrix=affine_matrix
    )

    # set up the distributed environment
    ds = distributed_state
    if distributed_state is None:
        ds = csd.distributedState()
        # TODO: expose cores/tpw, remove job_extra -P
        ds.initializeLSFCluster(
            job_extra=["-P scicompsoft"],
            ncpus=1,
            cores=1,
            threads_per_worker=2,
            memory="15GB",
            mem=15000,
            walltime="48:00",
        )
        ds.initializeClient()
        nchunks = np.ceil(np.array(fixed.shape)/block_size)
        ds.scaleCluster(njobs=np.prod(nchunks))

    # TODO: refactor into a function, generalize w.r.t. dimension, share on github
    # chunk ndarrays onto workers and stack as single dask array
    bs = block_size  # shorthand
    fixed_blocks = [[
        [da.from_array(fixed[i:i+bs[0], j:j+bs[1], k:k+bs[2]])
        for k in range(0, fixed.shape[2], bs[2])]
        for j in range(0, fixed.shape[1], bs[1])]
        for i in range(0, fixed.shape[0], bs[0])]
    fixed_da = da.block(fixed_blocks)
    moving_blocks = [[
        [da.from_array(moving_res[i:i+bs[0], j:j+bs[1], k:k+bs[2]])
        for k in range(0, moving_res.shape[2], bs[2])]
        for j in range(0, moving_res.shape[1], bs[1])]
        for i in range(0, moving_res.shape[0], bs[0])]
    moving_da = da.block(moving_blocks)

    # deform all chunks
    compute_blocks = [x + 2*overlap for x in block_size] + [3,]

    from functools import partial
    deformable_align = partial(deformableAlign,
                               fixed_vox=fixed_vox,
                               moving_vox=moving_vox,
                               affine_matrix=np.eye(4))

    deformation = da.map_overlap(deformable_align,
                                 fixed_da, moving_da,
                                 depth=overlap,
                                 dtype=np.float32,
                                 chunks=compute_blocks,
                                 new_axis=[3,],
                                 **kwargs
    ).compute()

    # release resources
    if distributed_state is None:
        ds.closeClient()

    # TODO: TEMP
    resampled = applyTransformToImage(
        fixed, moving_res, fixed_vox, fixed_vox, displacement=deformation
    )

    return deformation, resampled


def applyTransformToImage(
    fixed, moving,
    fixed_vox, moving_vox,
    matrix=None, displacement=None,
    ):
    """
    """

    # convert to sitk objects
    dtype = fixed.dtype
    fixed, moving = numpyToSITK(fixed, moving, fixed_vox, moving_vox)
    
    if not [x for x in (matrix, displacement) if x is None]:
        transform = sitk.CompositeTransform([matrixToAffineTransform(matrix),
                                             displacementToTransform(displacement, fixed_vox)])
    elif matrix is not None:
        transform = matrixToAffineTransform(matrix)
    elif displacement is not None:
        transform = displacementToTransform(displacement, fixed_vox)

    # set up resampler object
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.Cast(fixed, sitk.sitkFloat32))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    # execute, return as numpy array
    resampled = resampler.Execute(sitk.Cast(moving, sitk.sitkFloat32))

    del fixed, moving, transform, resampler
    
    return sitk.GetArrayFromImage(resampled).astype(dtype)


def applyMultipleTransformToImage(vol, transformation_dict, verbose=False):

    # apply transforms
    for i, transform in enumerate(transformation_dict):
        
        label = transform['label']
        xfm = transform['xfm']  ## either a matrix or a deformation field
        fixed = transform['fixed']
        fixed_vox = transform['fixed_vox']
        moving_vox = transform['moving_vox']

        ## figure out type of transformation
        if isinstance(xfm, list):
            
            if verbose: print(f'Applying {label} composite deformation field...')
            vol = applyTransformToImage(fixed, vol, fixed_vox, moving_vox, matrix=xfm[0], displacement=xfm[1])            
            
        
        else: 
            if xfm.shape == (4,4):  # must be a affine matrix

                if verbose: print(f'Applying {label} affine transformation...')
                vol = applyTransformToImage(fixed, vol, fixed_vox, moving_vox, matrix=xfm)

            else:  # must be a deformation field

                if verbose: print(f'Applying {label} deformation field...')
                vol = applyTransformToImage(fixed, vol, fixed_vox, moving_vox, displacement=xfm)

    return vol
        


def displacementToTransform(displacement, fixed_vox):

    displacement = displacement.astype(np.float64)
    transform = sitk.GetImageFromArray(displacement, isVector=True)
    transform.SetSpacing(fixed_vox[::-1])
    transform = sitk.DisplacementFieldTransform(transform)

    return transform
