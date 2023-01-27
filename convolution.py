import numpy as np

def checkShape(var, kernel):
    '''Check shapes for convolution
    Args:
        var (ndarray): 2d or 3d input array for convolution.
        kernel (ndarray): 2d or 3d convolution kernel.
    Returns:
        kernel (ndarray): 2d kernel reshape into 3d if needed.
    '''
    var_ndim = np.ndim(var)
    kernel_ndim = np.ndim(kernel)
    if var_ndim not in [2, 3]:
        raise Exception("<var> dimension should be in 2 or 3.")
    if kernel_ndim not in [2, 3]:
        raise Exception("<kernel> dimension should be in 2 or 3.")
    if var_ndim < kernel_ndim:
        raise Exception("<kernel> dimension > <var>.")
    if var_ndim == 3 and kernel_ndim == 2:
        kernel = np.repeat(kernel[:, :, None], var.shape[2], axis=2)
    return kernel

def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]
    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs

def conv3D(var, kernel, stride=1, pad=0):
    '''3D convolution by strided view.
    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        conv (ndarray): convolution result.
    '''
    kernel = checkShape(var, kernel)
    if pad > 0:
        var_pad = padArray(var, pad, pad)
    else:
        var_pad = var
    view = asStride(var_pad, kernel.shape, stride)
    if np.ndim(kernel) == 2:
        conv = np.sum(view*kernel, axis=(2, 3))
    else:
        conv = np.sum(view*kernel, axis=(2, 3, 4))
    return conv
