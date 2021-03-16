import numpy as np

def window(image:np.array, lower:float, upper:float)->np.array:
    """Normalize the image by a window. The image is clipped to the lower and
    upper value and then scaled to a range between -1 and 1.

    Parameters
    ----------
    image : np.array
        The image as numpy array
    lower : float
        The lower value to clip at
    upper : float
        The higher value to clip at

    Returns
    -------
    np.array
        The normalized image
    """
    # clip
    image = np.clip(image, a_min=lower, a_max=upper)
    # rescale to between 0 and 1
    image = (image - lower) / (upper - lower)
    # rescale to between -1 and 1
    image = (image * 2) - 1

    return image

def quantile(image:np.array, lower_q:float, upper_q:float)->np.array:
    """Normalize the image by quantiles. The image is clipped to the lower and
    upper quantiles of the image and then scaled to a range between -1 and 1.

    Parameters
    ----------
    image : np.array
        The image as numpy array
    lower_q : float
        The lower quantile (between 0 and 1)
    upper_q : float
        The upper quantile (between 0 and 1)

    Returns
    -------
    np.array
        The normalized image
    """
    assert upper_q > lower_q, 'Upper quantile has to be larger than the lower.'
    assert np.sum(np.isnan(image)) == 0, f'There are {np.sum(np.isnan(image)):.2f} NaNs in the image.'

    a_min = np.quantile(image, lower_q)
    a_max = np.quantile(image, upper_q)

    assert a_max > a_min, 'Both quantiles are the same.'

    return window(image, a_min, a_max)

def mean_std(image:np.array)->np.array:
    """Subtract the mean from image and the divide it by the standard deviation
    generating a value similar to the Z-Score.

    Parameters
    ----------
    image : np.array
        The image to normalize

    Returns
    -------
    np.array
        The normalized image
    """
    image = image - np.mean(image)
    std = np.std(image)
    assert std > 0, 'The standard deviation of the image is 0.'
    image = image / std

    return image