import numpy as np
import chainer
import chainer.functions as F


def compute_EI(f_min, mean, std, jitter=0.01):
    """Computes the Expected Improvement
    
    :param f_min: min f value currently
    :param mean: mean f value of current function at x point
    :param std: std of f value of current function at x point
    :param jitter: positive value to make the acquisition more explorative.
    """
    if isinstance(std, np.ndarray):
        std[std<1e-10] = 1e-10
    elif isinstance(std, float) and std < 1e-10:
        std = 1e-10
    else:
        raise ValueError("Invalid std type")
    assert jitter >= 0

    xp = chainer.backends.cuda.get_array_module(mean)
    u = (f_min - mean - jitter)
    norm_pdf_value = xp.exp(-0.5 * u**2) / xp.sqrt(2*xp.pi)
    norm_cdf_value = 0.5 * F.erfc(-u / xp.sqrt(2))
    f_acqu = std * (u * norm_cdf_value + norm_pdf_value)
    return f_acqu


def compute_EI_with_grad(f_min, mean, std, dmean, dstd, jitter=0.01):
    """Computes the Expected Improvement and its derivative
    
    :param f_min: min f value currently
    :param mean: mean f value of current function at x point
    :param std: std of f value of current function at x point
    :param jitter: positive value to make the acquisition more explorative.
    """
    if isinstance(std, np.ndarray):
        std[std<1e-10] = 1e-10
    elif isinstance(std, float) and std < 1e-10:
        std = 1e-10
    else:
        raise ValueError("Invalid std type")
    assert jitter >= 0

    xp = chainer.backends.cuda.get_array_module(mean)
    u = (f_min - mean - jitter)
    norm_pdf_value = xp.exp(-0.5 * u**2) / xp.sqrt(2*xp.pi)
    norm_cdf_value = 0.5 * F.erfc(-u / xp.sqrt(2))
    f_acqu = std * (u * norm_cdf_value + norm_pdf_value)
    df_acqu = dstd * norm_pdf_value - norm_cdf_value * dmean
    return f_acqu, df_acqu
