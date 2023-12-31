import os

import yaml

from scipy import signal

import jax.numpy as jnp
from jax import jit

import numpy as np

with open(
    os.path.join(os.path.dirname(__file__), '..', 'configs', 'filter_polynomials.yaml'),
    "r",
    encoding="utf-8",
) as f:
    FILTER_POLY = yaml.safe_load(f)


def bw_filter(fc: int, fps: int, sig: jnp.ndarray):
    """
    Implements 2nd order low pass butterworth filter

    Inputs:
        - fc: cuttoff frequency of filter
        - fps: sampling rate of signal in frames per second
        - sig: signal to be filtered

    Returns:
        - processed signal
    """
    w = fc / (fps / 2)
    rounded_value = jnp.round(w,3)
    b = jnp.array(FILTER_POLY[f"fps_300_w_{str(round(float(rounded_value), 3))}"]["b"])
    a = jnp.array(FILTER_POLY[f"fps_300_w_{str(round(float(rounded_value), 3))}"]["a"])
    return signal.filtfilt(b, a, sig)



def find_opt_fc(sig: jnp.ndarray, fps: int, l: int = 20, acc: int = 1):
    """
    Performs residual analysis to find optimal cutoff frequency for lpf

    Inputs:
        - sig: the signal being filtered (x,y,z coord)
        - fps: sampling rate of the signal
        - l: max cutoff frequency tried
        - acc: min cutoff frequency tried (and step)

    Returns:
        - f_cutoff: the optimal cutoff frequency for that given signal to be used in a BW Filter
    """
    # calculate residual curve
    fc_arr = jnp.arange(acc, l, 1)
    xsig = jnp.array([bw_filter(fc, fps, sig) for fc in fc_arr])
    rmse = jnp.sqrt(jnp.mean((sig[None, :] - xsig) ** 2, axis=1))

    # find where curve levels off
    dif = jnp.diff(rmse)
    tail_c = jnp.argmax(dif > -0.01)

    # fit line to tail
    tail_val = rmse[tail_c:].astype(jnp.float32)
    tail_t = fc_arr[tail_c:].astype(jnp.float32)

    theta = jnp.polyfit(tail_t, tail_val, 1)

    # find where line intersecpts curve
    idx = jnp.argmax(rmse <= theta[1])
    f_cutoff = fc_arr[idx]

    # return optimal cutoff
    return (xsig[idx], f_cutoff)


def initialize_params(x,y):

    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)

    n_data = len(x)

    # Difference vectors
    h = jnp.diff(x)  # x[i+1] - x[i] for i=0,...,n-1
    p = jnp.diff(y)  # y[i+1] - y[i]

    zero = jnp.array([0.0])
    one = jnp.array([1.0])
    A00 = jnp.array([h[1]])
    A01 = jnp.array([-(h[0] + h[1])])
    A02 = jnp.array([h[0]])
    ANN = jnp.array([h[-2]])
    AN1 = jnp.array([-(h[-2] + h[-1])])  # A[N, N-1]
    AN2 = jnp.array([h[-1]])  # A[N, N-2]

    # Construct the tri-diagonal matrix A
    A = jnp.diag(jnp.concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
    upper_diag1 = jnp.diag(jnp.concatenate((A01, h[1:])), k=1)
    upper_diag2 = jnp.diag(jnp.concatenate((A02, jnp.zeros(n_data - 3))), k=2)
    lower_diag1 = jnp.diag(jnp.concatenate((h[:-1], AN1)), k=-1)
    lower_diag2 = jnp.diag(jnp.concatenate((jnp.zeros(n_data - 3), AN2)), k=-2)
    A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2

    # Construct RHS vector s
    center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
    s = jnp.concatenate((zero, center, zero))
    # Compute spline coefficients by solving the system
    coefficients = jnp.linalg.solve(A, s)

    return x,y,coefficients

def compute_coefficients(x,y,coefficients,xs):

    knots = x

    # Determine the interval that x lies in
    ind = jnp.digitize(xs, knots) - 1
    # Include the right endpoint in spline piece C[m-1]
    ind = jnp.clip(ind, 0, len(knots) - 2)
    t = xs - knots[ind]
    h = jnp.diff(knots)[ind]

    c = coefficients[ind]
    c1 = coefficients[ind + 1]
    a = y[ind]
    a1 = y[ind + 1]
    b = (a1 - a) / h - (2 * c + c1) * h / 3.0
    d = (c1 - c) / (3 * h)
    result = (t, a, b, c, d)

    return result

@jit
def evaluate_spline(x,y,xs):
    """Evaluation of the spline.

    Notes
    -----
    Values are extrapolated if x is outside of the original domain
    of knots. If x is less than the left-most knot, the spline piece
    f[0] is used for the evaluation; similarly for x beyond the
    right-most point.

    """
    x,y,coefficients = initialize_params(x,y)
    t, a, b, c, d = compute_coefficients(x,y,coefficients,xs)
    result = a + b * t + c * t**2 + d * t**3

    return result
