'''mixmat implementation'''

import numpy as np
from threej import threejj
from numba import njit, prange

FOUR_PI = 4*np.pi


@njit(nogil=True, parallel=True, fastmath=True, cache=True)
def _mixmat_full(m, v, l1min, l2min, l1max, l2max, s1, s2, s1_, s2_):
    ncl = m.shape[:-2]
    for l1 in prange(l1min, l1max+1):
        for l2 in prange(l2min, l2max+1):
            buf = np.empty(l1max+l2max+1)
            buf_ = np.empty(l1max+l2max+1)
            l3min, thrcof = threejj(l1, l2, s1, -s2, out=buf)
            _, thrcof_ = threejj(l1, l2, s1_, -s2_, out=buf_)
            for i in np.ndindex(*ncl):
                t = 0.
                for x, y, z in zip(thrcof, thrcof_, v[i][int(l3min+0.01):]):
                    t += x*y*z
                m[i][l1, l2] = (2*l2+1)*t


@njit(nogil=True, parallel=True, fastmath=True, cache=True)
def _mixmat_full_same(m, v, l1min, l2min, l1max, l2max, s1, s2):
    ncl = m.shape[:-2]
    for l1 in prange(l1min, l1max+1):
        for l2 in prange(l2min, l2max+1):
            buf = np.empty(l1max+l2max+1)
            l3min, thrcof = threejj(l1, l2, s1, -s2, out=buf)
            for i in np.ndindex(*ncl):
                t = 0.
                for x, z in zip(thrcof, v[i][int(l3min+0.01):]):
                    t += x*x*z
                m[i][l1, l2] = (2*l2+1)*t


@njit(nogil=True, parallel=True, fastmath=True, cache=True)
def _mixmat_symm(m, v, lmin, lmax, s1, s2, s1_, s2_):
    ncl = m.shape[:-2]
    for l1 in prange(lmin, lmax+1):
        for l2 in prange(l1, lmax+1):
            buf = np.empty(2*lmax+1)
            buf_ = np.empty(2*lmax+1)
            l3min, thrcof = threejj(l1, l2, s1, -s2, out=buf)
            _, thrcof_ = threejj(l1, l2, s1_, -s2_, out=buf_)
            for i in np.ndindex(*ncl):
                t = 0.
                for x, y, z in zip(thrcof, thrcof_, v[i][int(l3min+0.01):]):
                    t += x*y*z
                m[i][l1, l2] = (2*l2+1)*t
                m[i][l2, l1] = (2*l1+1)*t


@njit(nogil=True, parallel=True, fastmath=True, cache=True)
def _mixmat_symm_same(m, v, lmin, lmax, s1, s2):
    ncl = m.shape[:-2]
    for l1 in prange(lmin, lmax+1):
        for l2 in prange(l1, lmax+1):
            buf = np.empty(2*lmax+1)
            l3min, thrcof = threejj(l1, l2, s1, -s2, out=buf)
            for i in np.ndindex(*ncl):
                t = 0.
                for x, z in zip(thrcof, v[i][int(l3min+0.01):]):
                    t += x*x*z
                m[i][l1, l2] = (2*l2+1)*t
                m[i][l2, l1] = (2*l1+1)*t


def mixmat(cl, l1max=None, l2max=None, l3max=None, spin=(0, 0), spin_out=None):
    r'''compute the mixing matrix for an angular power spectrum

    Computes the mixing matrix for the angular power spectrum ``cl``.  The
    number of output modes (matrix rows) and input modes (matrix columns) are
    given by ``l1max`` and ``l2max``, respectively.  The modes of ``cl`` are
    summed up to ``l3max``.

    Parameters
    ----------
    cl : array_like
        Angular power spectrum of the mixing matrix.
    l1max : int, optional
        Maximum output mode. If not given, the size of ``cl`` is used.
    l2max : int, optional
        Maximum input mode. If not given, the value of ``l1max`` is used.
    l3max : int, optional
        Maximum mode for summation. If not given, all ``cl`` values are summed.
    spin : (2,) tuple of int
        Spin weights of the input angular power spectrum which will be
        multiplied by the mixing matrix.
    spin_out : (2,) tuple of int
        Spin weights of the output angular power spectrum after the mixing
        matrix has been applied.

    Returns
    -------
    mat : (l1max+1, l2max+1) array
        Mixing matrix for given `cl2` and spin weights.

    '''

    if l1max is None:
        l1max = n-1

    if l2max is None:
        l2max = l1max

    if l3max is None or l3max > n-1:
        l3max = n-1

    if spin_out is None:
        spin_out = spin

    s1, s1_ = spin_out
    s2, s2_ = spin

    l1min, l2min = max(abs(s1), abs(s1_)), max(abs(s2), abs(s2_))

    same_spins = (s1 == s1_) and (s2 == s2_)
    symmetric = (s1 == s2) and (s1_ == s2_)

    m = np.zeros((*ncl, l1max+1, l2max+1))
    v = np.arange(1, 2*l3max+2, 2)*cl[..., :l3max+1]/FOUR_PI

    if symmetric:
        lmin, lmax = max(l1min, l2min), min(l1max, l2max)
        if same_spins:
            _mixmat_full_same(m, v, l1min, l2min, lmin-1, l2max, s1, s2)
            _mixmat_full_same(m, v, lmin, l2min, lmax, lmin-1, s1, s2)
            _mixmat_symm_same(m, v, lmin, lmax, s1, s2)
            _mixmat_full_same(m, v, lmin, lmax+1, lmax, l2max, s1, s2)
            _mixmat_full_same(m, v, lmax+1, l2min, l1max, l2max, s1, s2)
        else:
            _mixmat_full(m, v, l1min, l2min, lmin-1, l2max, s1, s2, s1_, s2_)
            _mixmat_full(m, v, lmin, l2min, lmax, lmin-1, s1, s2, s1_, s2_)
            _mixmat_symm(m, v, lmin, lmax, s1, s2, s1_, s2_)
            _mixmat_full(m, v, lmin, lmax+1, lmax, l2max, s1, s2, s1_, s2_)
            _mixmat_full(m, v, lmax+1, l2min, l1max, l2max, s1, s2, s1_, s2_)
    else:
        if same_spins:
            _mixmat_full_same(m, v, l1min, l2min, l1max, l2max, s1, s2)
        else:
            _mixmat_full(m, v, l1min, l2min, l1max, l2max, s1, s2, s1_, s2_)

    return m
