# convolvecl: convolution of angular power spectra
#
# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
#
# cython: language_level=3, boundscheck=False, embedsignature=True
# distutils: extra_compile_args=['-fopenmp', '-Ofast']
# distutils: extra_link_args=['-fopenmp']
'''

Convolve angular power spectra (:mod:`convolvecl`)
==================================================

This is a minimal Python package for convolution of angular power spectra, in
order to compute the angular power spectrum of a product of spherical functions.
Computation is currently limited to the mixing matrix.

The package can be installed using pip::

    pip install convolvecl

Then import the :func:`~convolve.mixmat` function from the package::

    from convolvecl import mixmat

Current functionality covers the absolutely minimal use case.  Please open an
issue on GitHub if you would like to see anything added.


Reference/API
-------------

.. autosummary::
   :toctree: api
   :nosignatures:

   mixmat

'''

__all__ = [
    'mixmat',
]

import numpy as np
from libc.stdlib cimport abort, malloc, free, abs
from cython.parallel import parallel, prange
cdef extern from "wigner_3j_l.c":
    int wigner_3j_l(double l2, double l3, double m2, double m3,
                    double* l1min_out, double* l1max_out, double* thrcof,
                    int ndim) nogil


cdef double FOUR_PI = 4*np.pi


def mixmat(cl2, lmax=None, l1max=None, l2max=None, spins1=(0, 0),
           spins2=(0, 0)):
    r'''compute the mixing matrix for an angular power spectrum

    Computes the mixing matrix for a fixed power spectrum `cl2` with `lmax`
    rows and `l1max` columns, by summing up to mode `l2max` of `cl2`.  The
    matrix is constructed for pairs of spherical functions with spin weights
    `spins1` and `spins2`, respectively.

    Parameters
    ----------
    cl2 : array_like
        Angular power spectrum of the mixing matrix.
    lmax : int
        Maximum output mode of the convolution. The mixing matrix has ``lmax+1``
        rows. If ``None``, the size of `cl2` is used.
    l1max : int
        Maximum input mode of the convolution. The mixing matrix has ``l1max+1``
        columns. If ``None``, the value of `lmax` is used.
    l2max : int
        Maximum mode for `cl2` summation. If ``None``, all modes are summed.
    spins1 : (2,) tuple of int
        Spin weights for the input power spectrum ``cl1`` to be multiplied by
        the mixing matrix.
    spins2 : (2,) tuple of int
        Spin weights for the mixing matrix power spectrum `cl2`.

    Returns
    -------
    mat : (lmax+1, l1max+1) array
        Mixing matrix for given `cl2` and spin weights.

    Notes
    -----
    Let :math:`h=fg` and :math:`h'=f'g'` be two not necessarily distinct
    products of spherical functions.  Let further :math:`f` and :math:`f'` have
    angular power spectrum :math:`C_1(l)`, let :math:`g` and :math:`g'` have
    angular power spectrum :math:`C_2(l)`, and let both :math:`f`, :math:`f'` be
    independent of :math:`g`, :math:`g'`.  The resulting angular power spectrum
    :math:`C(l)` of :math:`h` and :math:`h'` is then given by a convolution of
    :math:`C_1(l)` and :math:`C_2(l)`.

    For a fixed angular power spectrum :math:`C_2`, the convolution can be
    written as a matrix multiplication

    .. math::

        C(l) = \sum_{l_1} M_{ll_1}(C_2) \, C_1(l_1) \,

    where :math:`M_{ll_1}(C_2)` is the mixing matrix.  Besides :math:`C_2`, the
    mixing matrix further depends on the spin weights :math:`\mathtt{spins1} =
    (s_f, s_{f'})` and :math:`\mathtt{spins2} = (s_g, s_{g'})` of the pairs of
    spherical functions :math:`f, f'` and :math:`g, g'`.

    '''

    if lmax is None:
        lmax = len(cl2)-1

    if l1max is None:
        l1max = lmax

    if l2max is None or l2max > len(cl2)-1:
        l2max = len(cl2)-1

    cdef int lmax_ = lmax
    cdef int l1max_ = l1max
    cdef int l2max_ = l2max

    cdef int s1 = spins1[0]
    cdef int S1 = spins1[1]
    cdef int s2 = spins2[0]
    cdef int S2 = spins2[1]
    cdef int s = s1+s2
    cdef int S = S1+S2

    cdef int lmin = max(abs(s), abs(S))
    cdef int l1min = max(abs(s1), abs(S1))

    cdef int same_spins = (s == S) and (s1 == S1)
    cdef int symmetric = (s2 == 0) and (S2 == 0)

    # precompute (2l + 1) cl2
    twol2p1cl2 = 2*np.arange(l2max+1, dtype=float) + 1
    twol2p1cl2 *= cl2[:l2max+1]

    # output matrix
    m = np.zeros((lmax+1, l1max+1))

    # pure C interface
    cdef int l2min_int, l2max_int
    cdef int ndim = lmax+l1max+1
    cdef double* l2minmax
    cdef double* thr
    cdef double* thr2
    cdef double* thr_
    cdef double[::1] twol2p1cl2_ = twol2p1cl2
    cdef double[::, ::1] m_ = m
    cdef double mll1
    cdef int n

    cdef int l, l1, l1min_, l2, i
    with nogil, parallel():
        # set up local buffers
        l2minmax = <double*>malloc(2*sizeof(double))
        thr = <double*>malloc(ndim*sizeof(double))
        thr2 = <double*>malloc(ndim*sizeof(double))
        if not l2minmax or not thr or not thr2:
            abort()

        for l in prange(lmin, lmax_+1, schedule='dynamic'):
            if symmetric and l <= l1max_:
                l1min_ = l
            else:
                l1min_ = l1min
            for l1 in range(l1min_, l1max_+1):
                if abs(l-l1) > l2max_:
                    continue
                wigner_3j_l(l, l1, s, -s1, &l2minmax[0], &l2minmax[1], thr, ndim)
                l2min_int = int(l2minmax[0])
                l2max_int = int(l2minmax[1]) if l2minmax[1] < l2max_ else l2max_
                n = l2max_int - l2min_int + 1
                if same_spins:
                    thr_ = thr
                else:
                    wigner_3j_l(l, l1, S, -S1, &l2minmax[0], &l2minmax[1], thr2, ndim)
                    thr_ = thr2
                mll1 = 0
                for i in range(n):
                    mll1 = mll1 + twol2p1cl2_[l2min_int+i]*thr[i]*thr_[i]
                m_[l, l1] = (2*l1 + 1)/FOUR_PI*mll1
                if symmetric and l1 <= lmax_ and l <= l1max_:
                    m_[l1, l] = (2*l + 1)/FOUR_PI*mll1

        # free local buffers
        free(l2minmax)
        free(thr)
        free(thr2)

    return m
