# convolvecl: convolution of angular power spectra
#
# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
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

__version__ = '2021.6.23'

__all__ = [
    'mixmat',
]


import numpy as np
from wigner import wigner_3j_l


FOUR_PI = 4*np.pi


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

    s1, S1 = spins1
    s2, S2 = spins2
    s, S = s1+s2, S1+S2

    lmin = max(abs(s), abs(S))
    l1min = max(abs(s1), abs(S1))

    same_spins = (s == S) and (s1 == S1)
    symmetric = (s2 == 0) and (S2 == 0)

    twol2p1cl2 = 2*np.arange(l2max+1, dtype=float) + 1
    twol2p1cl2 *= cl2[:l2max+1]

    m = np.zeros((lmax+1, l1max+1))

    for l in range(lmin, lmax):
        if symmetric and l <= l1max:
            l1min_ = l
        else:
            l1min_ = l1min
        for l1 in range(l1min_, l1max+1):
            if abs(l-l1) > l2max:
                continue
            l2min, l2max_, thr = wigner_3j_l(l, l1, s, -s1)
            if same_spins:
                thr *= thr
            else:
                _, _, thr2 = wigner_3j_l(l, l1, S, -S1)
                thr *= thr2
            l2min, l2max_ = int(l2min), min(int(l2max_), l2max)
            m[l, l1] = np.dot(thr[:l2max_-l2min+1], twol2p1cl2[l2min:l2max_+1])

    if symmetric:
        lblk = min(lmax, l1max)
        d = m.diagonal().copy()
        m[:lblk, :lblk] += m[:lblk, :lblk].T
        np.fill_diagonal(m, d)

    m *= (2*np.arange(l1max+1) + 1)/FOUR_PI

    return m
