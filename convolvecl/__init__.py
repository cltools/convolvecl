# convolvecl: convolution of angular power spectra
#
# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''

Convolve angular power spectra (:mod:`convolvecl`)
==================================================

This is a minimal Python package for convolution of angular power spectra, in
order to compute the angular power spectrum of products of spherical functions.
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

__version__ = '2022.4.4'

__all__ = [
    'mixmat',
]

from ._mixmat import mixmat
