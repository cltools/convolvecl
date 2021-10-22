import numpy as np
import numpy.testing as npt
import unittest
from convolvecl import mixmat


class TestMixingMatrix(unittest.TestCase):

    def setUp(self):
        self.lmax = 1000
        self.twolp1 = 2*np.arange(self.lmax+1) + 1
        self.cl = 1/(1+np.arange(self.lmax+1))**2

    def test_spins1_eq_0_and_spins2_eq_0(self):
        m = mixmat(self.cl, spins1=(0, 0), spins2=(0, 0))

        npt.assert_equal(m.shape, (self.lmax+1, self.lmax+1))
        npt.assert_allclose(m[:, 0], 1/(4*np.pi)*self.cl)
        npt.assert_allclose(m[0, :], self.twolp1/(4*np.pi)*self.cl)

    def test_spins1_eq_2_and_spins2_eq_0(self):
        m = mixmat(self.cl, spins1=(2, 2), spins2=(0, 0))

        npt.assert_equal(m.shape, (self.lmax+1, self.lmax+1))
        npt.assert_equal(m[:, 0:2], 0)
        npt.assert_equal(m[0:2, :], 0)

    def test_spins1_neq_0_and_spins2_eq_0(self):
        m = mixmat(self.cl, spins1=(1, -1), spins2=(-1, 1))

        npt.assert_equal(m.shape, (self.lmax+1, self.lmax+1))
        npt.assert_equal(m[0, 0], 0)
        npt.assert_allclose(m[0, 1:], self.twolp1[1:]/(4*np.pi)*self.cl[1:])

    def test_spins1_eq_0_and_spin2_neq_0(self):
        m = mixmat(self.cl, spins1=(0, 0), spins2=(-1, 1))

        npt.assert_equal(m.shape, (self.lmax+1, self.lmax+1))
        npt.assert_equal(m[0, 0], 0)
        npt.assert_allclose(m[1:, 0], 1/(4*np.pi)*self.cl[1:])


if __name__ == '__main__':
    unittest.main()
