import numpy as np
import numpy.testing as npt
import unittest
from convolvecl import mixmat, mixmat_eb


class TestMixmat(unittest.TestCase):

    def setUp(self):
        self.lmax = 1000
        self.twolp1 = 2*np.arange(self.lmax+1) + 1
        self.cl = 1/(1+np.arange(self.lmax+1))**2

    def test_spin_00(self):
        m = mixmat(self.cl)

        npt.assert_equal(m.shape, (self.lmax+1, self.lmax+1))
        npt.assert_allclose(m[:, 0], 1/(4*np.pi)*self.cl)
        npt.assert_allclose(m[0, :], self.twolp1/(4*np.pi)*self.cl)

    def test_spin_22(self):
        m = mixmat(self.cl, spin=(2, 2))

        npt.assert_equal(m.shape, (self.lmax+1, self.lmax+1))
        npt.assert_equal(m[:, 0:2], 0)
        npt.assert_equal(m[0:2, :], 0)

    def test_spin_1m1_to_00(self):
        m = mixmat(self.cl, spin=(1, -1), spin_out=(0, 0))

        npt.assert_equal(m.shape, (self.lmax+1, self.lmax+1))
        npt.assert_equal(m[0, 0], 0)
        npt.assert_allclose(m[0, 1:], self.twolp1[1:]/(4*np.pi)*self.cl[1:])

    def test_spin_00_to_m11(self):
        m = mixmat(self.cl, spin=(0, 0), spin_out=(-1, 1))

        npt.assert_equal(m.shape, (self.lmax+1, self.lmax+1))
        npt.assert_equal(m[0, 0], 0)
        npt.assert_allclose(m[1:, 0], 1/(4*np.pi)*self.cl[1:])

    def test_symm_rect(self):
        m1 = mixmat(self.cl, l2max=2*self.lmax)
        m2 = mixmat(self.cl, l2max=self.lmax//2)

        npt.assert_equal(m1.shape, (self.lmax+1, 2*self.lmax+1))
        npt.assert_equal(m2.shape, (self.lmax+1, self.lmax//2+1))
        npt.assert_allclose(m1[:, :self.lmax//2+1], m2)

    def test_legmul(self):
        cl1 = np.random.rand(self.lmax+1)
        cl2 = np.random.rand(self.lmax+1)
        cl = np.polynomial.legendre.legmul(cl1, cl2)[:self.lmax+1]

        cl1 *= (4*np.pi)/self.twolp1
        cl2 *= (4*np.pi)/self.twolp1
        cl *= (4*np.pi)/self.twolp1

        npt.assert_allclose(mixmat(cl1)@cl2, cl)
        npt.assert_allclose(mixmat(cl2)@cl1, cl)

    def test_multi(self):
        m = mixmat([self.cl, 2*self.cl])

        npt.assert_array_equal(m.shape, (2, self.lmax+1, self.lmax+1))
        npt.assert_allclose(m[0], m[1]/2)


class TestMixmatEb(unittest.TestCase):

    def setUp(self):
        self.lmax = 1000
        self.twolp1 = 2*np.arange(self.lmax+1) + 1
        self.cl = 1/(1+np.arange(self.lmax+1))**2

    def test_spin_00(self):
        m = mixmat_eb(self.cl)

        npt.assert_equal(m.shape, (3, self.lmax+1, self.lmax+1))
        npt.assert_allclose(m[0], m[2])
        npt.assert_allclose(m[1], 0)

    def test_spin_22(self):
        m = mixmat_eb(self.cl, spin=(2, 2))

        npt.assert_equal(m.shape, (3, self.lmax+1, self.lmax+1))
        npt.assert_allclose(m[0] + m[1], m[2])


if __name__ == '__main__':
    unittest.main()
