"""
notes:
    -what is mu_n from equation 2.38
    -in step 3, the i value that has 1 - Phi(...) is 1 instead of 0

assumptions:
    -added k column to x[2] and both k and i column to x[3], thus the alpha's become part of the beta's

to do:
    -use hinge loss for regressions 2 and 3
    -use an EM algorithm for c's
        -first optimize for hinge loss, then select c's such that they minimize loss, repeat
    -find out how values of c work
    -99% of the time is computing Psi, is there a better way?
    -file IO
"""

from __future__ import print_function
import numpy as np
from scipy.misc import factorial
from scipy import stats, integrate


class const:
    L = 5
    max_visit_doctor = 2
    v, w = np.polynomial.hermite.hermgauss(L)
    sqrt2 = np.sqrt(2)
    sqrtpi = np.sqrt(np.pi)
    fact = np.array([factorial(i) for i in range(max_visit_doctor + 1)])

    # last_rho = None
    # wopts = None

""" ************* Helper Functions ********************* """


def Phi(c):
    return stats.norm.cdf(c)


def phi(x):
    return stats.norm.pdf(x)


def Psi(rho, a, b):
    """"
    from equation 1.3
    assumes rho ** 2 < 1
    """
    if np.abs(a + b) == np.inf:
        return 0

    cache1 = -(a ** 2 + b ** 2) / 2
    cache2 = a * b

    def Psi_integrand(t):
        tmp = (1 - t ** 2)
        return np.exp((t * cache2 + cache1) / tmp) / np.sqrt(tmp)

    # if rho != const.last_rho:
    #     const.last_rho = rho
    #     info_dict = integrate.quad(Psi_integrand, 0.0, rho,
    #                                full_output=110, weight="cos", wvar=0)[2]
    #     const.wopts = [info_dict["momcom"], info_dict["chebmo"]]

    # I = integrate.quad(Psi_integrand, 0, rho, epsrel=1e-1)[0] / 2 / np.pi
    # I = integrate.quad(Psi_integrand, 0, rho, weight="cos", wvar=0,
    #                    wopts=const.wopts)[0] / 2 / np.pi
    I = integrate.romberg(Psi_integrand, 0, rho, rtol=1e-2,
                          vec_func=True) / 2 / np.pi
    # I[np.isnan(I)] = 0
    return I

PsiVect = np.vectorize(Psi)

""" ************* Intermediary Step 1 ****************** """


def gauss_hermite_quadrature(func):
    """
    https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature

    takes in f and computes the integral:
        integral of f * exp(-x**2) dx from -inf to +inf
    """
    return np.dot(func(const.v), const.w)


def P_y1(k, sigma_1, mu_1):
    """
    from equation 2.24

    mu_1 = np.dot(x_1, beta_1)

    capable of taking in k, mu_1 that are vectors
    """
    outside_integral = mu_1 ** k / const.sqrtpi / const.fact[k]

    def inner_func(v):
        return np.exp(np.outer(sigma_1 * const.sqrt2 * k, v) - np.outer(mu_1, np.exp(const.sqrt2 * sigma_1 * v)))

    return gauss_hermite_quadrature(inner_func) * outside_integral

""" ** *********** Intermediary Step 2 ** **************** """


def P_y1_y2(k, i, sigma_1, rho_12, mu_1, mu_2):
    """
    from equations 2.40 (for i == 1) and 2.41 (for i == 0)

    mu_1 same as above
    mu_2 = np.dot(x_2, beta_2) + alpha_21 * k

    capable of taking in k, i, mu_1, mu_2 that are vectors
    """
    # put the proper mu_n**k here
    outside_integral = mu_1 ** k / const.sqrtpi / const.fact[k]

    def inner_func(v):
        p_y1 = np.exp(np.outer(sigma_1 * const.sqrt2 * k, v) - np.outer(
            mu_1, np.exp(const.sqrt2 * sigma_1 * v)))
        """because 1 - Phi(x) = Phi(-x), we can multiply the inner term by -1 when we need 1 - Phi(x) for better speed; this is what the (2 * i - 1) is for"""
        sign = (2 * i - 1)
        p_y2 = Phi((mu_2 + rho_12 * const.sqrt2 * v.reshape(
            (-1, 1))) / np.sqrt(1 - rho_12 ** 2) * sign)
        return p_y2.T * p_y1

    return gauss_hermite_quadrature(inner_func) * outside_integral


""" ************* Intermediary Step 3 ****************** """


def P_y3_y2_y1(k, i, j, sigma_1, rho_12, rho_13, rho_23, mu_1, mu_2, mu_3, c):
    """
    from equation 2.75 for the general equation
    and equations 2.44 (for i == 0) and 2.45 (for i == 1)

    mu_1, mu_2 same as above
    mu_3 = np.dot(x_3, beta_3) + alpha_31 * k + alpha_32 * i

    capable of taking in k, i, mu_1, mu_2 that are vectors
    """
    # which term is correct here?
    # outside_integral = 1 / const.sqrtpi
    outside_integral = mu_1 ** k / const.sqrtpi / const.fact[k]
    sign = (2 * i - 1)
    div_rho12 = 1 / np.sqrt(1 - rho_12 ** 2)
    div_rho13 = 1 / np.sqrt(1 - rho_13 ** 2)
    rho = (rho_23 - rho_12 * rho_13) * div_rho12 * div_rho13

    def inner_func(v):

        p_y1 = np.exp(np.outer(sigma_1 * const.sqrt2 * k, v) - np.outer(
            mu_1, np.exp(const.sqrt2 * sigma_1 * v)))

        y2_term = (mu_2 + rho_12 * const.sqrt2 * v.reshape(
            (-1, 1))) * div_rho12
        Phi_y2 = Phi(y2_term * sign)

        y3_term = -(mu_3 + rho_13 * const.sqrt2 * v.reshape(
            (-1, 1))) * div_rho13
        upper_term = c[j] * div_rho13 + y3_term
        lower_term = c[j - 1] * div_rho13 + y3_term
        Phi_y3 = Phi(upper_term) - Phi(lower_term)

        Psi_y3 = sign * (PsiVect(rho, y2_term, lower_term)
                         - PsiVect(rho, y2_term, upper_term))

        # print(Phi_y2.T, '\n', Phi_y3.T, '\n', Psi_y3.T, '\n', p_y1)
        return Phi_y2.T * Phi_y3.T * Psi_y3.T * p_y1

    return gauss_hermite_quadrature(inner_func) * outside_integral

if __name__ == "__main__":
    ones = np.ones(1000).astype(np.int)
    print(P_y1_y2(ones, ones, 0.1, 0.1, ones, ones)[0])
    print(P_y3_y2_y1(ones, ones, ones, 0.1, 0.1, 0.1, 0.1,
                     ones, ones, ones, np.array([0, 1, 2, 3, 4, 5]))[0])
