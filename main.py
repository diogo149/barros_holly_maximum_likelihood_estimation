"""
notes:
    -

assumptions:
    -u_i^0 is a scalar for i = 1, 2, 3
    -beta_i is a vector for i = 1, 2, 3
    -alpha_i is a scalar for i = 21, 31, 32
    -sigma_1 is a scalar
    -rho_i is a scalar for i = 12, 13, 23
    -betas and alphas are input parameters (this is definitely not the case and must be fixed)

to do:
    -
"""

from __future__ import print_function
import numpy as np
from scipy import stats
from scipy import integrate


def Phi(c):
    return stats.norm.cdf(c)


def phi(x):
    return stats.norm.pdf(x)


def Psi(rho, a, b):
    """"assumes rho < 1
    """

    '''for infinite integrals'''
    if abs(a + b) == np.inf:
        return 0

    def Psi_integrand(t):
        tmp = (1 - t ** 2)
        return np.exp(-(a ** 2 + b ** 2 - 2 * t * a * b) / (2 * tmp)) / np.sqrt(tmp)

    return integrate.quad(Psi_integrand, 0, rho)[0] / 2 / np.pi


class Likelihood(object):

    def __init__(self, data, beta_1, beta_2, beta_3, alpha_21, alpha_31, alpha_32, sigma_1, rho_12, rho_13, rho_23):
        self.data = data
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.alpha_21 = alpha_21
        self.alpha_31 = alpha_31
        self.alpha_32 = alpha_32
        self.sigma_1 = sigma_1
        self.rho_12 = rho_12
        self.rho_13 = rho_13
        self.rho_23 = rho_23

    def L(self):
        # put a summation over all data points here
        for n in range(1):
            k = i = j = 1
            # make sure to extract k, i, j from the data point here
            return np.log(self.prob(k, i, j))

    def prob(self, k, i, j):

        def prob_integrand(u_1):
            u_2, u_3 = self.generate_u(u_1)
            # generate alpha & beta
            # potentially pass in alpha & beta to functions
            left = self.P_y2_y3(i, j, k, u_1)
            middle = self.P_y1(k, u_1)
            right = self.d_phi(u_1)
            return left * middle * right

        return integrate.quad(prob_integrand, -np.inf, np.inf)

    def generate_u(self, u_1):
        cov = self.u_cov()
        mean = self.u_mean() * u_1
        return np.random.multivariate_normal(mean, cov)

    def u_mean(self):
        p12 = self.rho_12
        p13 = self.rho_13
        s1 = self.sigma_1
        return np.array([p12, p13]) / s1

    def u_cov(self):
        p12 = self.rho_12
        p13 = self.rho_13
        p23 = self.rho_23
        sigma = [[1 - p12 ** 2, p23 - p12 * p13], [
            p23 - p12 * p13, 1 - p13 ** 2]]
        return np.array(sigma)

    def d_phi(self, u_1):
        s1 = self.sigma_1
        return phi(u_1 / s1) / s1

    def P_y2_y3(self, i, j, k, u_1):
        pass

    def P_y1(self, k, u_1):
        pass


def main():
    z = Psi(0.9, 0.3, 0.5)
    print(z)

if __name__ == "__main__":
    main()
