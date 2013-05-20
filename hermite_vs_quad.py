import numpy as np
from scipy import integrate, stats, misc
from main import phi
from itertools import izip

"""
conclusion:
    -use fast_hermite_integral (vectorized) with cached hermite-gauss roots and weights
"""


def quad_integral(mu, sigma_1, k):

    def P_y1(u_1):
        lambda_ = np.exp(mu + u_1)
        return stats.poisson.pmf(k, lambda_)

    def d_phi(u_1):
        return phi(u_1 / sigma_1) / sigma_1

    def quad_integrand(u_1):
        return P_y1(u_1) * d_phi(u_1)

    return integrate.quad(quad_integrand, -np.inf, np.inf)[0]


def hermite_coefficients(n):
    '''
    https://en.wikipedia.org/wiki/Hermite_polynomials
    '''
    acc = 2 ** n
    coeffs = [acc]
    for m in range(int(n / 2)):
        acc *= (n - 2 * m) * (n - 2 * m - 1)
        acc /= m + 1
        acc /= 4
        coeffs.append(0)
        coeffs.append(acc)
    if n % 2:
        coeffs.append(0)
    return coeffs


def hermite_roots(n):
    # not sure which to use
    return np.imag(np.roots(hermite_coefficients(n)))
    return np.abs(np.roots(hermite_coefficients(n)))


def hermite_integral(mu, sigma_1, k, L=10):
    hermgauss = np.polynomial.hermite.hermgauss(L)
    p = 0.0
    for root, weight in izip(*hermgauss):
        term = np.sqrt(2) * sigma_1 * root
        inner = k * term - mu * np.exp(term)
        p += np.exp(inner) * weight
    return p * mu ** k / np.sqrt(np.pi) / misc.factorial(k)


def fast_hermite_integral(mu, sigma_1, k, L=10):
    v, w = np.polynomial.hermite.hermgauss(L)
    term = np.sqrt(2) * sigma_1 * v
    return np.dot(w, np.exp(k * term - mu * np.exp(
        term))) * mu ** k / np.sqrt(np.pi) / misc.factorial(k)


v, w = np.polynomial.hermite.hermgauss(32)
const = np.sqrt(2) * v
sqrtpi = np.sqrt(np.pi)


def cached_hermite_integral(mu, sigma_1, k, L=32):
    return np.dot(w, np.exp(k * sigma_1 * const - mu * np.exp(const * sigma_1))) * mu ** k / sqrtpi / misc.factorial(k)

if __name__ == "__main__":
    mu = 1
    sigma = 0.1
    k = 1
    print hermite_integral(mu, sigma, k)
    print fast_hermite_integral(mu, sigma, k)
    print cached_hermite_integral(mu, sigma, k)
    print quad_integral(mu, sigma, k)
