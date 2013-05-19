"""
notes:
    -

assumptions:
    -u_i^0 is a scalar for i = 1, 2, 3
    -beta_i is a vector for i = 1, 2, 3
    -alpha_i is a scalar for i = 21, 31, 32
    -alpha_i is a scalar for i = 1, 2, 3, 4 (at 0 and 5, alpha_i = infinity)
    -sigma_1 is a scalar
    -rho_i is a scalar for i = 12, 13, 23
    -betas and alphas are input parameters (this is definitely not the case and must be fixed)

to do:
    -find out how to generate beta and alpha
    -perform optimization on parameters
    -test for infinity's in P_y2_y3 to give result without computation
    -read file

"""

from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats
from scipy import integrate


def Phi(c):
    return stats.norm.cdf(c)


def phi(x):
    return stats.norm.pdf(x)


def Psi(rho, a, b):
    """"
    from equation 1.3
    assumes rho ** 2 < 1
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
        self.sigma_1 = sigma_1
        self.rho_12 = rho_12
        self.rho_13 = rho_13
        self.rho_23 = rho_23

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.alpha_21 = alpha_21
        self.alpha_31 = alpha_31
        self.alpha_32 = alpha_32
        self.alphas = [-np.inf, -2, -1, 1, 2, np.inf]

    def L(self):
        """
        from equation 2.26
        """
        # put a summation over all data points here
        likelihood = 0.0
        for i, j, k, x_1, x_2, x_3 in self.data:
            x = [x_1, x_2, x_3]
            likelihood += np.log(self.prob(k, i, j, x))
        return likelihood

    def prob(self, k, i, j, x):
        """
        from equation 2.22
        """

        def prob_integrand(u_1):
            """
            integrand from equation 2.22
            """
            u_2, u_3 = self.generate_u(u_1)
            # generate alpha & beta
            # potentially pass in alpha & beta to functions
            left = self.P_y2_y3(i, j, k, u_1)
            middle = self.P_y1(k, u_1, x[0])
            right = self.d_phi(u_1)
            return left * middle * right

        return integrate.quad(prob_integrand, -np.inf, np.inf)

    def generate_u(self, u_1):
        """
        from equation 2.23
        """
        cov = self.u_cov()
        mean = self.u_mean() * u_1
        return np.random.multivariate_normal(mean, cov)

    def u_mean(self):
        """
        first argument from equation 2.23
        """
        p12 = self.rho_12
        p13 = self.rho_13
        s1 = self.sigma_1
        return np.array([p12, p13]) / s1

    def u_cov(self):
        """
        second argument from equation 2.23
        """
        p12 = self.rho_12
        p13 = self.rho_13
        p23 = self.rho_23
        sigma = [[1 - p12 ** 2, p23 - p12 * p13], [
            p23 - p12 * p13, 1 - p13 ** 2]]
        return np.array(sigma)

    def d_phi(self, u_1):
        """
        from equation after equation 2.22
        """
        s1 = self.sigma_1
        return phi(u_1 / s1) / s1

    def P_y2_y3(self, i, j, k, u_1, x):
        """
        from equations 2.24 and 2.25
        """
        x2 = x[1]
        x3 = x[2]
        b2 = self.beta_2
        b3 = self.beta_3
        p12 = self.rho_12
        p13 = self.rho_13
        p23 = self.rho_23
        a21 = self.alpha_21
        a31 = self.alpha_31
        a = self.alphas

        '''caching computation'''
        x2b2 = np.dot(x2, b2)
        x3b3 = np.dot(x3, b3)
        a21k = a21 * k
        a31k = a31 * k
        u2mean, u3mean = self.u_mean()
        div_p13 = np.sqrt(1 - p13 ** 2)
        div_p12 = np.sqrt(1 - p12 ** 2)
        term_y2 = - (x2b2 + a21k + u2mean) / div_p12
        term_y3 = - (x3b3 + a31k + u3mean) / div_p13
        term_y3_a = a[j - 1] / div_p13 + term_y3
        term_y3_b = a[j] / div_p13 + term_y3
        rho = p23 - p12 * p13

        func1 = Phi(term_y2)
        func2 = Phi(term_y3_b) - Phi(term_y3_a)
        func3 = Psi(rho, term_y2, term_y3_a) - Psi(rho, term_y2, term_y3_b)

        if i == 0:
            return func1 * func2 + func3
        elif i == 1:
            return (1 - func1) * func2 - func3
        else:
            raise Exception("invalid i value: %s" % i)

    def P_y1(self, k, u_1, x_1):
        """
        from equation 2.15
        """
        b1 = self.beta_1
        lambda_ = np.exp(np.dot(b1, x_1) + u_1)
        return stats.poisson.pmf(k, lambda_)


def parse_file(filename, i_col, j_col, k_col, x_1_cols, x_2_cols, x_3_cols):
    df = pd.read_csv(filename)

    i = df[i_col]
    j = df[j_col]
    k = df[k_col]

    x_1 = df[x_1_cols].as_matrix()
    x_2 = df[x_2_cols].as_matrix()
    x_3 = df[x_3_cols].as_matrix()

    data = zip(i, j, k, x_1, x_2, x_3)
    return data


def main(smaller=True):
    filename = "barros-holly-data.csv"

    x_1_cols = """gender income income2 public_sub private_sub age age2
    schooling north center alentejo algarve acores madeira""".split()
    x_2_cols = """gender income age age2  schooling diabetes ashtma high_blood_p
    reumat pain ostheo retina glauco cancer kidney_stone renal anxiety enphisema
    stroke obese depression heart_attack public_sub private_sub private_insurance
    age_gender north center alentejo algarve acores madeira""".split()
    x_3_cols = """gender income public_sub private_sub age age2  schooling diabetes
    ashtma high_blood_p reumat pain ostheo retina glauco cancer kidney_stone renal
    anxiety enphisema stroke obese depression heart_attack light_smoker no_smoker
    wine_days single married widow north center alentejo algarve acores
    madeira""".split()

    if smaller:
        x_1_cols = ["gender", "income", "age"]
        x_2_cols = "gender age income schooling".split()
        x_3_cols = "gender income age public_sub private_sub".split()

    i_col = "visit_doctor"
    j_col = "pharma_use"
    k_col = "health"

    data = parse_file(
        filename, i_col, j_col, k_col, x_1_cols, x_2_cols, x_3_cols)

    print()

if __name__ == "__main__":
    main()
