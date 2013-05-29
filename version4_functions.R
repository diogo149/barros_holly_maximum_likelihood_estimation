# for the polr & mvrnorm functions
library("MASS", lib.loc="/usr/lib/R/library")
# for gauss-hermite weights and zeros
library(glmmML)

# splits a string into a vector of strings
split_to_vec <- function(str) {
  strsplit(gsub("\n", " ", str), " ")[[1]]
}

# given a data frame and a string containing column names, returns the
# columns of the data frame corresponding to those names
load_data <- function(df, col_str) {
  cols = split_to_vec(col_str)
  names_df = names(df)
  for(col in cols) {
    if(!(col %in% names_df)) {
      stop(paste(col, "not in data frame"))
    }
  }
  df[cols]
}

# returns zeros for gauss-hermite quadrature
gauss_hermite_zeros <- function(num) {ghq(num, modified=FALSE)$zeros}

# returns weights for gauss-hermite quadrature
gauss_hermite_weights <- function(num) {ghq(num, modified=FALSE)$weights}

# normal CDF
Phi <- function(x) {pnorm(x)}

# Psi equation, as in equation 1.3
Psi <- function(rho, a, b) {
  if(abs(a + b) == Inf) {
    0
  } else {
    cache1 <- -(a^2 + b^2) / 2
    cache2 <- a * b
    Psi_integrand <- function(t) {
      tmp <- 1 - t^2
      exp((t * cache2 + cache1) / tmp) / sqrt(tmp)
    }
    result <- integrate(Psi_integrand, 0, rho)
    result$value / (2 * pi)
  }
}

# Version of Psi function that can take in vectors for a and b
Psi_vect <- function(rho, a, b) {
  result <- rep(0, length(a))
  for(idx in 1:length(a)) {
    result[idx] <- Psi(rho, a[idx], b[idx])
  }
  result
}

# takes a subset of the data frame's rows
df_subset <- function(df, percent) {
  rows <- dim(df)[1]
  df[sample(1:rows, rows * percent), ]
}

# equation of the covariance matrix for u2 and u3, as in equation 2.44
generate_u_cov <- function(rho12, rho13, rho23) {
  matrix(c(1 - rho12^2, rho23 - rho12 * rho13, rho23 - rho12 * rho13, 1 - rho13^2) , 2)
}

# generate u2 and u3, as in equation 2.44
generate_u2_u3 <- function(n, u1, sigma1, rho12, rho13, rho23) {
  u_cov <- generate_u_cov(rho12, rho13, rho23)
  u_mean <- c(rho12 / sigma1, rho13 / sigma1) * u1
  mvrnorm(n=n, mu=u_mean, Sigma=u_cov)
}

# solve the poisson regression on y1 for dot_product1, as in equation 2.15
get_dot_product1 <- function(target, data, noise) {
  data$noise <- noise
  poisson_regression <- glm(target ~ . + offset(noise) + 0 - noise, data=data, family=poisson(link=log))
  dot_product1 <- predict(poisson_regression, newdata=data) - noise
  dot_product1
}

# solve the probit regression on y2 for dot_product2, as in equation 2.16
get_dot_product2 <- function(target, data, noise) {
  data$noise <- noise
  probit_regression <- glm(target ~ . + offset(noise) - noise, data=data, family=binomial(link=probit))
  dot_product2 <- predict(probit_regression, newdata=data) - noise
  dot_product2
}

# solve the ordered probit regression on y3 for dot_product3 and the intercepts, as in equation 2.16
get_dot_product3 <- function(target, data, noise) {
  data$noise <- noise
  new_target <- factor(target, labels=1:5)
  ordered_probit_regression <- polr(new_target ~ . + offset(noise) - noise, data=data, method="probit")
  data$noise <- NULL
  # need to do this because predict() outputs factors
  dot_product3 <- as.vector(as.matrix(data) %*% coef(ordered_probit_regression))
  intercepts <- c(-Inf, ordered_probit_regression$zeta, Inf)
  list(dot_product3, intercepts)
}

#
is_positive_definite <- function(rho12, rho13, rho23) {
  u_cov <- generate_u_cov(rho12, rho13, rho23)
  all(eigen(u_cov)$values > 0)
}
