notes = "
-reression 1 had no constant/intercept
-regression 2 had an intercept
-regression 3 had no intercept
-can tune function for Psi by using faster integrate function or adjusting absolute / relative errror
-provide a starting value to polr based on past starting values
"
# for the user created helper functions
source("version4_functions.R")

filename = "barros-holly-v2.csv"
gauss_hermite_points = 2

k_col = "visit_doctor"
i_col = "pharma_use"
j_col = "health"

x1_col_str = "gender income income2 public_sub private_sub age age2 schooling
north center alentejo algarve acores madeira"
x2_col_str = "gender income age age2 schooling diabetes ashtma high_blood_p
reumat pain ostheo retina glauco cancer kidney_stone renal anxiety enphisema
stroke obese depression heart_attack public_sub private_sub private_insurance
age_gender north center alentejo algarve acores madeira"
x3_col_str = "gender income public_sub private_sub age age2 schooling diabetes
ashtma high_blood_p reumat pain ostheo retina glauco cancer kidney_stone renal
anxiety enphisema stroke obese depression heart_attack light_smoker no_smoker
wine_days single married widow north center alentejo algarve acores
madeira"

# set to FALSE for using the full set of columns
if(TRUE) {
  x1_col_str = "gender income age"
  x2_col_str = "gender age income schooling"
  x3_col_str = "gender age income public_sub private_sub"
}

# this adds the y1 as a factor for the regression on y2, as in equation 2.16
x2_col_str = paste(x2_col_str, k_col)

# this adds the y2 and y3 as factos for the regression on y3, as in equation 2.16
x3_col_str = paste(x3_col_str, k_col)
x3_col_str = paste(x3_col_str, i_col)

# read the data frame from a file
df <- read.csv(filename)

# set to FALSE to use all the rows of the data
if(TRUE) {
  df <- df_subset(df, 0.01)
}

# keep the number of rows for later computations
num_rows <- dim(df)[1]

# and select the relevant x & y columns
y1 <- load_data(df, k_col)
y2 <- load_data(df, i_col)
y3 <- load_data(df, j_col)
x1 <- load_data(df, x1_col_str)
x2 <- load_data(df, x2_col_str)
x3 <- load_data(df, x3_col_str)

##### This area is for computations independent of the parameters of likelihood,
##### and thus, we only have to do once

# obtain the values of u1 as gauss-hermite quadrature roots
gh_zeros <- gauss_hermite_zeros(gauss_hermite_points)
gh_weights <- gauss_hermite_weights(gauss_hermite_points)

# the sign of y2, used for equations
y2_sign <- 2 * y2[,i_col] - 1

# vector versions of i, j, k
k <- y1[,k_col]
i <- y2[,i_col]
j <- y3[,j_col]

#####

# assuming the params is a vector composed of sigma1, rho12, rho13, rho23 in that order
likelihood <- function(params) {
  sigma1 <- params[1]
  rho12 <- params[2]
  rho13 <- params[3]
  rho23 <- params[4]
  
  ##### This area is for computations independent of the value of u1,
  ##### and thus, we only have to do once
  
  # commonly used constants
  div_rho12 <- 1 / sqrt(1 - rho12 ^ 2)
  div_rho13 <- 1 / sqrt(1 - rho13 ^ 2)
  
  # the value of rho used for each call of Psi
  Psi_rho <- (rho23 - rho12 * rho13) * div_rho12 * div_rho13
  
  #####
  
  # set each of the row likelihoods to be initially zero
  row_likelihood <- rep(0.0, num_rows)
  
  for (cnt in 1:gauss_hermite_points) {
    gh_zero <-gh_zeros[cnt] 
    gh_weight <- gh_weights[cnt]
    
    # get u1 as a function of the gauss-hermite zero/root, as in equation 2.76
    u1 <- gh_zero * sqrt(2 * sigma1)
    
    # generate u2 and u3, as in equation 2.44
    u2_u3 <- generate_u2_u3(num_rows, u1, sigma1, rho12, rho13, rho23)
    u2 <- u2_u3[,1]
    u3 <- u2_u3[,2]
    
    # solve the poisson regression on y1 for mu1, as in equation 2.15
    mu1 <- get_mu1(k, x1, rep(u1, num_rows))

    # solve the probit regression on y2 for mu2, as in equation 2.16
    mu2 <- get_mu2(i, x2, u2)

    # solve the ordered probit regression on y3 for mu3 and the intercepts, as in equation 2.16
    mu3_intercepts <- get_mu3(j, x3, u3)
    mu3 <- mu3_intercepts[[1]]
    intercepts <- mu3_intercepts[[2]]

    ##### the following sections computes the likelihood of each point, as in equation 2.54
    
    # solve for P(y1 | u1), as in equation 2.53
    p_y1_const <- (mu1 ^ k) / factorial(k)
    p_y1 <- p_y1_const * exp(k * u1 - mu1 * exp(u1))
    print(p_y1_const)
    print(p_y1)
    
    ### solve for P(y2, y3 | y1, u1), as in equations 2.45 and 2.46
    
    # the term with variables relating to y2
    y2_term <- -div_rho12 * (mu2 + rho12 * u1 / sigma1)
    
    # Phi of the y2 term. Using the fact that 1-Phi(x) = Phi(-x), to take into
    # account the different equations for the different i values
    Phi_y2 <- Phi(-y2_term * y2_sign)
    
    # the term with variables relating to y3
    y3_term <- -div_rho13 * (mu3 + rho13 * u1 / sigma1)
    
    # the upper and lower terms correspond to have c[j] and c[j-1], respectively
    # using j + 1 for upper and j for lower because R indexes from 1
    upper_term <- intercepts[j + 1] * div_rho13 + y3_term
    lower_term <- intercepts[j] * div_rho13 + y3_term
    
    # the term with Phi and the y3 variables
    Phi_y3 <- Phi(upper_term) - Phi(lower_term)
    
    # the term with Psi and the y3 variables
    Psi_y3 <- y2_sign * (Psi_vect(Psi_rho, y2_term, lower_term) - Psi_vect(Psi_rho, y2_term, upper_term))
    
    ###
    
    p_y1_y2_y3 <- (Phi_y2 * Phi_y3 + Psi_y3) * p_y1 / sqrt(pi)
    print(max(Phi_y2))
    print(min(Phi_y2))
    print(max(Phi_y3))
    print(min(Phi_y3))
    print(max(Psi_y3))
    print(min(Psi_y3)) # TO DO: both p_y1 and Psi_y3 get negative values, and p_y1 gets values > 1
    print(max(p_y1))
    print(min(p_y1))
    print(k[p_y1 < 0])
    print(j[Psi_y3 < 0])
    print("")
    #####
    
    # add the likelihood at this u1 to the total likelihood (for the gauss-hermite integration)
    row_likelihood <- row_likelihood + p_y1_y2_y3 * gh_weight
  }
  
  print(row_likelihood)
  # take the log and add them all up for total likelihood
  sum(log(row_likelihood))
}