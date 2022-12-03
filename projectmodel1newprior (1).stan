//Additive model
data {
  int<lower=0> N;
  int<lower=0> J;
  vector[J] y[N];
  vector[J] x1[N];
  vector[J] x2[N];
  real x1pred;
  real x2pred;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real hyperbeta[3];
  real<lower=0> hypersigma;
  vector[J] beta[3];
  real<lower=0> sigma;
}

model {
  hyperbeta[1] ~ normal(50000, 100000);
  hyperbeta[2] ~ normal(10, 10);
  hyperbeta[3] ~ normal(10, 10);
  hypersigma ~ gamma(3,3);
  sigma ~ gamma(3,3);
  for (j in 1:J)
  {
    beta[1,j] ~ normal(hyperbeta[1], hypersigma); 
    beta[2,j] ~ normal(hyperbeta[2], hypersigma);
    beta[3,j] ~ normal(hyperbeta[3], hypersigma);
  }
  for (j in 1:J)
    y[,j] ~ normal(beta[1,j] + beta[2,j]*to_vector(x1[,j]) + beta[3,j]*to_vector(x2[,j]), sigma);
}

generated quantities
{
  vector[J*N] log_lik;
  real ypred[J];
  int m = 1;
  for(j in 1:J)
  {
    for (i in 1:N)
    {
      log_lik[m] = normal_lpdf(y[i,j] | (beta[1,j] + beta[2,j]*x1[i,j] + beta[3,j]*x2[i,j]), sigma);
      m = m + 1;
    }
  }

  for (i in 1:J)
    ypred[i] = normal_rng(beta[1,i] + beta[2,i]*x1pred + beta[3,i]*x2pred, sigma);
}

