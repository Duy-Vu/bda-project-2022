data {
  int<lower=0> N;
  int<lower=0> V; // amount of variable
  vector[V] x[N];
  real y[N];
  real x1pred;
  real x2pred;
}

parameters {
  real mu;
  real<lower=0> sigma;
  vector[3] beta;
}

model {
  // priors
  mu ~ normal(0, 10);
  sigma ~ gamma(1,1);

  beta[1] ~ normal(100000, 5000); 
  beta[2] ~ normal(mu, sigma);
  beta[3] ~ normal(mu, sigma);
  
  // likelihood
  y ~ normal(beta[1] + beta[2]*to_vector(x[,1]) + beta[3]*to_vector(x[,2]), sigma);
}

generated quantities
{
  vector[N] log_lik;
  real ypred[N];
  int m = 1; 
  for (i in 1:N)
  {
    log_lik[m] = normal_lpdf(y[i] | beta[1] + beta[2]*to_vector(x[,1]) + beta[3]*to_vector(x[,2]), sigma);
    ypred[i] = normal_rng(beta[1] + beta[2]*x1pred + beta[3]*x2pred, sigma);
    m = m + 1;
  }
}
