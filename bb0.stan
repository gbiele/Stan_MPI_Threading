data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N,K] X;
  int y[N];
  int nT[N];
}

parameters {
  vector[K] beta;
  real<lower=0> phi;
}

model {
  vector[N] m = inv_logit(X * beta);
  beta ~ normal(0,1);
  phi ~ normal(0,10);
  y ~ beta_binomial(nT, m*phi, (1-m)*phi);
}


