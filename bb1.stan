functions {
  vector lp_reduce( vector beta_phi , vector theta , real[] xr , int[] xi ) {
    int M = size(xi)/2;
    int K = size(xr)/M;
    int y[M] = xi[1:M];
    int nT[M] = xi[(M+1):(2*M)];
    matrix[M,K] X;
    vector[M] m;
    real lp;
    real phi = beta_phi[K+1];
    vector[K] beta = beta_phi[1:K];
    for (k in 1:K)
       X[1:M,k] = to_vector(xr[(M*k-M+1):(M*k)]);
    m = inv_logit(X * beta);

    lp = beta_binomial_lpmf(y | nT, m*phi, (1-m)*phi);
    return [lp]';
  }
} 

data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N,K] X;
  int y[N];
  int nT[N];
  int n_shards;
}

transformed data {
  vector[0] theta[n_shards];
  int M = N/n_shards;
  int xi[n_shards, M*2];
  real xr[n_shards, M*K];
  
  for (s in 1:n_shards) {
    int i = (s-1)*M + 1;
    int j = s*M;
    int MK = M*K;
    xi[s,1:M] = y[i:j];           
    xi[s,(M+1):(2*M)] = nT[i:j];
    for (k in 1:K) {
      int sidx = (M*k-M+1);
      int eidx = (M*k);
      xr[s,sidx:eidx] = to_array_1d(X[i:j,k]);
    }
  }
}

parameters {
  vector[K] beta;
  real<lower=0> phi;
}

model {
  vector[K+1] beta_phi; 
  beta ~ normal(0,1);
  phi ~ normal(0,10);
  beta_phi = to_vector(append_row(beta,phi));  

  target += sum( map_rect( lp_reduce , beta_phi, theta, xr, xi ) );  
}
