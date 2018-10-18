# MPI and threading in Stan
Comparison of MPI and threading for beta binomial regression models in Stan.

Richard McElreath has a nice [tutorial on using map_rect and threading in Stan](https://github.com/rmcelreath/cmdstan_map_rect_tutorial). The aim of this repository is to briefly describe how to compile models with map_rect for MPI, and to compare speed gains through threading and MPI. Discussions in the [Stan forums](https://discourse.mc-stan.org/) show that MPI should be faster, so the main goal is to see how much faster MPI is compared to threading.

As threading and MPI currently only work with the cmdstan interface, this is what is used.

### Data
We are using simulated data for a beta-binomial regressions model. The beta binomial model is a good choice to tests MPI and threading, because differently than for linear or logistic regressions, vectorization does not lead to large efficiency gains for the `beta_binomial_lpf` in Stan. Therefore we can expect gains from threading already for relatively small data sets (i.e. thousands or tens of thousands rows).


The following R code generates data for with (around)  1000 ,5000, 10000, and 20000 rows and `K = 10` predictors. We use the `stan_rdump` function from the rstan package to save the data in a format that can be read by cmdstan models.

```R
library(rmutil)
library(rstan)

for (tN in c(1000,5000,10000,20000)) {
  # make sure N can be devided by 16
  N = round(tN/16)*16 
  K = 10
  nT = 10
  
  phi = 15
  X = matrix(rnorm(N*K),ncol = K)/3
  beta = matrix(scale(rnorm(K)),nrow = K)
  m = inv.logit(X %*% beta)
  y = rbetabinom(N,rep(T,N),m,phi)
  nT = rep(nT,N);
  
  stan_rdump(c("N","K","nT","X","y"),
  file = paste0("bbdata",tN,".Rdump"))
}
```

### Basic regression model
The basic beta binomial regression model (_bb0.stan_ in the repository) looks as follows:
```c++
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

```

### Regression model with map_rect
MPI and threading rely on the same map-reduce approach.
To make use of parallelisation, we need to split data into packages (shards). With the `map_rect` function the packages are distributed to the nodes or threads, where a user defined reduce function unpacks the data and calculates the log posterior. 

Stan's `map_rect` function has the  signature 
` map_rect(function, vector[], real[,], real[,], int[,])` where

- `function` is the reduce function
- `vector` holds global parameters
- the 1st `real[,]` array holds shard-specific parameters
- the 2nd `real[,]` array holds real valued variables and
- the `int[,]` array holds integer valued variables.

In the arrays, the size of the first dimension is equal to the number of shards, the second dimension of the parameter array is the number of shard specific parameters, and the second dimension of the variable arrays is equal to tje size of each shard (which has to be constant).

For the beta binomial regression implemented here

- the first vector contains regression weights and the over-dispersion parameter,
- the first `real[,]` array is empty because there are no shard specific parameters,
- the second `real[,]` holds all predictor variables from the matrix X,
- the `int[,]` array holds the outcome `y` and the number of trials `nT`.
