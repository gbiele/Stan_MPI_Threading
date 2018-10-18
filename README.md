# Stan_MPI_Threading
Comparison of MPI and threading for beta binomial regression models in Stan.

Richard McElreath has a nice tutorial on using map_rect and threading in Stan. The aim of this repository is to briefly describe how to compile models with map_rect for MPI, and to compare speed gains through threading and MPI. Discussions on the Stan forum show that MPI should be faster, so the main goal is to see how much faster MPI is compared to threading.

As threading and MPI currently only work with the cmdstan interface, this is what is used.

## Data
We are using simulated data for a beta-binomial regressions model. The beta binomial model is a good choice to tests MPI and threading, because differently than for than linear or logistic regressions, vectorization does not lead to large efficiency gains for the `beta_binomial_lpf` in Stan. More generally, the beta binomial model is a good model for integer valued outcomes with a lower and upper bound, for which one assumes some overdispersion (without overdispersion a binomial model would do).


The following lines of code generate data for N of (around)  1000,5000,10000,20000 and for K predictors. We use the `stan_rdump` function from the rstan package to save the data in a format that can be read by cmdstan.

```
library(rmutil)
library(rstan)

for (tN in c(1000,5000,10000,20000)) {

N = round(tN/16)*16
K = 10
nT = 10

phi = 15
X = matrix(rnorm(N*K),ncol = K)/3
beta = matrix(scale(rnorm(K)),nrow = K)
m = inv.logit(X %*% beta)
y = rbetabinom(N,rep(T,N),m,phi)

nT = rep(nT,N);

standata = list(N = N,
                K = K,
                nT  = nT,
                X = X,
                y = y)
stan_rdump(names(standata),file = paste0("bbdata",tN,".Rdump"))

}
```
