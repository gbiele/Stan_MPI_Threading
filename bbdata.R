library(boot)
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
  #par(mfrow = c(1,3))
  #hist(m)
  #barplot(table(y))
  #plot(m,y)
  nT = rep(nT,N);
  
  
  for (n_shards in c(4,8,16)) {
    standata = list(N = N,
                    K = K,
                    nT  = nT,
                    X = X,
                    y = y,
                    n_shards = n_shards)
    stan_rdump(names(standata),
               file = paste0("data/bbdata_N",tN,
                             "_s",n_shards,".Rdump"))
  }
}
