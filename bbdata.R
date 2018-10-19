library(boot)
library(rmutil)
library(rstan)

set.seed(123)

N = 20000
K = 10
fnT = rep(10,N)
fX = matrix(rnorm(N*K),ncol = K)/3
beta = matrix(scale(rnorm(K)),nrow = K)
phi = 15
m = inv.logit(fX %*% beta)
fy = rbetabinom(N,fnT,m,phi)

cat(round(beta,digits = 2), file = "true_beta.txt")

for (tN in c(1000,5000,10000,20000)) {

  N = round(tN/16)*16
  #par(mfrow = c(1,3))
  #hist(m)
  #barplot(table(y))
  #plot(m,y)
  X = fX[1:N,]
  nT  = fnT[1:N]
  y = fy[1:N]
  for (n_shards in c(1,4,8,16)) {
    stan_rdump(c("N","K","nT","X","y","n_shards"),
               file = paste0("data/bbdata_n",tN,
                             "_s",n_shards,".Rdump"))
  } 
}
