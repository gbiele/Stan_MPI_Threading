# MPI and threading in Stan

Richard McElreath has a nice [tutorial on using map_rect and threading in Stan](https://github.com/rmcelreath/cmdstan_map_rect_tutorial). The aim of this repository here is to compare speed gains through threading and MPI. Discussions in the [Stan forums](https://discourse.mc-stan.org/) show that MPI should be faster, so the main goal is to see how much faster MPI is.

Current, only cmdstan supports threading and and MPI and the latter is only supported on OS-X and Linux.

### Data
We simulate data for a beta-binomial regressions model. The beta-binomial model is a good choice to tests MPI and threading, because vectorization does not lead to large efficiency gains for the `beta_binomial_lpf` in Stan (as it does for example for the `normal_lpdf`). Therefore we can expect gains from MPI and threading for relatively small data sets.

The R code generates data for with (around)  1000, 5000, 10000, and 20000 rows and `K = 10` predictors. The `stan_rdump` function from the rstan package saves the data in a format readable by cmdstan models (and also saves some other data later needed for the fitting).

```R
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
  X = fX[1:N,]
  nT  = fnT[1:N]
  y = fy[1:N]
  for (n_shards in c(1,4,8,16)) {
    stan_rdump(c("N","K","nT","X","y","n_shards"),
               file = paste0("data/bbdata_n",tN,
                             "_s",n_shards,".Rdump"))
  } 
}

```

### Basic regression model
Here is the basic beta binomial regression model [bb0.stan](https://github.com/gbiele/Stan_MPI_Threading/blob/master/bb0.stan):
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
To make use of parallelisation, we have to split data into packages (shards). The `map_rect` function distributes the data packages to the nodes or threads, where a user defined `reduce` function unpacks the data and calculates the log posterior. 

Stan's `map_rect` function has the  signature 
` map_rect(function, vector[], real[,], real[,], int[,])` where

- `function` is the reduce function
- `vector` holds global parameters
- the 1st `real[,]` array holds shard-specific parameters
- the 2nd `real[,]` (`xr`) array holds real valued variables and
- the `int[,]` (`xi`) array holds integer valued variables.

The size of the first array dimension is equal to the number of shards, the second dimension of the parameter array is the number of shard specific parameters, the second dimension of the variable arrays `xr`and `xi`is equal to the size of a shard (which is constant).

For the beta binomial regression implemented here

- the first vector contains regression weights and the over-dispersion parameter,
- the first `real[,]` array `theta` is empty because there are no shard specific parameters,
- the second `real[,]` array `xr` holds all predictor variables from the matrix X,
- the `int[,]` array `xi` holds the outcome `y` and the number of trials `nT`.

In the `transformed data` block predictors (`X`) the outcome (`y`) and the number of trials (`nT`) are packed in `xi`and `xr`:
```c++
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
```

The `lp_reduce` function in the `functions` block takes, in addition to `xi` and `xr`, global parameters `beta_phi` and shard-specific parameters `theta` as arguments, unpacks all inputs, and calculates the log posterior:
```c++
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
```

Similar to the real and integer data, global parameters also have to be be packed into a vector before they are submitted to the `map_rect` function: 
```c++
model {
  vector[K+1] beta_phi; 
  beta ~ normal(0,1);
  phi ~ normal(0,10);
  beta_phi = to_vector(append_row(beta,phi));  

  target += sum( map_rect( lp_reduce , beta_phi, theta, xr, xi ) );  
}
```

The file
[bb1.stan](https://github.com/gbiele/Stan_MPI_Threading/blob/master/bb1.stan) in this repository has the complete model `using map_rect`.


### Compiling for threading and MPI

On the cluster the analysis was performed I first loaded **gcc 6.3.0** and openmpi **2.1.0**.

Threading and MPI use the same Stan model, but the compilation flags need to be adjusted. In the main cmdstan directory for **MPI**:
```sh
echo "STAN_MPI=true" > make/local
echo "CXX=mpicxx" >> make/local
```
and for **threading**:
```sh
echo "CXXFLAGS += -DSTAN_THREADS" > make/local
echo "CXXFLAGS += -pthread" >> make/local
```

**MPI models** are started with

```sh
mpirun -np 4 bb1 sample data file=my_data.Rdump
```

where the `-np` flag indicates the number of cores to be used.

**Threading models** require setting the number of threads:

```sh
export STAN_NUM_THREADS=4
./bb2 sample data file=my_data.Rdump
```
The scripts [submit.sh](https://github.com/gbiele/Stan_MPI_Threading/blob/master/submit.sh) and [run_stan.sh](https://github.com/gbiele/Stan_MPI_Threading/blob/master/run_stan.sh) in the repository submit a batch of analyses and runs them. [comp.sl](https://github.com/gbiele/Stan_MPI_Threading/blob/master/comp.sl) is a template [slurm](https://slurm.schedmd.com/) job-script.

The analysis was performed on a [cluster](https://www.uio.no/english/services/it/research/hpc/abel/more/index.html) with dual Intel E5-2670 (Sandy Bridge) processors running at 2.6 GHz and 16 physical compute cores per node. Each node has 64 GB RAM (1600 MHz). The operating system is Linux CentOS release 6.9.

### Result: Comparison of the basic, MPI and threading models.

I compared the models by fitting beta-binomial regression models with `K = 10` predictors and `N = ` 1000, 5000, 10000, and 20000 rows. The basic model used one core, threading and MPI used 4, 8 or 16 cores, each time the same number of shards as cores. I did not further investigate the optimal number of shards!

The basic model took  63, 320, 661, 1267 seconds for `N =` 1000, 5000, 10000, and 20000, respectively (1000 warmup and 1000 post warmup samples). The table below shows the proportion of the time of the basic model the threading and MPI analyses took. <sup>1</sup>


| analysis  |  1000 | 5000  | 10000  | 20000  | 
|---|---|---|---|---|
| 4 shards, MPI  | 0.46 | 0.42 | 0.41 | 0.39 |
| 4 shards, Threading  | 0.40 | 0.45 | 0.65 | 0.52 |
| 8 shards, MPI | 0.35 | 0.22 | 0.20 | 0.22
| 8 shards, Threading | 0.53 | 0.43 | 0.41 | 0.30 |
| 16 shards, MPI | 0.22 | 0.12 | 0.12 | 0.12 |
| 16 shards, Threading | 0.22 | 0.15 | 0.14 | 0.13 |


_MPI and threading reduce computation (waiting) time already for relatively small data sets (N = 1000). MPI is generally faster than threading, but the advantage of MPI is smaller when many cores are available._

The next table shows what proportion of a linear speed up MPI and threading achieve (i.e,. if 16 cores would be 16 times as fast as 1 core the value would be 1.)

| analysis  |  1000 | 5000  | 10000  | 20000 | 
|---|---|---|---|---|
| 4 shards, MPI  | 0.54 | 0.60 | 0.61 | 0.63 |
| 4 shards, Threading  | 0.62 | 0.56 | 0.39 | 0.48 |
| 8 shards, MPI | 0.36 | 0.58 | 0.62 | 0.56 |
| 8 shards, Threading | 0.24 | 0.29 | 0.30 | 0.42 |
| 16 shards, MPI | 0.28 | 0.50 |  0.53 |  0.52 |
| 16 shards, Threading | 0.29 | 0.42 | 0.45 | 0.47 |

### Conclusion

As expected, MPI is faster than threading, but the advantage gets smaller as more cores become available.  It appears useful to keep in mind that speeding up model fitting comes at the cost of using more computational resources and energy.

**These results were obtained with a beta-binomial regression with few efficiency gains through vectorization in Stan. The pictures can be much different for other models like linear regression, where much larger data set are needed to realize speed up through threading or MPI.**. 


<sup>1</sup>Averaged over 5 runs of a model for the basic model and over 10 runs for the MPI and threading analyses.
