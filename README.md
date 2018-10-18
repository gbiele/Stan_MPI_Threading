# Stan_MPI_Threading
Comparison of MPI and threading for beta binomial regression models in Stan.

Richard McElreath has a nice tutorial on using map_rect and threading in Stan. The aim of this repository is to briefly describe how to compile models with map_rect for MPI, and to compare speed gains through threading and MPI. Discussions on the Stan forum show that MPI should be faster, so the main goal is to see how much faster MPI is compared to threading.



## Data
We are using simulated data for a beta-binomial regressions model. The beta binomial model is a good choice to tests MPI and threading, because differently than for than linear or logistic regressions, vectorization does not lead to large efficiency gains for the `beta_binomial_lpf` in Stan. More generally, the beta binomial model is a good model for integer valued outcomes with a lower and upper bound, for which one assumes some overdispersion (without overdispersion a binomial model would do).
