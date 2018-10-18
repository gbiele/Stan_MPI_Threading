#!/bin/bash

#SBATCH --job-name=msubMODEL_nsubN_ssubSHARDS
#SBATCH --account=NN9469K
#SBATCH --ntasks-per-node=subSHARDS
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=3G

module load openmpi.gnu/2.1.0

#cd /usit/abel/u1/guidopb/software/cmdstan-develop
#make ~/programming/R/stan_comp/bb0
#echo "CXXFLAGS += -DSTAN_THREADS" > make/local
#echo "CXXFLAGS += -pthread" >> make/local
#make ~/programming/R/stan_MPI_vs_Threading/bb2
#echo "STAN_MPI=true" > make/local
#echo "CXX=mpicxx" >> make/local
#make ~/programming/R/stan_MPI_vs_Threading/bb1

cd  ~/programming/R/stan_MPI_vs_Threading


bash run_stan.sh -m subMODEL -n subN -s subSHARDS -i subREPS
