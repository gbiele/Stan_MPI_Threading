#!/bin/bash

while  getopts m:n:s:i: option  ; do 
  case "${option}"
    in
      m) MODEL=${OPTARG};;
      n) N=${OPTARG};;
      s) SHARDS=${OPTARG};;
      i) REPS=${OPTARG};;
  esac
done

COUNTER=0
fn=results/times_m${MODEL}_n${N}_s${SHARDS}.txt
echo model n shards seconds  > $fn
dataf=data/bbdata_n${N}_s${SHARDS}.Rdump
outf=out/output_m${MODEL}_n${N}_s${SHARDS}

while [ $COUNTER -lt $REPS ] ; do
  outf=out/output_m${MODEL}_n${N}_s${SHARDS}_i${COUNTER}.csv
  SECONDS=0
  if [ $MODEL == 0 ]
  then 
    ./bb0 sample data file=$dataf init=1 output file=$outf
  elif [ $MODEL == 1 ]
  then
    mpirun -np $SHARDS -q bb1 sample data file=$dataf init=1 output file=$outf
  elif [ $MODEL == 2 ]
  then
    export STAN_NUM_THREADS=$SHARDS
    ./bb2 sample data file=$dataf init=1 output file=$outf
  fi
 
  echo $MODEL $N  $SHARDS $SECONDS >> $fn
  echo $MODEL $N  $SHARDS $SECONDS >> results/all.txt
  let COUNTER=COUNTER+1
done

