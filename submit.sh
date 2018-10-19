#!/bin/bash

for N in  5000 10000 20000
do
   for MODEL in 1 2
   do
      for SHARDS in 4 8 16
      do
         slf=m${MODEL}_n${N}_s${SHARDS}.sl
         sed s/subMODEL/$MODEL/ comp.sl | sed s/subN/$N/ | sed s/subSHARDS/$SHARDS/ | sed s/subREPS/10/ > $slf
         sbatch $slf
         sleep 3
      done
   done

   MODEL=0
   slf=m${MODEL}_n${N}_s${SHARDS}.sl
   sed s/subMODEL/$MODEL/ comp.sl | sed s/subN/$N/ | sed s/subSHARDS/1/ | sed s/subREPS/5/ > $slf
   sbatch $slf
   sleep 3
done

