#!/bin/sh
#$ -cwd
#$ -l q_node=64
#$ -l h_rt=24:00:00
#$ -N cyc_ALL_1E-4_b0.5
. /etc/profile.d/modules.sh
module load gcc/8.3.0 cuda/10.1.105 cudnn/7.6 python/3.6.5 openmpi nccl/2.4.2
rm -rf __pycache__
mpirun -n 64 -bind-to none --map-by node -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH python cyclegan.py 0
#mpirun -n 2 -bind-to none --map-by node -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH python cyclegan.py 0 
#qrsh -l q_node=2 -l h_rt=2:00:00 -g tga-yamaguchi.m