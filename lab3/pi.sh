#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 12
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plgmpr21zeus

srun --nodes=1 --ntasks=12 --time=00:10:00 --partition=plgrid --account=plgmpr21zeus --pty /bin/bash

module add plgrid/tools/openmpi

mpicc -o mc_par.x mc_par.c

mpiexec -np 12 ./mc_par.x 1000000000
