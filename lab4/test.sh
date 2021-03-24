#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 12
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plgmpr21zeus
#SBATCH --constraint="intel" 

srun --nodes=1 --ntasks=12 --time=00:10:00 --partition=plgrid --account=plgmpr21zeus --pty /bin/bash
module add plgrid/tools/openmpi

make build

chmod 777 *

make strong n=500000000
make strong n=22360679l
make strong n=1000000

make weak n=50000000
make weak n=2236067
make weak n=100000