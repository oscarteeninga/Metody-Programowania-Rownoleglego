#!/bin/bash
srun -p plgrid-gpu -N 1 -n 1 -A plgmpragh --gres=gpu:1 --pty /bin/bash -l
