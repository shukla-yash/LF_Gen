#!/bin/bash
#SBATCH -n 4                # Number of cores
#SBATCH -t 11:30          # Runtime in D-HH:MM
#SBATCH -p gpu               # Partition: gpu, preempt
#SBATCH --gres=gpu:1        # gpu:1, gpu:k20xm:1, gpu:p100:1, gpu:t4:1
#SBATCH --mem=15g
##SBATCH --mem-per-cpu=10000  # Memory (in MB) per cpu
#SBATCH -o log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
export MPLBACKEND='agg'
python test_curr.py
