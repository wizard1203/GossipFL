#!/bin/bash
#SBATCH -e logs/slurm-error.log
#SBATCH -o logs/slurm.log
#SBATCH -N 2
#SBATCH -w hkbugpusrv04,hkbugpusrv05
#srun -n 2 /home/comp/20481896/shshi/pytorch1.4/bin/python /home/comp/20481896/shshi/helloworld.py


# 
SBATCH -o logs/slurm.log
SBATCH --nodes=1
SBATCH --ntasks-per-node=15
SBATCH -w hkbugpusrv06
srun /home/comp/20481896/ddl-zoo/experiments/mpi_based/batch_experiment_scripts_for_TCC/daai_worker14.sh


# source ~/.bashrc


























