#!/bin/bash
#SBATCH -p normal
#SBATCH -t 2-00:00
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --mem=5G
#SBATCH -o slurm_out/slurm-%A_%a.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=jerryjin@andrew.cmu.edu

umask 022

PYFILE="${1:-/home/jerryjin/CMR-IA/Modeling/fitting/pso_cmr.py}"
PY_COMMAND="$GROUP_HOME/miniforge3/envs/cmr/bin/python"

COMMAND="$PY_COMMAND $PYFILE"
echo $COMMAND
$COMMAND
