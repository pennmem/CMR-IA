#!/bin/bash
#SBATCH -p cpu
#SBATCH -t 7-00:00
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --mem=5G
#SBATCH -o slurm_out/slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jerryjin@andrew.cmu.edu
##SBATCH -w mind-0-15
#SBATCH --exclude mind-1-15,mind-1-29,mind-0-26,mind-0-28

umask 022

PYFILE="${1:-/home/jerryjin/CMR-IA/Modeling/fitting/pso_cmr.py}"
PY_COMMAND="/home/jerryjin/miniconda3/envs/cmr/bin/python"

COMMAND="$PY_COMMAND $PYFILE"
echo $COMMAND
$COMMAND
