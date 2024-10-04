#!/bin/bash
#PBS -j oe
#IGNORE PBS -l walltime=72:00:00
#IGNORE PBS -l mem=1500mb
#PBS -l nodes=1
#

# set umask to world readable
umask 022

# PY_COMMAND="/usr/local/python/bin/python"
# PY_COMMAND="/usr/global/python/bin/python"
# PY_COMMAND="~/anaconda3/bin/python"
# PYFILE="/home1/beigejin/CMR_IA/Modeling/fitting/pso_cmr.py"
PY_COMMAND="/home1/beigejin/.conda/envs/CMR_IA/bin/python"

# go to the working directory
# echo "cd $SGE_O_WORKDIR"
# cd $SGE_O_WORKDIR

# start the python job
COMMAND="$PY_COMMAND $PYFILE"
echo $COMMAND
$COMMAND

