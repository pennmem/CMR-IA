#!/bin/bash
echo Please enter a name for your new Anaconda environment:
read ENV_NAME
conda create -n $ENV_NAME python=3.9 ipykernel
source activate $ENV_NAME
pip install -r requirements.txt
python -m ipykernel install --user --name $ENV_NAME --display-name "$ENV_NAME"
python setup.py install
