#!/bin/bash

module load miniconda-3
source activate tf2-gpu-py3.8.3

cwd=$1
cd $1

python3 ${1}/prg_cossim.py

echo NORMAL END