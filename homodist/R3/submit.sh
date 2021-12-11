#!/bin/bash
#PBS -l nodes=1:ppn=1,walltime=24:00:00
#PBS -A ACF-UTK0009
#PBS -m ae
#PBS -j eo
#PBS -N ABM
module load anaconda2/4.4.0-gnu
cd $PBS_O_WORKDIR
python Driver.py 0.005 0.02

