#!/bin/bash

#SBATCH -p gpu22
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --gres gpu:1
#SBATCH -t 0-05:59:59
#SBATCH -o stats_pf_%j.out
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ahmed.abbas@mpi-inf.mpg.de
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA_new

python data_stats.py --root-dir /home/ahabbas/data/learnDBCA/cv_structure_pred/mrf/protein_folding
#python data_stats.py --root-dir /home/ahabbas/data/learnDBCA/shape_matching/train_split/instances/
exit 0