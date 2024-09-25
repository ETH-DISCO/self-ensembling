#!/bin/bash
#SBATCH --mail-type=NONE # disable email notifications can be [NONE, BEGIN, END, FAIL, REQUEUE, ALL]
#SBATCH --output=/itet-stor/{{USERNAME}}/net_scratch/slurm/%j.out # redirection of stdout (%j is the job id)
#SBATCH --error=/itet-stor/{{USERNAME}}/net_scratch/slurm/%j.err # redirection of stderr
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=tikgpu[08-10]
#CommentSBATCH --nodelist=tikgpu01 # example: specify a node
#CommentSBATCH --account=tik-internal # example: charge a specific account
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb' # example: specify a gpu

set -o errexit # exit on error
mkdir -p /itet-stor/${USER}/net_scratch/slurm

echo "running on node: $(hostname)"
echo "in directory: $(pwd)"
echo "starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

eval "$(/itet-stor/$USER/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
conda activate con

filepath=$1
echo "running script: $filepath"
python3 $filepath

echo "finished at: $(date)"
exit 0
