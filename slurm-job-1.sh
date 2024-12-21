#!/bin/bash
#SBATCH --mail-type=NONE # disable email notifications can be [NONE, BEGIN, END, FAIL, REQUEUE, ALL]
#SBATCH --output=/scratch/yjabary/slurm/job-1/%j.out # redirection of stdout (%j is the job id)
#SBATCH --error=/scratch/yjabary/slurm/job-1/%j.err # redirection of stderr
#SBATCH --nodelist=tikgpu10 # choose specific node
#SBATCH --mem=150G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#CommentSBATCH --cpus-per-task=4
#CommentSBATCH --account=tik-internal # example: charge a specific account
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb' # example: specify a gpu

set -o errexit # exit on error
mkdir -p /scratch/$USER/slurm

echo "running on node: $(hostname)"
echo "in directory: $(pwd)"
echo "starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

eval "$(/itet-stor/$USER/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
conda activate con

filepath=$1
echo "running script: $filepath"

# restart on failure
attempts=25
for attempt in $(seq 1 $attempts); do
    python3 $filepath
    if [ $? -eq 0 ]; then
        echo "finished at: $(date)"
        exit 0
    else
        echo "Attempt $attempt failed."
        sleep 60
    fi
done
echo "job failed after $attempts attempts"
exit 1
