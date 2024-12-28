rm -rf /scratch/$USER/*
cd /scratch/$USER

git clone https://github.com/ETH-DISCO/self-ensembling/ && cd self-ensembling

# create conda `environment.yml`
eval "$(/itet-stor/$USER/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
conda info --envs
if conda env list | grep -q "^con "; then
    read -p "the 'con' environment already exists. recreate? (y/n): " answer
    if [[ $answer =~ ^[Yy]$ ]]; then
        conda remove --yes --name con --all
        rm -rf /itet-stor/$USER/net_scratch/conda_envs/con && conda remove --yes --name con --all || true
    fi
fi
conda env create --file environment.yml

# dispatch
sbatch slurm-job-1.sh "./1-resnet/resnet.py"
sbatch slurm-job-2.sh "./2-self-ensemble/self_ensemble.py" # got kicked out of tikgput10, can't train this anymore!

# got kicked out of tikgput10!

# monitor
# srun --mem=10GB --nodelist tikgpu10 --pty bash -i

# watch -n 0.5 "squeue -u $USER --states=R"
# tail -f $(ls -v /scratch/$USER/slurm/job-1/*.err 2>/dev/null | tail -n 300)
# tail -f $(ls -v /scratch/$USER/slurm/job-1/*.out 2>/dev/null | tail -n 300)
# tail -f $(ls -v /scratch/$USER/slurm/job-2/*.err 2>/dev/null | tail -n 300)
# tail -f $(ls -v /scratch/$USER/slurm/job-2/*.out 2>/dev/null | tail -n 300)
