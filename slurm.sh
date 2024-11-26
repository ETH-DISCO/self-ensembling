# 
# 1. generate a conda environment
# 2. follow tutorial to attach to a shell
# 3. run this in the terminal
# 

rm -rf /scratch/$USER/*

cd /scratch/$USER
git clone https://github.com/ETH-DISCO/self-ensembling/ && cd self-ensembling
FILEPATH="./1_resnet/resnet.py"

# ---

# create 'con' environment from environment.yml
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

# dispatch job
git clone https://github.com/ETH-DISCO/cluster-tutorial/ && mv cluster-tutorial/job.sh . && rm -rf cluster-tutorial # get job.sh
sed -i 's/{{USERNAME}}/'$USER'/g' job.sh
sed -i 's/{{NODE}}/'tikgpu08'/g' job.sh
sbatch job.sh $FILEPATH

# ---

# check status
watch -n 0.5 "squeue -u $USER --states=R"
tail -f $(ls -v /scratch/$USER/slurm/*.err 2>/dev/null | tail -n 300)
tail -f $(ls -v /scratch/$USER/slurm/*.out 2>/dev/null | tail -n 300)
