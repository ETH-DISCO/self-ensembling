cd /scratch/$USER
rm -rf ./*

git clone https://github.com/ETH-DISCO/self-ensembling/ && cd self-ensembling
FILEPATH="./src/0_hyperparams_v2.py"

# ------------------- dispatch job

# create environment.yml
eval "$(/itet-stor/$USER/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
conda info --envs
if conda env list | grep -q "^con "; then
    read -p "the 'con' environment already exists. do you want to remove and recreate it? (y/n): " answer
    if [[ $answer =~ ^[Yy]$ ]]; then
        conda remove --yes --name con --all
        rm -rf /itet-stor/$USER/net_scratch/conda_envs/con && conda remove --yes --name con --all || true
    fi
fi
conda env create --file environment.yml

# dispatch job
git clone https://github.com/ETH-DISCO/cluster-tutorial/ && mv cluster-tutorial/job.sh . && rm -rf cluster-tutorial # get job.sh
sed -i 's/{{USERNAME}}/'$USER'/g' job.sh # template username
sed -i 's/{{NODE}}/'tikgpu07'/g' job.sh # template node
sbatch job.sh $FILEPATH

# check status
watch -n 1 "squeue | grep $USER"
ls -v cd /scratch/$USER/slurm/* | tail -n 1 | xargs cat
