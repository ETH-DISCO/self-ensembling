# clone project
mkdir -p /scratch/$USER
cd /scratch/$USER
git clone https://github.com/ETH-DISCO/self-ensembling/ && cd self-ensembling

# create conda `environment.yml` from project
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

#
# dispatch
#

sbatch \
    --output=$(pwd)/%j.out \
    --error=$(pwd)/%j.err \
    --nodelist=$(hostname) \
    --mem=150G \
    --nodes=1 \
    --gres=gpu:1 \
    --wrap="bash -c 'source /itet-stor/${USER}/net_scratch/conda/etc/profile.d/conda.sh && conda activate con && python3 $(pwd)/1-resnet/resnet.py'"

sbatch \
    --output=$(pwd)/%j.out \
    --error=$(pwd)/%j.err \
    --nodelist=$(hostname) \
    --mem=150G \
    --nodes=1 \
    --gres=gpu:1 \
    --wrap="bash -c 'source /itet-stor/${USER}/net_scratch/conda/etc/profile.d/conda.sh && conda activate con && python3 $(pwd)/2-self-ensemble/self_ensemble.py'"

#
# monitoring
#

watch -n 0.5 "squeue -u $USER --states=R"
tail -f $(ls -v $(pwd)/*.err 2>/dev/null | tail -n 300)
tail -f $(ls -v $(pwd)/*.out 2>/dev/null | tail -n 300)
