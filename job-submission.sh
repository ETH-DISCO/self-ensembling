# https://github.com/ETH-DISCO/cluster-tutorial/blob/main/README.md

# ---------------------------------------------

cd /itet-stor/$USER/net_scratch/

# clone project
rm -rf self-ensembling
git clone https://github.com/ETH-DISCO/self-ensembling
cd self-ensembling

# create env
eval "$(/itet-stor/$USER/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
rm -rf /itet-stor/$USER/net_scratch/conda_envs/con && conda remove --yes --name con --all || true
conda info --envs
conda env create --file environment.yml
conda activate con
python3 -c "import torch; print(f'pytorch version: {torch.__version__}')"
conda deactivate

# dispatch job
sed -i 's/{{USERNAME}}/'$USER'/g' job.sh # insert username into template
sbatch job.sh ./mnist.py

# check status
watch -n 1 "squeue | grep $USER"
for file in /itet-stor/$USER/net_scratch/slurm/*; do if [ -f "$file" ]; then echo -e "\e[32m$(basename "$file")\e[0m"; cat "$file"; echo -e "\n----------\n"; fi; done
