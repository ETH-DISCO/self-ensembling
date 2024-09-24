cd /itet-stor/$USER/net_scratch/
rm -rf /itet-stor/$USER/net_scratch/slurm # clean up previous slurm output

# clone this repository
rm -rf self-ensembling
git clone https://github.com/ETH-DISCO/self-ensembling
cd self-ensembling
sed -i 's/{{USERNAME}}/'$USER'/g' job.sh # insert username into template

# install conda
chmod +x ./install-conda.sh && ./install-conda.sh
eval "$(/itet-stor/${USER}/net_scratch/conda/bin/conda shell.bash hook)"
echo '[[ -f /itet-stor/${USER}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${USER}/net_scratch/conda/bin/conda shell.bash hook)"' >> /home/$USER/.bashrc

# create conda env
[[ -f /itet-stor/${USER}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${USER}/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
rm -rf /itet-stor/yjabary/net_scratch/conda_envs/con && conda remove --yes --name con --all || true # remove if exists
conda env create --file environment.yml
conda activate con
python3 -c "import torch; print(torch.__version__)"
conda deactivate

# dispatch job
sbatch job.sh ./src/0_hyperparam_optim_v2.py

# check results
watch -n 1 "squeue | grep $USER"

# check logs
for file in /itet-stor/$USER/net_scratch/slurm/*; do if [ -f "$file" ]; then echo -e "\e[32m$(basename "$file")\e[0m"; cat "$file"; echo -e "\n----------\n"; fi; done

# clean up
[[ -f /itet-stor/${USER}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${USER}/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
conda remove --yes --name con --all
