# https://github.com/ETH-DISCO/cluster-tutorial/blob/main/README.md

# set slurm path
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf

# clean up storage
find /home/$USER -mindepth 1 -maxdepth 1 ! -name 'public_html' -exec rm -rf {} +
rm -rf /scratch/$USER/*
rm -rf /scratch_net/$USER/*

# fix locale issues
unset LANG
unset LANGUAGE
unset LC_ALL
unset LC_CTYPE
echo 'export LANG=C.UTF-8' >> ~/.bashrc
export LANG=C.UTF-8

# convenience aliases for ~/.bashrc.$USER
alias ll="ls -alF"
alias smon_free="grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt"
alias smon_mine="grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt"
alias watch_smon_free="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt\""
alias watch_smon_mine="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt\""

# install conda
cd /itet-stor/$USER/net_scratch/
rm -rf ./install-conda.sh
git clone https://github.com/ETH-DISCO/cluster-tutorial/
mv cluster-tutorial/install-conda.sh . && rm -rf cluster-tutorial # only keep install-conda.sh
chmod +x ./install-conda.sh && ./install-conda.sh
eval "$(/itet-stor/$USER/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
echo '[[ -f /itet-stor/${USER}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${USER}/net_scratch/conda/bin/conda shell.bash hook)"' >> ~/.bashrc # add to bashrc

# ---------------------------------------------

cd /itet-stor/$USER/net_scratch/

# clone project
rm -rf self-ensembling
git clone https://github.com/ETH-DISCO/self-ensembling
cd self-ensembling

# ––––––

# remove previous env if exists
eval "$(/itet-stor/$USER/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
rm -rf /itet-stor/$USER/net_scratch/conda_envs/con && conda remove --yes --name con --all || true
conda info --envs

# create new env
conda env create --file environment.yml
conda activate con
python3 -c "import torch; print(f'pytorch version: {torch.__version__}')"
conda deactivate

# dispatch job
sed -i 's/{{USERNAME}}/'$USER'/g' job.sh # insert username into template
sbatch job.sh ./src/0_hyperparam_v2.py

# check if running
watch -n 1 "squeue | grep $USER"

# check results
for file in /itet-stor/$USER/net_scratch/slurm/*; do if [ -f "$file" ]; then echo -e "\e[32m$(basename "$file")\e[0m"; cat "$file"; echo -e "\n----------\n"; fi; done
