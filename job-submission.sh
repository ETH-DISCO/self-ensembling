# set slurm path
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf

# clean up storage
find /home/$USER -mindepth 1 -maxdepth 1 ! -name 'public_html' -exec rm -rf {} +
rm -rf /scratch/$USER/*
rm -rf /scratch_net/$USER/*
# rm -rf /itet-stor/$USER/net_scratch/* # also wipes conda

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

# ---------------------------------------------

cd /itet-stor/$USER/net_scratch/
rm -rf /itet-stor/$USER/net_scratch/slurm # clean up previous slurm output

# clone this repository
rm -rf self-ensembling
git clone https://github.com/ETH-DISCO/self-ensembling/
cd self-ensembling
sed -i 's/{{USERNAME}}/'$USER'/g' job.sh # insert username into template

# install conda
chmod +x ./install-conda.sh && ./install-conda.sh

# remove previous env (if exists)
[[ -f /itet-stor/${USER}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${USER}/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
rm -rf /itet-stor/$USER/net_scratch/conda_envs/con && conda remove --yes --name con --all || true # remove previous env if exists

# create new env
conda env create --file environment.yml
conda activate con
python3 -c "import torch; print(torch.__version__)"
conda deactivate

# dispatch job
sbatch job.sh

# check results
watch -n 1 "squeue | grep $USER"
for file in /itet-stor/$USER/net_scratch/slurm/*; do if [ -f "$file" ]; then echo -e "\e[32m$(basename "$file")\e[0m"; cat "$file"; echo -e "\n----------\n"; fi; done


# dispatch job
sbatch job.sh ./src/0_hyperparam_optim_v2.py
