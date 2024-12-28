export SLURM_CONF=/home/sladmitet/slurm/slurm.conf

find /home/$USER -mindepth 1 -maxdepth 1 ! -name 'public_html' -exec rm -rf {} +
rm -rf /scratch/$USER/*
rm -rf /scratch_net/$USER/*
cd /itet-stor/$USER/net_scratch/
shopt -s extglob
rm -rf !("conda"|"conda_envs"|"conda_pkgs")
shopt -u extglob

unset LANG
unset LANGUAGE
unset LC_ALL
unset LC_CTYPE
echo 'export LANG=C.UTF-8' >> ~/.bashrc
export LANG=C.UTF-8

alias ll="ls -alF"
alias smon_free="grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt"
alias smon_mine="grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt"
alias watch_smon_free="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt\""
alias watch_smon_mine="watch --interval 300 --no-title --differences --color \"grep --color=always --extended-regexp '${USER}|$' /home/sladmitet/smon.txt\""

cd /itet-stor/$USER/net_scratch/
if [ ! -d "/itet-stor/${USER}/net_scratch/conda" ] && [ ! -d "/itet-stor/${USER}/net_scratch/conda_pkgs" ]; then
  git clone https://github.com/ETH-DISCO/cluster-tutorial/ && mv cluster-tutorial/install-conda.sh . && rm -rf cluster-tutorial # only keep install-conda.sh
  chmod +x ./install-conda.sh && ./install-conda.sh
  eval "$(/itet-stor/$USER/net_scratch/conda/bin/conda shell.bash hook)" # conda activate base
  echo '[[ -f /itet-stor/${USER}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${USER}/net_scratch/conda/bin/conda shell.bash hook)"' >> ~/.bashrc # add to bashrc
fi

grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt

srun --mem=100GB --gres=gpu:01 --nodelist tikgpu10 --pty bash -i

# 
# dispatch
# 

cd /scratch/$USER
git clone https://github.com/ETH-DISCO/self-ensembling/ && cd self-ensembling

sbatch ./batch-job.sh ./1-batch/batch.py 0
sbatch ./batch-job.sh ./1-batch/batch.py 1
sbatch ./batch-job.sh ./1-batch/batch.py 2
sbatch ./batch-job.sh ./1-batch/batch.py 3
sbatch ./batch-job.sh ./1-batch/batch.py 4
sbatch ./batch-job.sh ./1-batch/batch.py 5
sbatch ./batch-job.sh ./1-batch/batch.py 6
sbatch ./batch-job.sh ./1-batch/batch.py 7

#
# monitoring
#

watch -n 0.5 "squeue -u $USER --states=R"
grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt

salloc --mem=10GB --nodelist=tikgpu07

tail -f $(ls -v /scratch/$USER/slurm/*.err 2>/dev/null | tail -n 300)
tail -f $(ls -v /scratch/$USER/slurm/*.out 2>/dev/null | tail -n 300)
