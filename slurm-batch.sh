# `salloc --mem=10GB --nodelist=artongpu01`
# `salloc --mem=10GB --nodelist=artongpu02`
# `salloc --mem=10GB --nodelist=artongpu03` --> dispatched
# `salloc --mem=10GB --nodelist=artongpu04` --> dispatched
# `salloc --mem=10GB --nodelist=artongpu05` --> dispatched
# `salloc --mem=10GB --nodelist=artongpu06` --> dispatched
# `salloc --mem=10GB --nodelist=artongpu07` --> dispatched
# `salloc --mem=10GB --nodelist=tikgpu05` --> dispatched
# `salloc --mem=10GB --nodelist=tikgpu10` --> running old jobs, can't read

# 
# dispatch
# 

rm -rf /scratch/$USER/*

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

# max 3 jobs per node (idk why)
sbatch --output=$(pwd)/%j.out --error=$(pwd)/%j.err --nodelist=$(hostname) --mem=150G --nodes=1 --gres=gpu:1 --wrap="bash -c 'source /itet-stor/${USER}/net_scratch/conda/etc/profile.d/conda.sh && conda activate con && python3 $(pwd)/1-batch/batch.py 0 3'"
sbatch --output=$(pwd)/%j.out --error=$(pwd)/%j.err --nodelist=$(hostname) --mem=150G --nodes=1 --gres=gpu:1 --wrap="bash -c 'source /itet-stor/${USER}/net_scratch/conda/etc/profile.d/conda.sh && conda activate con && python3 $(pwd)/1-batch/batch.py 1 3'"
sbatch --output=$(pwd)/%j.out --error=$(pwd)/%j.err --nodelist=$(hostname) --mem=150G --nodes=1 --gres=gpu:1 --wrap="bash -c 'source /itet-stor/${USER}/net_scratch/conda/etc/profile.d/conda.sh && conda activate con && python3 $(pwd)/1-batch/batch.py 2 3'"

# 
# monitoring
# 

grep --color=always --extended-regexp 'free|$' /home/sladmitet/smon.txt
watch -n 0.5 "squeue -u $USER --states=R"
tail -f $(ls -v $(pwd)/*.err 2>/dev/null | tail -n 300)
tail -f $(ls -v $(pwd)/*.out 2>/dev/null | tail -n 300)

# read results
cat /scratch/$USER/self-ensembling/1-batch/resnet_0.jsonl
cat /scratch/$USER/self-ensembling/1-batch/resnet_1.jsonl
cat /scratch/$USER/self-ensembling/1-batch/resnet_2.jsonl

# cancel remaining
scancel --user=$USER --state=PENDING
squeue -u $USER --states=R | grep wrap | awk '{print $1}' | xargs scancel
