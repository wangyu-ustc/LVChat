# export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_PORT=8100
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=1

torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS}\
    train_it_long.py \
    scripts/config_7b_stage4.py
    
