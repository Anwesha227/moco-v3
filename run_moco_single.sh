#!/bin/bash
#SBATCH --get-user-env=L
#SBATCH --job-name=moco-vit-multi
#SBATCH --account=132716300188
#SBATCH --time=24:00:00
#SBATCH --nodes=1                 # Single node
#SBATCH --ntasks-per-node=1        # Two tasks (one per GPU)
#SBATCH --gres=gpu:a100:1          # Request 2 GPUs
#SBATCH --partition=gpu
#SBATCH --mem=50G
#SBATCH --output=slurm_log/%j_%x.log
#SBATCH --mail-type=END
#SBATCH --mail-user=anwesha.basu@tamu.edu

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Dynamically compute MASTER_PORT based on job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Compute WORLD_SIZE
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE=$WORLD_SIZE"

# Set MASTER_ADDR dynamically
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR=$MASTER_ADDR"

# Set RANK dynamically
export RANK=$SLURM_PROCID
echo "RANK=$RANK"

# Ensure correct GPU visibility
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


# GPU availability check
echo "Job started on $(hostname) at $(date)"
nvidia-smi


# Run training script directly 
python main_moco_anwesha.py \
    -a vit_base \
    --optimizer adamw \
    --lr 1.5e-4 \
    --weight-decay 0.1 \
    --epochs 5 \
    --warmup-epochs 40 \
    --stop-grad-conv1 \
    --moco-m-cos \
    --moco-t=0.2 \
    --batch-size 32 \
    --dataset semi-aves \
    --multiprocessing-distributed \
    --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
    --world-size $WORLD_SIZE \
    --rank $RANK

echo "Job completed at $(date)"
