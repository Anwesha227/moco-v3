#!/bin/bash
## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

## NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=moco-vit-multi
#SBATCH --account=132716300188
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:1         # Request 2 GPUs (since each node has only 2)
#SBATCH --partition=gpu
#SBATCH --mem=50G
#SBATCH --output=slurm_log/%j_%x.log
#SBATCH --mail-type=END
#SBATCH --mail-user=anwesha.basu@tamu.edu

## ----------- Setup Environment Variables for Debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

## ----------- GPU Availability Check
echo "Job started on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOBID"
echo "Checking GPU availability..."
nvidia-smi

## ----------- Run Multi-GPU Training
echo "Running training on multiple GPUs..."
python main_moco_anwesha.py \
    -a vit_base \
    --optimizer=adamw --lr=1.5e-4 --weight-decay=0.1 \
    --epochs=300 --warmup-epochs=40 \
    --stop-grad-conv1 --moco-m-cos --moco-t=0.2 \
    --batch-size 256 \
    --dataset semi-aves \
    --workers 8

echo "Job completed at $(date)"
