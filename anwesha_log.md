# Creating Environment for Running MoCo V3 on HPRC

```bash
# Create a new Conda environment
conda create --name moco_v3 python=3.8 -y

# Activate the environment
conda activate moco_v3

# Install PyTorch 1.9.0 with CUDA 10.2 support
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install timm 0.4.9
pip install timm==0.4.9

# Install additional dependencies
pip install tensorboard
pip install setuptools==59.5.0
pip install six
pip install pyyaml

