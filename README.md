# diffusion-hub
Repository for training diffusion models for different applications



## 🛠️ Installation

This repository depends on the following external GitHub repository. Please make sure to clone and install them manually:

- https://github.com/bilalhsp/ml-utils.git 

You can install each of them using the following commands:

```bash
git clone <repo_url>
cd <repo_folder>
pip install -e .
```

Once the dependencies are installed, clone and install this repository:

```bash
git clone https://github.com/bilalhsp/diffusion-hub.git
cd diffusion-hub
pip install -e .
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```


## Creating the Conda Environment

Define the paths for the environment installation and Conda package cache.

```bash
# Path where the conda environment will be installed
ENV_PATH=/depot/jgmakin/data/bilal/env/diffusion

# Path where conda will cache downloaded packages
CACHE_PATH=/scratch/gilbreth/ahmedb/conda/pkgs
```

### Environment creation

```bash
# Use scratch space for conda package cache
export CONDA_PKGS_DIRS=$CACHE_PATH

# Create the environment
conda env create -f environment.yml -p $ENV_PATH

# Activate the environment
conda activate $ENV_PATH
```

### Notes

- Replace `ENV_PATH` with the desired location where the conda environment should be installed.
- Replace `CACHE_PATH` with a directory that has sufficient disk space (e.g., scratch storage on a cluster).
- `environment.yml` contains the Python version and all required dependencies.
- Using a custom `CONDA_PKGS_DIRS` prevents Conda from filling the home directory quota.

