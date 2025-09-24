# Setup instructions
Steps for running the notebooks in this repo, as well as experimenting with the dimex library I created.

- **Windows** users must enable WSL (step #1)
- **Mac/Linux** → skip step #1



## 1. WSL (Windows Subsystem for Linux)
Control Panel → Turn Windows features on or off → Make sure Windows Subsystem for Linux is enabled (may need to restart device) → Search `wsl` and launch it

## 2. Miniconda
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh
bash Miniconda3-latest-Linux-x86\_64.sh
source ~/.bashrc
```

## 3. Clone this repository
```bash
cd "/mnt/c/where/you/want/to/clone/this/repo/to"
git clone https://github.com/VarunBabbar/dimacs-externship.git
cd dimacs-externship
```

## 4. Create the environment
```bash
conda env create -f environment.yml
conda activate dimex-env
```

## 5. Install system dependencies
```bash
conda install -c conda-forge libgcc-ng libstdcxx-ng
sudo apt update
sudo apt install -y build-essential cmake ninja-build
sudo apt install -y libtbb-dev pkg-config
sudo apt install -y libgmp-dev
```

## 6. Clone the SPLIT repository
```bash
git clone https://github.com/VarunBabbar/SPLIT-ICML.git
```

## 7. Copy the SPLIT-ICML folder to Linux
```bash
mkdir -p ~/projects
cp -r "/mnt/c/your/local/path/SPLIT-ICML" ~/projects/
cd ~/projects/SPLIT-ICML
pip install split/
```

## 8. Install SPLIT
```bash
pip install --upgrade pip
pip install split/
```

## 9. Install Dimex
```bash
pip install -e .
```

## 10. Verify installations
```bash
python -c "import split; print('SPLIT installed successfully')"
python -c "import dimex; print('Dimex installed successfully')"
```

