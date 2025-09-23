# Setup instructions
Steps for running the notebooks in this repo, as well as experimenting with the dimex library I created. The [SPLIT](https://github.com/VarunBabbar/SPLIT-ICML/) library installation (detailed further) is required.

- **Windows** users must enable WSL (step #1)
- **Mac/Linux** → skip step #1



## 1. WSL (Windows Subsystem for Linux)
- Control Panel → Turn Windows features on or off → Make sure Windows Subsystem for Linux is enabled (may need to restart device)

## 2. Miniconda
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh
bash Miniconda3-latest-Linux-x86\_64.sh
source ~/.bashrc
```

## 3. Create the environment
```bash
conda env create -f environment.yml
conda activate dimex-env
```

## 4. Install system dependencies
```bash
conda install -c conda-forge libgcc-ng libstdcxx-ng
sudo apt update
sudo apt install -y build-essential cmake ninja-build
sudo apt install -y libtbb-dev pkg-config
sudo apt install -y libgmp-dev
```

## 5. Clone the SPLIT repository
```bash
git clone https://github.com/VarunBabbar/SPLIT-ICML.git
```

## 6. Copy the SPLIT-ICML folder to Linux
```bash
mkdir -p ~/projects
cp -r "/mnt/c/your/local/path/SPLIT-ICML" ~/projects/
cd ~/projects/SPLIT-ICML
pip install split/
```

## 7. Install SPLIT
```bash
pip install --upgrade pip
pip install split/
```

## 8. Verify installation
```bash
python -c "import split; print('SPLIT installed successfully')"
```

