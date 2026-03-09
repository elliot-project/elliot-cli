# Leonardo HPC — Environment Setup Guide

## 1. Setting Up an Account

Create your account at the CINECA UserDB portal:
[https://userdb.hpc.cineca.it/](https://userdb.hpc.cineca.it/)

Once you have set up your UserDB profile and obtained your CINECA account name, follow the steps below.

---

## 2. SSH Access (macOS)

### Install `step` CLI
```zsh
brew install step
```

> If Homebrew is not yet installed, run the following first:
> ```zsh
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> ```

### Bootstrap the CA
```zsh
step ca bootstrap \
  --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

### Generate an SSH Certificate
Replace `your.name@email.com` with the email you used when registering on UserDB:
```zsh
step ssh certificate your.name@email.com --provisioner cineca-hpc key_filename
```

### Configure SSH
Add the following to `~/.ssh/config` (create it if it doesn't exist via `nano ~/.ssh/config`):
```
Host leonardo
    HostName login07-ext.leonardo.cineca.it
    User your_cineca_username
    IdentityFile /Users/your_mac_username/.ssh/key_filename
```

Set correct permissions:
```zsh
chmod 600 ~/.ssh/config
```

You can now connect simply with:
```zsh
ssh leonardo
```

---

## 3. Setting Up the Python Environment on the Cluster

### Load Python module
```zsh
module purge
module load python/3.11.7
```

### Install `uv` via a bootstrap environment
Since the cluster does not have `uv` available by default and PyPI access is restricted outside a venv, first create a temporary environment to install `uv`:
```zsh
python -m venv elliot-env
source elliot-env/bin/activate
pip install uv
```

### Make `uv` permanently available
Copy the `uv` binary to `~/.local/bin` so it persists across all environments:
```zsh
mkdir -p ~/.local/bin
cp $HOME/elliot-env/bin/uv ~/.local/bin/uv
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```zsh
uv --version
```

### Install Python 3.12 via `uv`
The project requires Python 3.12, which is not available as a cluster module. `uv` can fetch it directly:
```zsh
uv python install 3.12
```

### Create the project environment
```zsh
uv venv -p 3.12 elliot-venv
source elliot-venv/bin/activate
```

---

## 4. Install the Project

### Clone the repository
```zsh
git clone https://github.com/elliot-project/elliot-cli.git
```

### Install in editable mode
```zsh
uv pip install -e ./elliot-cli
```

---

## 5. Set HuggingFace Cache Directory

Compute nodes have no internet access, so all models and datasets must be pre-downloaded. Set `HF_HOME` to point to your work storage:

```zsh
mkdir -p $WORK/hf_cache
echo 'export HF_HOME="$HOME/hf_cache"' >> ~/.bashrc
source ~/.bashrc
```

---

## 6. Running Evaluations

```zsh
# Run evaluations using a task group (recommended)
oellm schedule-eval \
    --models "microsoft/DialoGPT-medium,EleutherAI/pythia-160m" \
    --task_groups "open-sci-0.01"

# Or specify individual tasks
oellm schedule-eval \
    --models "EleutherAI/pythia-160m" \
    --tasks "hellaswag,mmlu" \
    --n_shot 5
```

---

## Additional Resources

Full instructions are also available here:
[https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Leonardo-Access-and-Usage-LAION-Open-Psi-open-sci-Ontocord-AI-openEuroLLM](https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Leonardo-Access-and-Usage-LAION-Open-Psi-open-sci-Ontocord-AI-openEuroLLM)
