# Leonardo HPC — Access Guide

## 1. Setting Up an Account

Create your account at the CINECA UserDB portal:
[https://userdb.hpc.cineca.it/](https://userdb.hpc.cineca.it/)

Once you have set up your UserDB profile and obtained your CINECA account name, follow the steps below.

---

## 2. Install `step` CLI

### macOS (via Homebrew)

```zsh
brew install step
```

> If Homebrew is not yet installed, run the following first:
> ```zsh
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> ```

---

## 3. Bootstrap the CA

```zsh
step ca bootstrap \
  --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

---

## 4. Generate an SSH Certificate

Replace `your.name@email.com` with the email you used when registering on UserDB, and choose a name for your key file:

```zsh
step ssh certificate your.name@email.com --provisioner cineca-hpc key_filename
```

---

## 5. Connect to Leonardo

```zsh
ssh -i ~/.ssh/key_filename username@login07-ext.leonardo.cineca.it
```

Replace `username` with your CINECA account name.

---

## Additional Resources

Full instructions are also available here:
[https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Leonardo-Access-and-Usage-LAION-Open-Psi-open-sci-Ontocord-AI-openEuroLLM](https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Leonardo-Access-and-Usage-LAION-Open-Psi-open-sci-Ontocord-AI-openEuroLLM)
