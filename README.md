# EDDDe
Electron Density Derived Descriptors for molecular learning and property prediction

# Installation

EDDDe requires the ElektroNN package which is currently under development and only available for selected users.

## Git repo

```bash
    git clone git@github.com:Hackefleisch/EDDDe.git
```

## uv prep

Install uv if you haven't already.

Download python with version higher than 3.10, for example with
```bash
    uv python install 3.11
```

Create virtual env with
```bash
    cd /path/to/EDDDe
    uv venv
```

Activate virtual env with
```bash
    source .venv/bin/activate
```

## Final install

### For development install (editable mode):

```bash
    uv pip install -e .
```

### For production install:

```bash
    uv pip install .
```

# Download welQrate:
    https://vanderbilt.app.box.com/v/WelQrate-Datasets/folder/270082782052
    Download the folder and move it to data/qsar/
