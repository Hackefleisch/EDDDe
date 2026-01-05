# EDDDe
Electron Density Derived Descriptors for molecular learning and property prediction

# Installation

Git repo
```bash
    git clone --recurse-submodules PROJECT_URL
```

For development install (editable mode):

```bash
    uv pip install -e .
```

For production install:

```bash
    uv pip install .
```

Download welQrate:
    https://vanderbilt.app.box.com/v/WelQrate-Datasets/folder/270082782052
    Download the folder and move it to data/qsar/


## Generating ElektroNN predictions
```bash
    python -m eddde.embed data.csv=data/qsar/AID1798_actives.csv experiment.output_dir=outputs/AID1798_actives
```


## Running SVM classification
```bash
    python -m eddde.svm \
    svm.actives_predictions="${actives_path}" \
    svm.inactives_predictions="${inactives_path}" \
    svm.output_dir="${svm_output_dir}" \
    experiment.name="svm_${dataset_name}"
```