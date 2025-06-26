# Solar Eclipse Toolkit

Toolkit for aligning and processing total solar eclipse photographs.

## Setting up

Create and activate a conda environment using the provided `environment.yml` file. This file contains all the necessary
dependencies for the project.

```bash
conda env create -f environment.yml
conda activate eclipsetools
python main.py --help
```

## Tests

Execute:

```bash
pytest -n auto
```