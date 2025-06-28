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

## Usage

Run the main script to see available commands and options:

```bash
python main.py --help
```

The script caches many operations on disk to speed up processing.
You can clear by removing the `.cache` directory.

## Tests

Execute:

```bash
pytest -n auto
```