# Solar Eclipse Toolkit

Toolkit for aligning and processing total solar eclipse photographs.

## Installation

```bash
pip install eclipsetools
```

## Usage

```bash
eclipsetools --help
```

## Development

**TODO: Update development instructions**

### Setting up

Create and activate a conda environment using the provided `environment.yml` file. This file contains all the necessary
dependencies for the project.

```bash
conda env create -f environment.yml
conda activate eclipsetools
python -m eclipsetools.cli --help
```

### Running in development

Run the main script to see available commands and options:

```bash
python -m eclipsetools.cli --help
```

### Tests

Execute:

```bash
pytest -n auto
```