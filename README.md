# Total Solar Eclipse Toolkit

Toolkit for aligning and processing total solar eclipse photographs.

## Installation

```bash
pip install tse-tools
```

## Usage

```bash
tse-tools --help
```

## Development

### Setting up

Create a virtual environment and install the package in editable mode with the
development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e . --group dev
```

This installs the runtime dependencies, the test tooling, and `black`, and puts
the `tse-tools` command on your `PATH` (pointing at your working tree).

### Running in development

```bash
tse-tools --help               # or: python -m eclipsetools.cli --help
```

### Formatting

```bash
black src tests
```

### Tests

```bash
pytest -n auto
```

The alignment tests depend on the binary asset `tests/images/eclipse_5ms.CR3`.