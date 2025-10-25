# VerifAI

A toolkit for the formal design and analysis of systems that include artificial intelligence (AI) and machine learning (ML) components.

## Prerequisites
- Linux (tested on Ubuntu)
- Python 3.8+ (but < 4.0)
- Git
- (Optional) System packages needed for some ML/vision extras (e.g., `build-essential`, `python3-dev`, SDL libs for `pygame`, OpenCV dependencies)

## Quick Start
```bash
# 1. Create and populate the virtual environment
./setup_venv.sh

# 2. Activate the environment
source .venv/bin/activate
```

## Running the Data Augmentation Example
```bash
# Navigate to the example directory
cd examples/data_augmentation

# Terminal 1: start the falsifier
python falsifier.py

# Terminal 2: (in a new terminal, activate venv again) run the classifier
python classifier.py
```

## Installation Options

The project uses `flit` as the build backend and can be installed with optional extras:

```bash
# Install with example dependencies (TensorFlow, OpenCV, etc.)
pip install -e .[examples]

# Install with Bayesian optimization support
pip install -e .[bayesopt]

# Install with parallel execution support
pip install -e .[parallel]

# Install development dependencies
pip install -e .[dev]
```

## Notes
- The first installation may take a while due to heavy optional dependencies (TensorFlow, OpenCV, etc.)
- The project uses PEP 621 metadata format with `flit_core` as the build backend

## Project Structure
```
.
├── setup_venv.sh          # Environment setup script
├── pyproject.toml         # Project metadata and dependencies (PEP 621)
├── poetry.toml            # Poetry configuration
├── poetry.lock            # Locked dependency versions
├── LICENSE                # BSD-3-Clause license
├── tox.ini                # Test automation config
├── .readthedocs.yml       # ReadTheDocs configuration
├── src/                   # VerifAI source code
├── examples/              # Example applications
│   └── data_augmentation/ # Data augmentation example
├── tests/                 # Test suite
└── docs/                  # Documentation source
```

## License
BSD-3-Clause (see LICENSE file)

## Citation
If you publish work using VerifAI, please cite the original project. See the [documentation](https://verifai.readthedocs.io) for citation information.

---
