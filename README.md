# VerifAI Environment Wrapper

This repository provides a convenience script and instructions for setting up a reproducible Python environment and running the VerifAI data augmentation example.

## Prerequisites
- Linux (tested on Ubuntu)
- Python 3.8+ (but < 4.0 as required by VerifAI)
- Git
- (Optional) System packages needed for some ML / vision extras (e.g., `build-essential`, `python3-dev`, SDL libs for `pygame`, OpenCV dependencies)

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
cd VerifAI/examples/data_augmentation

# Terminal 1: start the falsifier
python falsifier.py

# Terminal 2: (new shell, activate venv again) run the classifier
python classifier.py
```

## Notes
- The first run of `poetry install -E examples` may take a while because it resolves and installs heavy optional dependencies (TensorFlow, OpenCV, etc.). Subsequent runs use the lock file and are faster.

## Project Structure (Local Wrapper)
```
.
├── setup_venv.sh          # Environment + dependency bootstrap
├── pyproject.toml         # Project + dependency metadata
├── poetry.lock            # Resolved, locked dependency versions
├── src/                   # (Placeholder for local Python modules)
└── VerifAI/               # Cloned upstream VerifAI repository
```

## License
VerifAI itself is BSD-licensed (see its repository). This wrapper repo inherits that licensing intention; add a `LICENSE` file if distributing.

## Citation
If you publish work using VerifAI, cite the upstream project (see its README / documentation).

---
