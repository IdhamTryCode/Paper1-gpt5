## User Guide

### Prerequisites
- **Python**: 3.9+
- **Rust**: 1.70+
- **Nextflow**: 22.10+

### Setup (recommended virtual environment)
```bash
cd /Users/rismanadnan/RAM-Papers/Paper1

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Run the smoke workflow
```bash
# From the project root
nextflow run workflows/smoke.nf
```

### If a task fails due to missing Python deps (classical ML)
The Python classical ML smoke requires local Python dependencies. After creating a venv and running `pip install -r requirements.txt` as above, re-run with resume:
```bash
nextflow run workflows/smoke.nf -resume
```

### Current smoke status
- **CNN (DL)**: green
- **LLM**: green
- **RL**: green
- **RNN (DL)**: green
- **Python Classical ML**: requires venv + `pip install -r requirements.txt`; then `-resume`

### Troubleshooting
- If Nextflow isn’t found: install via `brew install nextflow` (macOS) or see Nextflow docs.
- If `pip install` fails on Apple Silicon: ensure a recent Python and consider `pip install --upgrade pip setuptools wheel` first.
- To exit venv: `deactivate`.

