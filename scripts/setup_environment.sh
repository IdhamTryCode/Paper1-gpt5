#!/bin/bash

# Rust vs Python ML Benchmark System - Environment Setup Script
# This script sets up the environment for running benchmarks

set -e

echo "Setting up Rust vs Python ML Benchmark environment..."

# Create necessary directories
mkdir -p data
mkdir -p results/{phase1_selection,phase2_implementation,phase3_experiment,phase4_benchmark,phase5_analysis,phase6_assessment}
mkdir -p logs

# Check if we're in a container
if [ -f /.dockerenv ]; then
    echo "Running in container environment"
    
    # Install additional dependencies if needed
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y python3-pip python3-dev
    fi
    
    # Install Python dependencies
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    fi
    
    # Build Rust benchmarks
    if [ -f Cargo.toml ]; then
        cargo build --release
    fi
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/app/src/python:/app/src/shared"
export RUST_LOG=info

# Create configuration files if they don't exist
if [ ! -f config/benchmarks.yaml ]; then
    echo "Creating default benchmark configuration..."
    cat > config/benchmarks.yaml << 'EOF'
benchmarks:
  classical_ml:
    regression:
      datasets: ["boston_housing", "california_housing"]
      algorithms: ["linear", "ridge", "lasso"]
      metrics: ["rmse", "mae", "r2_score"]
      hyperparameters:
        alpha: [0.1, 1.0, 10.0]
      repetitions: 10
EOF
fi

if [ ! -f config/frameworks.yaml ]; then
    echo "Creating default framework configuration..."
    cat > config/frameworks.yaml << 'EOF'
python:
  classical_ml:
    primary: "scikit-learn==1.3.2"
    dependencies:
      - "numpy==1.24.3"
      - "pandas==2.0.3"
      - "scipy==1.11.1"

rust:
  classical_ml:
    linfa:
      version: "0.7.0"
      features: ["all"]
    smartcore:
      version: "0.3.2"
      features: ["all"]
EOF
fi

if [ ! -f config/hardware.yaml ]; then
    echo "Creating default hardware configuration..."
    cat > config/hardware.yaml << 'EOF'
system:
  cpu:
    model: "Intel Core i9-13900K"
    cores: 24
    threads: 32
    frequency: "3.0 GHz"
    cache: "36 MB"
  
  memory:
    total: "64 GB"
    type: "DDR5"
    speed: "5600 MHz"
  
  gpu:
    model: "NVIDIA RTX 4090"
    memory: "24 GB GDDR6X"
    cuda_cores: 16384
    tensor_cores: 512
  
  storage:
    type: "NVMe SSD"
    capacity: "2 TB"
    read_speed: "7000 MB/s"
    write_speed: "5300 MB/s"

resource_limits:
  cpu_intensive:
    max_cpus: 8
    max_memory: "16 GB"
    max_time: "2 hours"
  
  gpu_training:
    max_cpus: 4
    max_memory: "32 GB"
    max_gpu_memory: "20 GB"
    max_time: "6 hours"
  
  memory_intensive:
    max_cpus: 2
    max_memory: "64 GB"
    max_time: "4 hours"
  
  llm_inference:
    max_cpus: 6
    max_memory: "48 GB"
    max_gpu_memory: "22 GB"
    max_time: "1 hour"

monitoring:
  sampling_interval: 0.1
  enable_cpu_monitoring: true
  enable_memory_monitoring: true
  enable_gpu_monitoring: true
  enable_energy_monitoring: false
  enable_network_monitoring: false
EOF
fi

# Create requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    echo "Creating Python requirements file..."
    cat > requirements.txt << 'EOF'
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
scikit-learn==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
psutil==5.9.5
pyyaml==6.0.1
click==8.1.7
tqdm==4.65.0
jupyter==1.0.0
notebook==7.0.0
EOF
fi

# Create Cargo.toml if it doesn't exist
if [ ! -f Cargo.toml ]; then
    echo "Creating Rust Cargo.toml..."
    cat > Cargo.toml << 'EOF'
[package]
name = "rust-ml-benchmark"
version = "0.1.0"
edition = "2021"

[dependencies]
linfa = { version = "0.7.0", features = ["all"] }
linfa-linear = "0.7.0"
ndarray = "0.15"
ndarray-rand = "0.14"
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.0", features = ["v4", "serde"] }
clap = { version = "4.0", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
log = "0.4"
env_logger = "0.10"
sysinfo = "0.29"

[dev-dependencies]
criterion = "0.4"
EOF
fi

# Create .gitignore
if [ ! -f .gitignore ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Rust
target/
Cargo.lock
*.pdb

# Benchmark results
results/
data/
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Nextflow
.nextflow.log*
work/
*.trace
EOF 