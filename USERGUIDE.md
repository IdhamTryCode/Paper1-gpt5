## User Guide

### 1) Prerequisites
- Python 3.11 (recommended) with pip
- Rust toolchain (stable, MSVC) via rustup
- PowerShell on Windows (this guide uses PowerShell commands)
- Optional: NVIDIA GPU + CUDA drivers (for Deep Learning CUDA runs)

### 2) Clone and create a Python virtual environment
```powershell
# From your working directory
git clone <repo-url> Paper1-gpt5
cd Paper1-gpt5

# Create and activate venv (Windows)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# Install Python deps
pip install -r requirements.txt
```

### 3) Enable CUDA for Python (optional but recommended for DL)
```powershell
# Install PyTorch CUDA 12.1 wheels (matches Rust torch bindings in this project)
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Verify GPU (single line)
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

### 4) Running Classical ML benchmarks
Outputs are written under `smoke_results/` (Python) and `smoke_results_rust/` (Rust).

- Python (examples; use more challenging datasets to avoid saturated scores)
```powershell
$env:PYTHONPATH='.'
# Clustering (wine)
python src\python\classical_ml\clustering_benchmark.py --mode training --dataset wine --algorithm kmeans --run-id py_kmeans --output-dir smoke_results

# Regression (synthetic_nonlinear)
python src\python\classical_ml\regression_benchmark.py --mode training --dataset synthetic_nonlinear --algorithm linear --run-id py_reg --output-dir smoke_results

# SVM (wine)
python src\python\classical_ml\svm_benchmark.py --mode training --dataset wine --algorithm svc --run-id py_svm --output-dir smoke_results --binary-base-class 0
```

- Rust (examples)
```powershell
# Build individual benches
cargo build -p clustering_benchmark -p regression_benchmark -p svm_benchmark

# Clustering (wine)
.\target\debug\clustering_benchmark.exe --mode training --dataset wine --algorithm kmeans --run-id rs_kmeans --output-dir smoke_results_rust

# Regression (synthetic_nonlinear)
.\target\debug\regression_benchmark.exe --mode training --dataset synthetic_nonlinear --algorithm linear --run-id rs_reg --output-dir smoke_results_rust

# SVM (wine; binary mapping applied internally)
.\target\debug\svm_benchmark.exe --mode training --dataset wine --algorithm svc --run-id rs_svm --output-dir smoke_results_rust
```

### 5) Running Deep Learning benchmarks
There are two layers: CNN and RNN, in Python and Rust. Results go to `smoke_results/` (Python) and `smoke_results_rust/` (Rust).

#### 5.1 Python DL (CUDA automatically used if available)
```powershell
$env:PYTHONPATH='.'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'   # deterministic + CUDA

# CNN (MNIST, LeNet)
python src\python\deep_learning\cnn_benchmark.py --mode training --dataset mnist --architecture lenet --hyperparams "{}" --run-id py_cnn --output-dir smoke_results

# RNN (Synthetic, GRU)
python src\python\deep_learning\rnn_benchmark.py --mode training --dataset synthetic --architecture gru --epochs 2 --batch-size 32 --learning-rate 0.001 --run-id py_rnn --output-dir smoke_results
```

#### 5.2 Rust DL (CUDA)
Recommended path is to link against the CUDA-enabled PyTorch installed in your venv to avoid manual libtorch management.
```powershell
# Activate venv so we can reuse its CUDA Torch DLLs
. .\.venv\Scripts\Activate.ps1

# Tell torch-sys/tch to use Python's installed torch
$env:LIBTORCH_USE_PYTORCH='1'
# Add venv torch DLLs to PATH for runtime
$torchlib=(Resolve-Path .\.venv\Lib\site-packages\torch\lib).Path
$env:Path=$torchlib+';'+$env:Path

# Build Rust DL benches (with venv active)
cargo build -p cnn_benchmark -p rnn_benchmark

# CNN (MNIST, LeNet)
.\target\debug\cnn_benchmark.exe --mode training --dataset mnist --architecture lenet --run-id rs_cnn --output-dir smoke_results_rust

# RNN (Synthetic, GRU)
.\target\debug\rnn_benchmark.exe --mode training --dataset synthetic --architecture gru --run-id rs_rnn --output-dir smoke_results_rust
```

Notes:
- Jika Anda ingin memakai libtorch terpisah (bukan dari venv), unduh libtorch 2.1.0+cu121 (Windows) dan set:
  ```powershell
  $env:LIBTORCH=\path\to\libtorch
  $env:Path=(Join-Path $env:LIBTORCH 'bin')+';'+$env:Path
  ```
  Lalu `cargo build` dan jalankan seperti di atas.

### 6) One-command end-to-end run (Python vs Rust, Classical ML + DL) + Visualizations
Script PowerShell berikut menjalankan seluruh benchmark (Python dan Rust, Classical ML dan DL), membuat CSV perbandingan gabungan, menyusun hasil untuk analisis statistik, lalu menghasilkan visualisasi terpisah per task (classical_ml dan deep_learning).
```powershell
# From repo root, with venv activated and (optional) CUDA set up as above
powershell -ExecutionPolicy Bypass -File scripts\run_all_benchmarks.ps1 -Cuda
```
Hasil:
- Python: `smoke_results/`
- Rust: `smoke_results_rust/`
- CSV gabungan: `smoke_results/comparison_dl_and_classical.csv`
- JSON gabungan (untuk analisis): `smoke_results/comparison_results.json`
- Hasil analisis statistik: `smoke_results/stats_results.json`
- Visualisasi (PNG): `plots/`
  - `performance_comparison_classical_ml.png`, `quality_metrics_comparison_classical_ml.png`, `resource_usage_comparison_classical_ml.png`, `effect_size_heatmap_classical_ml.png`
  - `performance_comparison_deep_learning.png`, `quality_metrics_comparison_deep_learning.png`, `resource_usage_comparison_deep_learning.png`, `effect_size_heatmap_deep_learning.png`

### 7) Output folders & files
- Python results: `smoke_results/`
- Rust results: `smoke_results_rust/`
- Comparison CSV: `smoke_results/comparison_dl_and_classical.csv`
- Stats & Viz inputs: `smoke_results/comparison_results.json`, `smoke_results/stats_results.json`
- Plots: `plots/` (see filenames above)

### 8) Git ignore: libtorch artifacts
Tambahkan ke `.gitignore` agar repo tidak bengkak oleh file binary:
```
libtorch/
libtorch_full/
```
Jika direktori tersebut sudah terlanjur muncul di git status sebagai perubahan, jalankan:
```powershell
# stop tracking existing paths
git rm -r --cached libtorch libtorch_full
# commit perubahan .gitignore
git add .gitignore
git commit -m "chore: ignore libtorch artifacts"
```

### 9) Troubleshooting
- CUDA tidak terdeteksi di Python: pastikan memasang wheel `torch==2.1.0+cu121` dan GPU driver terbaru.
- Rust DL error STATUS_DLL_NOT_FOUND: pastikan PATH mengarah ke `...\torch\lib` (dari venv) atau `libtorch\bin`.
- MNIST/CIFAR gagal ditemukan (Rust): unduh dataset via torchvision terlebih dahulu (Python) sehingga folder `data/` terbuat.
- Deterministik CUDA (cuBLAS): gunakan `CUBLAS_WORKSPACE_CONFIG=':4096:8'` sebelum menjalankan Python DL.

