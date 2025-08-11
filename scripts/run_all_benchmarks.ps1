param(
  [switch]$Cuda = $true
)

$ErrorActionPreference = 'Stop'

function Activate-Venv {
  if (Test-Path .\.venv\Scripts\Activate.ps1) { . .\.venv\Scripts\Activate.ps1 }
}

function Ensure-PythonCUDA {
  if ($Cuda) {
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available());\nprint(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')" | Write-Host
  }
}

function Run-Python-ClassicalML {
  $env:PYTHONPATH='.'
  python src\python\classical_ml\clustering_benchmark.py --mode training --dataset wine --algorithm kmeans --run-id py_kmeans --output-dir smoke_results
  python src\python\classical_ml\regression_benchmark.py --mode training --dataset synthetic_nonlinear --algorithm linear --run-id py_reg --output-dir smoke_results
  python src\python\classical_ml\svm_benchmark.py --mode training --dataset wine --algorithm svc --run-id py_svm --output-dir smoke_results --binary-base-class 0
}

function Run-Rust-ClassicalML {
  cargo build -p clustering_benchmark -p regression_benchmark -p svm_benchmark | Out-Null
  .\target\debug\clustering_benchmark.exe --mode training --dataset wine --algorithm kmeans --run-id rs_kmeans --output-dir smoke_results_rust
  .\target\debug\regression_benchmark.exe --mode training --dataset synthetic_nonlinear --algorithm linear --run-id rs_reg --output-dir smoke_results_rust
  .\target\debug\svm_benchmark.exe --mode training --dataset wine --algorithm svc --run-id rs_svm --output-dir smoke_results_rust
}

function Run-Python-DL {
  $env:PYTHONPATH='.'
  if ($Cuda) { $env:CUBLAS_WORKSPACE_CONFIG=':4096:8' }
  python src\python\deep_learning\cnn_benchmark.py --mode training --dataset mnist --architecture lenet --hyperparams '{}' --run-id py_cnn --output-dir smoke_results
  python src\python\deep_learning\rnn_benchmark.py --mode training --dataset synthetic --architecture gru --epochs 2 --batch-size 32 --learning-rate 0.001 --run-id py_rnn --output-dir smoke_results
}

function Run-Rust-DL {
  if ($Cuda) {
    $env:LIBTORCH_USE_PYTORCH='1'
    $torchlib=(Resolve-Path .\.venv\Lib\site-packages\torch\lib).Path
    $env:Path=$torchlib+';'+$env:Path
  }
  cargo build -p cnn_benchmark -p rnn_benchmark | Out-Null
  .\target\debug\cnn_benchmark.exe --mode training --dataset mnist --architecture lenet --run-id rs_cnn --output-dir smoke_results_rust
  .\target\debug\rnn_benchmark.exe --mode training --dataset synthetic --architecture gru --run-id rs_rnn --output-dir smoke_results_rust
}

function Compare-All {
  python scripts\compare_benchmarks.py smoke_results smoke_results_rust smoke_results\comparison_dl_and_classical.csv
}

function Analyze-And-Visualize {
  python scripts\assemble_comparison_results.py smoke_results smoke_results_rust smoke_results\comparison_results.json
  python scripts\perform_statistical_analysis.py --results smoke_results\comparison_results.json --output smoke_results\stats_results.json
  python scripts\create_visualizations.py --statistical-results smoke_results\stats_results.json --output-visualizations smoke_results\visualizations.json --output-plots plots --split-by-task
}

# Main
Activate-Venv
Ensure-PythonCUDA

Write-Host 'Running Python Classical ML...'
Run-Python-ClassicalML
Write-Host 'Running Rust Classical ML...'
Run-Rust-ClassicalML

Write-Host 'Running Python Deep Learning...'
Run-Python-DL
Write-Host 'Running Rust Deep Learning...'
Run-Rust-DL

Write-Host 'Comparing results...'
Compare-All

Write-Host 'Generating analysis and visualizations...'
Analyze-And-Visualize

Write-Host 'Done. Consolidated CSV: smoke_results\comparison_dl_and_classical.csv'
Write-Host 'Plots written to: plots'


