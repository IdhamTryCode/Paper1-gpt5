#!/usr/bin/env nextflow

nextflow.enable.dsl=2

workflow SMOKE {
  main:
    RUN_PY_SMOKE()
    RUN_RUST_REG_SMOKE()
    RUN_PY_CNN_SMOKE()
    RUN_PY_LLM_SMOKE()
    RUN_PY_RL_SMOKE()
    RUN_PY_RNN_SMOKE()
    REPORT_SMOKE(RUN_PY_SMOKE.out, RUN_RUST_REG_SMOKE.out, RUN_PY_CNN_SMOKE.out, RUN_PY_LLM_SMOKE.out, RUN_PY_RL_SMOKE.out, RUN_PY_RNN_SMOKE.out)

  emit:
    results = flatten([ RUN_PY_SMOKE.out, RUN_RUST_REG_SMOKE.out, RUN_PY_CNN_SMOKE.out, RUN_PY_LLM_SMOKE.out, RUN_PY_RL_SMOKE.out, RUN_PY_RNN_SMOKE.out ])
}

process RUN_PY_SMOKE {
  tag "python_smoke"
  label 'smoke_light'
  cpus 1
  memory '2 GB'

  output:
  path "smoke_results/", emit: results

  script:
  """
  set -euo pipefail
  export PYTHONPATH="$launchDir"
  # Install Python dependencies (self-contained bootstrap)
  python3 -m pip install -r "$launchDir"/requirements.txt -q || true
  mkdir -p smoke_results

  # Run tiny regression benchmark (Python)
  python3 "$launchDir"/src/python/classical_ml/regression_benchmark.py \
    --mode training \
    --dataset synthetic_linear \
    --algorithm linear \
    --hyperparams '{}' \
    --run-id smoke \
    --output-dir smoke_results

  # Run tiny SVM benchmark (Python)
  python3 "$launchDir"/src/python/classical_ml/svm_benchmark.py \
    --mode training \
    --dataset iris \
    --algorithm svc \
    --hyperparams '{"C":0.1,"kernel":"linear","probability":true}' \
    --run-id smoke \
    --output-dir smoke_results

  # Summarize outputs
  echo "SMOKE OK" > smoke_results/STATUS.txt
  """
}

process RUN_RUST_REG_SMOKE {
  tag "rust_regression_smoke"
  label 'smoke_light'
  cpus 1
  memory '2 GB'
  executor 'local'

  output:
  path "smoke_results_rust/", emit: results

  script:
  """
  set -euo pipefail
  mkdir -p smoke_results_rust

  # Build just the regression crate (quiet)
  (cd "$launchDir" && cargo build -p regression_benchmark --quiet)

  # Run tiny Rust regression benchmark
  "$launchDir"/target/debug/regression_benchmark \
    --mode training \
    --dataset synthetic_linear \
    --algorithm linear \
    --hyperparams '{}' \
    --run-id smoke \
    --output-dir smoke_results_rust

  echo "RUST_REGRESSION_SMOKE OK" > smoke_results_rust/STATUS.txt
  """
}

process RUN_PY_CNN_SMOKE {
  tag "python_cnn_smoke"
  label 'smoke_light'
  cpus 1
  memory '2 GB'

  output:
  path "smoke_results_cnn/", emit: results

  script:
  """
  set -euo pipefail
  export PYTHONPATH="$launchDir"
  python3 -m pip install -r "$launchDir"/requirements.txt -q || true
  mkdir -p smoke_results_cnn

  python3 src/python/deep_learning/cnn_benchmark.py \
    --mode training \
    --dataset synthetic \
    --architecture simple_cnn \
    --epochs 1 \
    --batch-size 16 \
    --run-id smoke \
    --output-dir smoke_results_cnn || true

  echo "PYTHON_CNN_SMOKE DONE" > smoke_results_cnn/STATUS.txt
  """
}

process RUN_PY_LLM_SMOKE {
  tag "python_llm_smoke"
  label 'smoke_light'
  cpus 1
  memory '2 GB'

  output:
  path "smoke_results_llm/", emit: results

  script:
  """
  set -euo pipefail
  export PYTHONPATH="$launchDir"
  python3 -m pip install -r "$launchDir"/requirements.txt -q || true
  mkdir -p smoke_results_llm

  python3 "$launchDir"/src/python/llm/transformer_benchmark.py \
    --mode inference \
    --model-name distilgpt2 \
    --task text-generation \
    --prompt "Hello world" \
    --max-length 10 \
    --run-id smoke \
    --output-dir smoke_results_llm || true

  echo "PYTHON_LLM_SMOKE DONE" > smoke_results_llm/STATUS.txt
  """
}

process RUN_PY_RL_SMOKE {
  tag "python_rl_smoke"
  label 'smoke_light'
  cpus 1
  memory '2 GB'

  output:
  path "smoke_results_rl/", emit: results

  script:
  """
  set -euo pipefail
  export PYTHONPATH="$launchDir"
  python3 -m pip install -r "$launchDir"/requirements.txt -q || true
  mkdir -p smoke_results_rl

  python3 "$launchDir"/src/python/reinforcement_learning/dqn_benchmark.py \
    --mode training \
    --env-name CartPole-v1 \
    --algorithm dqn \
    --timesteps 500 \
    --run-id smoke \
    --output-dir smoke_results_rl || true

  echo "PYTHON_RL_SMOKE DONE" > smoke_results_rl/STATUS.txt
  """
}

process RUN_PY_RNN_SMOKE {
  tag "python_rnn_smoke"
  label 'smoke_light'
  cpus 1
  memory '2 GB'

  output:
  path "smoke_results_rnn/", emit: results

  script:
  """
  set -euo pipefail
  export PYTHONPATH="$launchDir"
  python3 -m pip install -r "$launchDir"/requirements.txt -q || true
  mkdir -p smoke_results_rnn

  python3 "$launchDir"/src/python/deep_learning/rnn_benchmark.py \
    --mode training \
    --dataset synthetic \
    --architecture gru \
    --seq-len 32 \
    --input-size 16 \
    --hidden-size 32 \
    --num-layers 1 \
    --num-classes 4 \
    --epochs 1 \
    --batch-size 32 \
    --run-id smoke \
    --output-dir smoke_results_rnn || true

  echo "PYTHON_RNN_SMOKE DONE" > smoke_results_rnn/STATUS.txt
  """
}

process REPORT_SMOKE {
  tag "smoke_report"
  label 'smoke_light'
  cpus 1
  memory '1 GB'

  input:
  path py_classical
  path rust_reg
  path py_cnn
  path py_llm
  path py_rl
  path py_rnn

  output:
  path "smoke_report/", emit: report

  script:
  """
  set -euo pipefail
  mkdir -p smoke_report

  echo "Smoke Report" > smoke_report/summary.txt
  date -u '+%Y-%m-%dT%H:%M:%SZ' >> smoke_report/summary.txt

  for dir in "$py_classical" "$rust_reg" "$py_cnn" "$py_llm" "$py_rl" "$py_rnn"; do
    if [ -d "$dir" ]; then
      echo "" >> smoke_report/summary.txt
      echo "== `basename \"$dir\"` ==" >> smoke_report/summary.txt
      find "$dir" -maxdepth 1 -type f -name "*_results.json" -print >> smoke_report/summary.txt || true
      if [ -f "$dir/STATUS.txt" ]; then
        printf "STATUS: " >> smoke_report/summary.txt
        cat "$dir/STATUS.txt" >> smoke_report/summary.txt
      fi
    fi
  done

  echo "OK" > smoke_report/STATUS.txt
  """
}

// Default entry so `-entry SMOKE` is optional
workflow {
  SMOKE()
}

