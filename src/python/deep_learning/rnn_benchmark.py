#!/usr/bin/env python3
"""
Enhanced Python RNN Benchmark Implementation

Provides sequence classification benchmarks using PyTorch RNN/GRU/LSTM models.
Aligned metrics and CLI with Rust `rnn_benchmark` and other Python benchmarks.
"""

import argparse
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
from pathlib import Path
import os
import psutil
import platform

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Prefer absolute package import; allow fallback for direct CLI
try:
    from src.shared.schemas.metrics import (
        BenchmarkResult, PerformanceMetrics, ResourceMetrics, QualityMetrics,
        HardwareConfig, Language, TaskType
    )
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.shared.schemas.metrics import (
        BenchmarkResult, PerformanceMetrics, ResourceMetrics, QualityMetrics,
        HardwareConfig, Language, TaskType
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequenceClassifier(nn.Module):
    def __init__(self, arch: str, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        if arch == "rnn":
            self.core = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif arch == "gru":
            self.core = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif arch == "lstm":
            self.core = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.core(x)
        last = out[:, -1, :]
        return self.fc(last)


class EnhancedResourceMonitor:
    def __init__(self):
        self._mem_samples = []
        self._cpu_samples = []

    def start(self):
        proc = psutil.Process(os.getpid())
        self._mem_samples = [proc.memory_info().rss]
        self._cpu_samples = [psutil.cpu_percent(interval=None)]

    def stop(self) -> ResourceMetrics:
        proc = psutil.Process(os.getpid())
        self._mem_samples.append(proc.memory_info().rss)
        self._cpu_samples.append(psutil.cpu_percent(interval=None))
        peak_mb = max(self._mem_samples) / (1024.0 * 1024.0)
        avg_mb = (sum(self._mem_samples) / len(self._mem_samples)) / (1024.0 * 1024.0)
        avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples)
        return ResourceMetrics(
            peak_memory_mb=peak_mb,
            average_memory_mb=avg_mb,
            cpu_utilization_percent=avg_cpu,
            peak_gpu_memory_mb=None,
            average_gpu_memory_mb=None,
            gpu_utilization_percent=None,
            energy_consumption_joules=None,
            network_io_mb=None,
        )


def generate_synthetic_sequence(n_samples: int, seq_len: int, input_size: int, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n_samples, seq_len, input_size)
    # Create linearly separable-ish labels from sum over features thresholded
    scores = X.sum(dim=(1, 2))
    thresholds = torch.linspace(scores.min(), scores.max(), num_classes + 1)[1:-1]
    y = torch.bucketize(scores, thresholds)
    return X, y


def get_hardware_config() -> HardwareConfig:
    try:
        import pynvml  # type: ignore
        gpu_model = None
        gpu_mem = None
        try:
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_model = pynvml.nvmlDeviceGetName(h).decode('utf-8')
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024.0 * 1024.0 * 1024.0)
        except Exception:
            pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        gpu_model = None
        gpu_mem = None

    mem = psutil.virtual_memory().total / (1024.0 * 1024.0 * 1024.0)
    return HardwareConfig(
        cpu_model=platform.processor() or platform.machine(),
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        cpu_threads=psutil.cpu_count() or 1,
        memory_gb=mem,
        gpu_model=gpu_model,
        gpu_memory_gb=gpu_mem,
    )


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # Data
    if args.dataset == "synthetic":
        X_train, y_train = generate_synthetic_sequence(1024, args.seq_len, args.input_size, args.num_classes)
        X_test, y_test = generate_synthetic_sequence(256, args.seq_len, args.input_size, args.num_classes)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    # Model
    model = SequenceClassifier(arch=args.architecture, input_size=args.input_size, hidden_size=args.hidden_size,
                               num_layers=args.num_layers, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    monitor = EnhancedResourceMonitor()
    monitor.start()
    t0 = time.time()

    if args.mode == "training":
        model.train()
        for _ in range(args.epochs):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    train_time = time.time() - t0
    resources = monitor.stop()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            test_loss += float(loss.item()) * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
    accuracy = correct / max(1, total)
    test_loss /= max(1, total)

    perf = PerformanceMetrics(
        training_time_seconds=train_time if args.mode == "training" else None,
        inference_latency_ms=None,
        throughput_samples_per_second=None,
        latency_p50_ms=None,
        latency_p95_ms=None,
        latency_p99_ms=None,
        tokens_per_second=None,
        convergence_epochs=args.epochs if args.mode == "training" else None,
    )
    quality = QualityMetrics(
        accuracy=accuracy,
        f1_score=None,
        precision=None,
        recall=None,
        loss=test_loss,
        rmse=None,
        mae=None,
        r2_score=None,
    )

    result = BenchmarkResult(
        framework="pytorch",
        language=Language.PYTHON,
        task_type=TaskType.DEEP_LEARNING,
        model_name=f"{args.architecture}_rnn",
        dataset=args.dataset,
        run_id=args.run_id or str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        hardware_config=get_hardware_config(),
        performance_metrics=perf,
        resource_metrics=resources,
        quality_metrics=quality,
        metadata={
            "seq_len": args.seq_len,
            "input_size": args.input_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RNN benchmark (PyTorch)")
    ap.add_argument("--mode", choices=["training", "inference"], default="training")
    ap.add_argument("--dataset", choices=["synthetic"], default="synthetic")
    ap.add_argument("--architecture", choices=["rnn", "gru", "lstm"], default="gru")
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--input-size", type=int, default=16)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=".")
    return ap.parse_args()


def main():
    args = parse_args()
    result = run_benchmark(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.dataset}_{args.architecture}_{args.run_id or 'run'}_{args.mode}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, default=lambda o: o.__dict__, indent=2)
    print(f"Benchmark completed. Results saved to: {out_file}")


if __name__ == "__main__":
    main()

