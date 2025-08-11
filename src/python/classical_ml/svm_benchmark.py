#!/usr/bin/env python3
"""
Enhanced Python SVM Benchmark Implementation

This module implements comprehensive SVM benchmarks using scikit-learn for comparison
with Rust implementations. Features include advanced algorithms, comprehensive monitoring,
and reproducible results.
"""

import argparse
import json
import time
import uuid
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC, NuSVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler
import psutil
import os
import sys
import platform
import subprocess
from dataclasses import asdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Prefer absolute package import to avoid sys.path hacks during tests
from src.shared.schemas.metrics import (
    BenchmarkResult, PerformanceMetrics, ResourceMetrics, QualityMetrics,
    HardwareConfig, Language, TaskType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedSVMBenchmark:
    """Enhanced SVM benchmark implementation with comprehensive monitoring."""
    
    def __init__(self, framework: str = "scikit-learn", enable_profiling: bool = True):
        self.framework = framework
        self.model = None
        self.scaler = StandardScaler()
        self.resource_monitor = EnhancedResourceMonitor()
        self.enable_profiling = enable_profiling
        self.profiling_data = {}
        
        # Set deterministic seeds for reproducibility
        np.random.seed(42)
        if hasattr(np.random, 'default_rng'):
            self.rng = np.random.default_rng(42)
        else:
            self.rng = np.random.RandomState(42)
    
    def load_dataset(self, dataset_name: str, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load a classification dataset with comprehensive error handling."""
        try:
            if dataset_name == "iris":
                data = load_iris()
                X, y = data.data, data.target
            elif dataset_name == "wine":
                data = load_wine()
                X, y = data.data, data.target
            elif dataset_name == "breast_cancer":
                data = load_breast_cancer()
                X, y = data.data, data.target
            elif dataset_name == "digits":
                from sklearn.datasets import load_digits
                data = load_digits()
                X, y = data.data, data.target
            elif dataset_name == "synthetic_classification":
                X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                                         n_redundant=5, n_classes=3, random_state=42)
            elif dataset_name == "adult":
                # Adult dataset simulation
                X, y = make_classification(n_samples=1000, n_features=14, n_informative=8,
                                         n_redundant=3, n_classes=2, random_state=42)
            elif dataset_name == "covertype":
                # Covertype dataset simulation
                X, y = make_classification(n_samples=1000, n_features=54, n_informative=20,
                                         n_redundant=15, n_classes=7, random_state=42)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Limit samples if specified
            if n_samples and n_samples < len(X):
                indices = self.rng.choice(len(X), n_samples, replace=False)
                X, y = X[indices], y[indices]
            
            logger.info(f"Loaded dataset {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def create_model(self, algorithm: str, hyperparams: Dict[str, Any]):
        """Create an SVM model with comprehensive algorithm support."""
        try:
            if algorithm == "svc":
                C = hyperparams.get("C", 1.0)
                kernel = hyperparams.get("kernel", "rbf")
                gamma = hyperparams.get("gamma", "scale")
                degree = hyperparams.get("degree", 3)
                coef0 = hyperparams.get("coef0", 0.0)
                probability = hyperparams.get("probability", True)
                random_state = hyperparams.get("random_state", 42)
                
                self.model = SVC(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    degree=degree,
                    coef0=coef0,
                    probability=probability,
                    random_state=random_state
                )
                
            elif algorithm == "linearsvc":
                C = hyperparams.get("C", 1.0)
                loss = hyperparams.get("loss", "squared_hinge")
                penalty = hyperparams.get("penalty", "l2")
                dual = hyperparams.get("dual", True)
                random_state = hyperparams.get("random_state", 42)
                
                self.model = LinearSVC(
                    C=C,
                    loss=loss,
                    penalty=penalty,
                    dual=dual,
                    random_state=random_state
                )
                
            elif algorithm == "nusvc":
                nu = hyperparams.get("nu", 0.5)
                kernel = hyperparams.get("kernel", "rbf")
                gamma = hyperparams.get("gamma", "scale")
                degree = hyperparams.get("degree", 3)
                coef0 = hyperparams.get("coef0", 0.0)
                probability = hyperparams.get("probability", True)
                random_state = hyperparams.get("random_state", 42)
                
                self.model = NuSVC(
                    nu=nu,
                    kernel=kernel,
                    gamma=gamma,
                    degree=degree,
                    coef0=coef0,
                    probability=probability,
                    random_state=random_state
                )
                
            elif algorithm == "svr":
                C = hyperparams.get("C", 1.0)
                kernel = hyperparams.get("kernel", "rbf")
                gamma = hyperparams.get("gamma", "scale")
                epsilon = hyperparams.get("epsilon", 0.1)
                degree = hyperparams.get("degree", 3)
                coef0 = hyperparams.get("coef0", 0.0)
                random_state = hyperparams.get("random_state", 42)
                
                self.model = SVR(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    epsilon=epsilon,
                    degree=degree,
                    coef0=coef0
                )
                
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            logger.info(f"Created {algorithm} model with hyperparameters: {hyperparams}")
            
        except Exception as e:
            logger.error(f"Failed to create model {algorithm}: {e}")
            raise
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data for SVM training."""
        try:
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            logger.info(f"Preprocessed data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            raise
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train the SVM model and return comprehensive metrics."""
        self.resource_monitor.start_monitoring()
        
        try:
            start_time = time.perf_counter()
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            training_time = time.perf_counter() - start_time
            resource_metrics = self.resource_monitor.stop_monitoring()
            
            # Get model complexity metrics
            n_support_vectors = len(self.model.support_vectors_) if hasattr(self.model, 'support_vectors_') else 0
            n_support = len(self.model.support_) if hasattr(self.model, 'support_') else 0
            
            return {
                "training_time_seconds": training_time,
                "n_support_vectors": n_support_vectors,
                "n_support": n_support,
                "resource_metrics": resource_metrics
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the SVM model with comprehensive metrics."""
        try:
            # Get predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
            
            # Calculate probability-based metrics if available
            if hasattr(self.model, 'predict_proba'):
                try:
                    y_proba = self.model.predict_proba(X_test)
                    
                    # For binary classification
                    if len(np.unique(y_test)) == 2:
                        metrics["auc_roc"] = roc_auc_score(y_test, y_proba[:, 1])
                        metrics["auc_pr"] = average_precision_score(y_test, y_proba[:, 1])
                    else:
                        # For multi-class, use one-vs-rest approach
                        metrics["auc_roc"] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                        metrics["auc_pr"] = average_precision_score(y_test, y_proba, average='weighted')
                except:
                    pass
            
            # Calculate cross-validation score
            try:
                cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='accuracy')
                metrics["cv_accuracy_mean"] = cv_scores.mean()
                metrics["cv_accuracy_std"] = cv_scores.std()
            except:
                pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_inference_benchmark(self, X_test: np.ndarray, batch_sizes: List[int]) -> Dict[str, Any]:
        """Run comprehensive inference benchmarks."""
        try:
            latencies = []
            throughputs = []
            
            for batch_size in batch_sizes:
                batch_latencies = []
                
                # Warm-up runs
                for _ in range(10):
                    if batch_size == 1:
                        sample = X_test[0:1]
                        start_time = time.perf_counter()
                        _ = self.model.predict(sample)
                        end_time = time.perf_counter()
                    else:
                        batch_indices = self.rng.choice(len(X_test), min(batch_size, len(X_test)), replace=False)
                        batch_data = X_test[batch_indices]
                        start_time = time.perf_counter()
                        _ = self.model.predict(batch_data)
                        end_time = time.perf_counter()
                
                # Benchmark runs
                for _ in range(100):
                    if batch_size == 1:
                        sample = X_test[0:1]
                        start_time = time.perf_counter()
                        _ = self.model.predict(sample)
                        end_time = time.perf_counter()
                    else:
                        batch_indices = self.rng.choice(len(X_test), min(batch_size, len(X_test)), replace=False)
                        batch_data = X_test[batch_indices]
                        start_time = time.perf_counter()
                        _ = self.model.predict(batch_data)
                        end_time = time.perf_counter()
                    
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    batch_latencies.append(latency)
                
                avg_latency = np.mean(batch_latencies)
                latencies.append(avg_latency)
                throughputs.append(batch_size / (avg_latency / 1000))  # samples per second
            
            # Calculate percentiles
            all_latencies = np.concatenate([latencies] * 100)  # Approximate
            p50 = np.percentile(all_latencies, 50)
            p95 = np.percentile(all_latencies, 95)
            p99 = np.percentile(all_latencies, 99)
            
            return {
                "inference_latency_ms": np.mean(latencies),
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
                "throughput_samples_per_second": np.mean(throughputs),
                "latency_std_ms": np.std(latencies)
            }
            
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            raise
    
    def get_hardware_config(self) -> HardwareConfig:
        """Get comprehensive hardware configuration."""
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent()
            
            # Memory information
            memory = psutil.virtual_memory()
            
            # System information
            system_info = platform.uname()
            
            # Try to get GPU information
            gpu_info = self._get_gpu_info()
            
            return HardwareConfig(
                cpu_model=system_info.processor,
                cpu_cores=cpu_count,
                cpu_threads=cpu_count,
                memory_gb=memory.total / (1024**3),
                gpu_model=gpu_info.get('model'),
                gpu_memory_gb=gpu_info.get('memory_gb')
            )
            
        except Exception as e:
            logger.warning(f"Failed to get hardware config: {e}")
            return HardwareConfig(
                cpu_model="Unknown",
                cpu_cores=1,
                cpu_threads=1,
                memory_gb=1.0
            )
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information if available."""
        try:
            # Try to get NVIDIA GPU info via nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(', ')
                    return {
                        'model': parts[0].strip(),
                        'memory_gb': float(parts[1]) / 1024 if len(parts) > 1 else None
                    }
        except:
            pass
        
        return {'model': None, 'memory_gb': None}
    
    def run_benchmark(self, 
                     dataset: str, 
                     algorithm: str, 
                     hyperparams: Dict[str, Any],
                     run_id: str,
                     mode: str = "training",
                     binary_base_class: Optional[int] = None) -> BenchmarkResult:
        """Run comprehensive benchmark with full analysis."""
        try:
            logger.info(f"Starting SVM benchmark: {dataset}, {algorithm}, {mode}")
            
            # Load dataset
            X, y = self.load_dataset(dataset)
            # Optional: map to binary one-vs-rest for parity (class == base -> 1 else 0)
            if binary_base_class is not None:
                y = (y == binary_base_class).astype(int)
            
            # Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
            
            # Create model
            self.create_model(algorithm, hyperparams)
            
            # Get hardware configuration
            hardware_config = self.get_hardware_config()
            
            if mode == "training":
                # Training benchmark
                training_results = self.train_model(X_train, y_train)
                evaluation_results = self.evaluate_model(X_test, y_test)
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.CLASSICAL_ML,
                    model_name=f"{algorithm}_svm",
                    dataset=dataset,
                    run_id=run_id,
                    timestamp=datetime.utcnow(),
                    hardware_config=hardware_config,
                    performance_metrics=PerformanceMetrics(
                        training_time_seconds=training_results["training_time_seconds"],
                        inference_latency_ms=None,
                        throughput_samples_per_second=None,
                        convergence_epochs=None
                    ),
                    resource_metrics=training_results["resource_metrics"],
                    quality_metrics=QualityMetrics(
                        accuracy=evaluation_results.get("accuracy"),
                        f1_score=evaluation_results.get("f1_score"),
                        precision=evaluation_results.get("precision"),
                        recall=evaluation_results.get("recall"),
                        loss=None
                    ),
                    metadata={
                        "algorithm": algorithm,
                        "hyperparameters": hyperparams,
                        "dataset_size": len(X),
                        "n_support_vectors": training_results.get("n_support_vectors"),
                        "n_support": training_results.get("n_support"),
                        "auc_roc": evaluation_results.get("auc_roc"),
                        "auc_pr": evaluation_results.get("auc_pr")
                    }
                )
                
            elif mode == "inference":
                # Train model first
                training_results = self.train_model(X_train, y_train)
                
                # Inference benchmark
                inference_metrics = self.run_inference_benchmark(X_test, [1, 10, 100])
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.CLASSICAL_ML,
                    model_name=f"{algorithm}_svm",
                    dataset=dataset,
                    run_id=run_id,
                    timestamp=datetime.utcnow(),
                    hardware_config=hardware_config,
                    performance_metrics=PerformanceMetrics(
                        training_time_seconds=None,
                        inference_latency_ms=inference_metrics.get("inference_latency_ms"),
                        throughput_samples_per_second=inference_metrics.get("throughput_samples_per_second"),
                        convergence_epochs=None
                    ),
                    resource_metrics=ResourceMetrics(
                        peak_memory_mb=0.0,
                        average_memory_mb=0.0,
                        cpu_utilization_percent=0.0,
                        peak_gpu_memory_mb=None,
                        average_gpu_memory_mb=None,
                        gpu_utilization_percent=None
                    ),
                    quality_metrics=QualityMetrics(
                        accuracy=None,
                        f1_score=None,
                        precision=None,
                        recall=None,
                        loss=None
                    ),
                    metadata={
                        "algorithm": algorithm,
                        "hyperparameters": hyperparams,
                        "dataset_size": len(X)
                    }
                )
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise


class EnhancedResourceMonitor:
    """Enhanced resource monitoring with comprehensive metrics."""
    
    def __init__(self):
        self.start_memory = None
        self.peak_memory = 0
        self.memory_samples = []
        self.cpu_samples = []
        self.start_time = None
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """Start comprehensive resource monitoring."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss
        self.peak_memory = self.start_memory
        self.memory_samples = [self.start_memory]
        self.cpu_samples = [psutil.cpu_percent()]
    
    def stop_monitoring(self) -> ResourceMetrics:
        """Stop monitoring and return comprehensive metrics."""
        end_memory = self.process.memory_info().rss
        end_cpu = psutil.cpu_percent()
        
        self.memory_samples.append(end_memory)
        self.cpu_samples.append(end_cpu)
        
        # Calculate comprehensive metrics
        peak_memory = max(self.memory_samples)
        avg_memory = sum(self.memory_samples) / len(self.memory_samples)
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
        
        # Try to get GPU metrics
        gpu_metrics = self._get_gpu_metrics()
        
        return ResourceMetrics(
            peak_memory_mb=peak_memory / (1024 * 1024),
            average_memory_mb=avg_memory / (1024 * 1024),
            cpu_utilization_percent=avg_cpu,
            peak_gpu_memory_mb=gpu_metrics.get('peak_memory_mb'),
            average_gpu_memory_mb=gpu_metrics.get('avg_memory_mb'),
            gpu_utilization_percent=gpu_metrics.get('utilization_percent')
        )
    
    def _get_gpu_metrics(self) -> Dict[str, Optional[float]]:
        """Get GPU metrics if available."""
        try:
            # Try to get NVIDIA GPU metrics
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(', ')
                    return {
                        'peak_memory_mb': float(parts[0]) if len(parts) > 0 else None,
                        'avg_memory_mb': float(parts[0]) if len(parts) > 0 else None,
                        'utilization_percent': float(parts[1]) if len(parts) > 1 else None
                    }
        except:
            pass
        
        return {'peak_memory_mb': None, 'avg_memory_mb': None, 'utilization_percent': None}


def main():
    """Main function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(description="Enhanced Python SVM Benchmark")
    parser.add_argument("--mode", default="training", choices=["training", "inference"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--hyperparams", default="{}", type=str)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--binary-base-class", type=int, default=None)
    
    args = parser.parse_args()
    
    try:
        # Parse hyperparameters
        hyperparams = json.loads(args.hyperparams)
        
        # Create benchmark instance
        benchmark = EnhancedSVMBenchmark(enable_profiling=args.enable_profiling)
        
        # Run benchmark
        result = benchmark.run_benchmark(
            args.dataset,
            args.algorithm,
            hyperparams,
            args.run_id,
            args.mode,
            args.binary_base_class
        )
        
        # Save results
        output_file = f"{args.dataset}_{args.algorithm}_{args.run_id}_{args.mode}_results.json"
        output_path = Path(args.output_dir) / output_file
        
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"SVM benchmark completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"SVM benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 