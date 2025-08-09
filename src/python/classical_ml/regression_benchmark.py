#!/usr/bin/env python3
"""
Enhanced Python Regression Benchmark Implementation

This module implements comprehensive regression benchmarks using scikit-learn for comparison
with Rust implementations. Features include advanced statistical analysis, comprehensive
resource monitoring, and reproducible results.
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing, make_regression, load_diabetes
from sklearn.preprocessing import StandardScaler
import psutil
import os
import sys
import platform
import subprocess
import hashlib
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


class EnhancedRegressionBenchmark:
    """Enhanced regression benchmark implementation with comprehensive monitoring."""
    
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
        """Load a regression dataset with comprehensive error handling."""
        try:
            if dataset_name == "boston_housing":
                # Fallback for deprecated dataset: use diabetes dataset as proxy
                data = load_diabetes()
                X, y = data.data, data.target
            elif dataset_name == "california_housing":
                data = fetch_california_housing()
                X, y = data.data, data.target
            elif dataset_name == "synthetic_linear":
                X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                                     random_state=42, noise=0.1)
            elif dataset_name == "synthetic_nonlinear":
                X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                                     random_state=42, noise=0.5)
            elif dataset_name == "synthetic_sparse":
                X, y = make_regression(n_samples=1000, n_features=50, n_informative=5, 
                                     random_state=42, noise=0.1)
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
        """Create a regression model with comprehensive hyperparameter support."""
        try:
            if algorithm == "linear":
                self.model = LinearRegression()
            elif algorithm == "ridge":
                alpha = hyperparams.get("alpha", 1.0)
                self.model = Ridge(alpha=alpha, random_state=42)
            elif algorithm == "lasso":
                alpha = hyperparams.get("alpha", 1.0)
                self.model = Lasso(alpha=alpha, random_state=42)
            elif algorithm == "elastic_net":
                alpha = hyperparams.get("alpha", 1.0)
                l1_ratio = hyperparams.get("l1_ratio", 0.5)
                self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            logger.info(f"Created {algorithm} model with hyperparameters: {hyperparams}")
            
        except Exception as e:
            logger.error(f"Failed to create model {algorithm}: {e}")
            raise
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data with scaling and train-test split."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            logger.info(f"Data preprocessed: train={X_train.shape}, test={X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            raise
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train the regression model with comprehensive monitoring."""
        self.resource_monitor.start_monitoring()
        
        try:
            # Cross-validation for model validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
            
            # Training
            start_time = time.perf_counter()
            self.model.fit(X_train, y_train)
            training_time = time.perf_counter() - start_time
            
            # Get model coefficients for analysis
            if hasattr(self.model, 'coef_'):
                n_nonzero = np.count_nonzero(self.model.coef_)
                sparsity = 1.0 - (n_nonzero / len(self.model.coef_))
            else:
                sparsity = 0.0
            
            resource_metrics = self.resource_monitor.stop_monitoring()
            
            return {
                "training_time_seconds": training_time,
                "resource_metrics": resource_metrics,
                "cv_r2_mean": cv_scores.mean(),
                "cv_r2_std": cv_scores.std(),
                "model_sparsity": sparsity,
                "n_nonzero_coefficients": n_nonzero if hasattr(self.model, 'coef_') else 0
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model with comprehensive metrics."""
        try:
            y_pred = self.model.predict(X_test)
            
            # Calculate comprehensive metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            explained_variance = np.var(y_pred) / np.var(y_test)
            
            # Calculate residuals statistics
            residuals = y_test - y_pred
            residual_std = np.std(residuals)
            residual_skew = self._calculate_skewness(residuals)
            residual_kurtosis = self._calculate_kurtosis(residuals)
            
            return {
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "mape": mape,
                "explained_variance": explained_variance,
                "residual_std": residual_std,
                "residual_skew": residual_skew,
                "residual_kurtosis": residual_kurtosis,
                "mse": mse
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of residuals."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of residuals."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def run_inference_benchmark(self, X_test: np.ndarray, batch_sizes: List[int]) -> Dict[str, Any]:
        """Run comprehensive inference benchmarks."""
        try:
            latencies = []
            throughputs = []
            
            for batch_size in batch_sizes:
                batch_latencies = []
                
                # Warm-up runs
                for _ in range(10):
                    _ = self.model.predict(X_test[:batch_size])
                
                # Benchmark runs
                for _ in range(100):
                    start_time = time.perf_counter()
                    _ = self.model.predict(X_test[:batch_size])
                    end_time = time.perf_counter()
                    batch_latencies.append((end_time - start_time) * 1000)  # Convert to ms
                
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
            # Try to get NVIDIA GPU info
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
                     mode: str = "training") -> BenchmarkResult:
        """Run comprehensive benchmark with full analysis."""
        try:
            logger.info(f"Starting benchmark: {dataset}, {algorithm}, {mode}")
            
            # Load and preprocess data
            X, y = self.load_dataset(dataset)
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
            
            # Create model
            self.create_model(algorithm, hyperparams)
            
            # Get hardware configuration
            hardware_config = self.get_hardware_config()
            
            if mode == "training":
                # Training benchmark
                training_results = self.train_model(X_train, y_train)
                quality_metrics = self.evaluate_model(X_test, y_test)
                
                # Combine quality metrics
                combined_quality = {**quality_metrics, **training_results}
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.CLASSICAL_ML,
                    model_name=f"{algorithm}_regression",
                    dataset=dataset,
                    run_id=run_id,
                    timestamp=datetime.utcnow(),
                    hardware_config=hardware_config,
                    performance_metrics=PerformanceMetrics(
                        training_time_seconds=training_results["training_time_seconds"],
                        inference_latency_ms=None,
                        throughput_samples_per_second=None,
                        convergence_epochs=None,
                        tokens_per_second=None
                    ),
                    resource_metrics=training_results["resource_metrics"],
                    quality_metrics=QualityMetrics(
                        accuracy=None,
                        loss=None,
                        f1_score=None,
                        precision=None,
                        recall=None,
                        rmse=combined_quality.get("rmse"),
                        mae=combined_quality.get("mae"),
                        r2_score=combined_quality.get("r2_score")
                    ),
                    metadata={
                        "algorithm": algorithm,
                        "hyperparameters": hyperparams,
                        "dataset_size": len(X),
                        "features": X.shape[1],
                        "cv_r2_mean": training_results.get("cv_r2_mean"),
                        "cv_r2_std": training_results.get("cv_r2_std"),
                        "model_sparsity": training_results.get("model_sparsity"),
                        "n_nonzero_coefficients": training_results.get("n_nonzero_coefficients")
                    }
                )
                
            elif mode == "inference":
                # Train model first
                self.train_model(X_train, y_train)
                
                # Inference benchmark
                inference_metrics = self.run_inference_benchmark(X_test, [1, 10, 100])
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.CLASSICAL_ML,
                    model_name=f"{algorithm}_regression",
                    dataset=dataset,
                    run_id=run_id,
                    timestamp=datetime.utcnow(),
                    hardware_config=hardware_config,
                    performance_metrics=PerformanceMetrics(
                        training_time_seconds=None,
                        inference_latency_ms=inference_metrics.get("inference_latency_ms"),
                        throughput_samples_per_second=inference_metrics.get("throughput_samples_per_second"),
                        convergence_epochs=None,
                        tokens_per_second=None
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
                        loss=None,
                        f1_score=None,
                        precision=None,
                        recall=None,
                        rmse=None,
                        mae=None,
                        r2_score=None
                    ),
                    metadata={
                        "algorithm": algorithm,
                        "hyperparameters": hyperparams,
                        "dataset_size": len(X),
                        "features": X.shape[1]
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
    parser = argparse.ArgumentParser(description="Enhanced Python Regression Benchmark")
    parser.add_argument("--mode", default="training", choices=["training", "inference"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--hyperparams", default="{}", type=str)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--n-samples", type=int, help="Number of samples to use")
    
    args = parser.parse_args()
    
    try:
        # Parse hyperparameters
        hyperparams = json.loads(args.hyperparams)
        
        # Create benchmark instance
        benchmark = EnhancedRegressionBenchmark(enable_profiling=args.enable_profiling)
        
        # Run benchmark
        result = benchmark.run_benchmark(
            args.dataset,
            args.algorithm,
            hyperparams,
            args.run_id,
            args.mode
        )
        
        # Save results
        output_file = f"{args.dataset}_{args.algorithm}_{args.run_id}_{args.mode}_results.json"
        output_path = Path(args.output_dir) / output_file
        
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"Benchmark completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 