#!/usr/bin/env python3
"""
Enhanced Python Clustering Benchmark Implementation

This module implements comprehensive clustering benchmarks using scikit-learn for comparison
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
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


class EnhancedClusteringBenchmark:
    """Enhanced clustering benchmark implementation with comprehensive monitoring."""
    
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
    
    def load_dataset(self, dataset_name: str, n_samples: Optional[int] = None) -> np.ndarray:
        """Load a clustering dataset with comprehensive error handling."""
        try:
            if dataset_name == "iris":
                data = load_iris()
                X = data.data
            elif dataset_name == "wine":
                data = load_wine()
                X = data.data
            elif dataset_name == "breast_cancer":
                data = load_breast_cancer()
                X = data.data
            elif dataset_name == "synthetic_blobs":
                X, _ = make_blobs(n_samples=1000, n_features=2, centers=3, 
                                random_state=42, cluster_std=1.0)
            elif dataset_name == "synthetic_moons":
                X, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
            elif dataset_name == "synthetic_circles":
                X, _ = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
            elif dataset_name == "synthetic_3d":
                X, _ = make_blobs(n_samples=1000, n_features=3, centers=4, 
                                random_state=42, cluster_std=1.0)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Limit samples if specified
            if n_samples and n_samples < len(X):
                indices = self.rng.choice(len(X), n_samples, replace=False)
                X = X[indices]
            
            logger.info(f"Loaded dataset {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
            return X
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def create_model(self, algorithm: str, hyperparams: Dict[str, Any]):
        """Create a clustering model with comprehensive algorithm support."""
        try:
            if algorithm == "kmeans":
                n_clusters = hyperparams.get("n_clusters", 3)
                init = hyperparams.get("init", "k-means++")
                n_init = hyperparams.get("n_init", 10)
                max_iter = hyperparams.get("max_iter", 300)
                random_state = hyperparams.get("random_state", 42)
                
                self.model = KMeans(
                    n_clusters=n_clusters,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=random_state
                )
                
            elif algorithm == "dbscan":
                eps = hyperparams.get("eps", 0.5)
                min_samples = hyperparams.get("min_samples", 5)
                metric = hyperparams.get("metric", "euclidean")
                
                self.model = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric=metric
                )
                
            elif algorithm == "agglomerative":
                n_clusters = hyperparams.get("n_clusters", 3)
                linkage = hyperparams.get("linkage", "ward")
                affinity = hyperparams.get("affinity", "euclidean")
                
                self.model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    affinity=affinity
                )
                
            elif algorithm == "gaussian_mixture":
                n_components = hyperparams.get("n_components", 3)
                covariance_type = hyperparams.get("covariance_type", "full")
                max_iter = hyperparams.get("max_iter", 100)
                random_state = hyperparams.get("random_state", 42)
                
                self.model = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                    random_state=random_state
                )
                
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            logger.info(f"Created {algorithm} model with hyperparameters: {hyperparams}")
            
        except Exception as e:
            logger.error(f"Failed to create model {algorithm}: {e}")
            raise
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess data for clustering."""
        try:
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Preprocessed data: {X_scaled.shape}")
            return X_scaled
            
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            raise
    
    def train_model(self, X: np.ndarray) -> Dict[str, Any]:
        """Train the clustering model and return comprehensive metrics."""
        self.resource_monitor.start_monitoring()
        
        try:
            start_time = time.perf_counter()
            
            # Fit the model
            if hasattr(self.model, 'fit_predict'):
                labels = self.model.fit_predict(X)
            else:
                self.model.fit(X)
                labels = self.model.predict(X)
            
            training_time = time.perf_counter() - start_time
            resource_metrics = self.resource_monitor.stop_monitoring()
            
            # Calculate clustering metrics
            n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise points
            silhouette_avg = silhouette_score(X, labels) if n_clusters > 1 else 0.0
            calinski_harabasz = calinski_harabasz_score(X, labels) if n_clusters > 1 else 0.0
            davies_bouldin = davies_bouldin_score(X, labels) if n_clusters > 1 else 0.0
            
            # Get inertia for KMeans
            inertia = None
            if hasattr(self.model, 'inertia_'):
                inertia = self.model.inertia_
            
            return {
                "training_time_seconds": training_time,
                "n_clusters": n_clusters,
                "silhouette_score": silhouette_avg,
                "calinski_harabasz_score": calinski_harabasz,
                "davies_bouldin_score": davies_bouldin,
                "inertia": inertia,
                "resource_metrics": resource_metrics,
                "labels": labels
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, X: np.ndarray) -> Dict[str, float]:
        """Evaluate the clustering model with comprehensive metrics."""
        try:
            # Get predictions
            if hasattr(self.model, 'predict'):
                labels = self.model.predict(X)
            else:
                labels = self.model.labels_
            
            # Calculate evaluation metrics
            n_clusters = len(np.unique(labels[labels != -1]))
            
            metrics = {
                "n_clusters": n_clusters,
                "n_samples": len(X),
                "n_features": X.shape[1]
            }
            
            # Clustering quality metrics
            if n_clusters > 1:
                metrics["silhouette_score"] = silhouette_score(X, labels)
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
                metrics["davies_bouldin_score"] = davies_bouldin_score(X, labels)
            else:
                metrics["silhouette_score"] = 0.0
                metrics["calinski_harabasz_score"] = 0.0
                metrics["davies_bouldin_score"] = 0.0
            
            # Model-specific metrics
            if hasattr(self.model, 'inertia_'):
                metrics["inertia"] = self.model.inertia_
            
            if hasattr(self.model, 'converged_'):
                metrics["converged"] = self.model.converged_
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_inference_benchmark(self, X: np.ndarray, batch_sizes: List[int]) -> Dict[str, Any]:
        """Run comprehensive inference benchmarks."""
        try:
            latencies = []
            throughputs = []
            
            for batch_size in batch_sizes:
                batch_latencies = []
                
                # Warm-up runs
                for _ in range(10):
                    if batch_size == 1:
                        sample = X[0:1]
                        start_time = time.perf_counter()
                        _ = self.model.predict(sample)
                        end_time = time.perf_counter()
                    else:
                        batch_indices = self.rng.choice(len(X), min(batch_size, len(X)), replace=False)
                        batch_data = X[batch_indices]
                        start_time = time.perf_counter()
                        _ = self.model.predict(batch_data)
                        end_time = time.perf_counter()
                
                # Benchmark runs
                for _ in range(100):
                    if batch_size == 1:
                        sample = X[0:1]
                        start_time = time.perf_counter()
                        _ = self.model.predict(sample)
                        end_time = time.perf_counter()
                    else:
                        batch_indices = self.rng.choice(len(X), min(batch_size, len(X)), replace=False)
                        batch_data = X[batch_indices]
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
                     mode: str = "training") -> BenchmarkResult:
        """Run comprehensive benchmark with full analysis."""
        try:
            logger.info(f"Starting clustering benchmark: {dataset}, {algorithm}, {mode}")
            
            # Load dataset
            X = self.load_dataset(dataset)
            
            # Preprocess data
            X_scaled = self.preprocess_data(X)
            
            # Create model
            self.create_model(algorithm, hyperparams)
            
            # Get hardware configuration
            hardware_config = self.get_hardware_config()
            
            if mode == "training":
                # Training benchmark
                training_results = self.train_model(X_scaled)
                evaluation_results = self.evaluate_model(X_scaled)
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.CLASSICAL_ML,
                    model_name=f"{algorithm}_clustering",
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
                        accuracy=None,
                        loss=None,
                        f1_score=None,
                        precision=None,
                        recall=None,
                        silhouette_score=evaluation_results.get("silhouette_score"),
                        inertia=evaluation_results.get("inertia")
                    ),
                    metadata={
                        "algorithm": algorithm,
                        "hyperparameters": hyperparams,
                        "dataset_size": len(X),
                        "n_clusters": training_results.get("n_clusters"),
                        "n_features": X.shape[1],
                        "silhouette_score": training_results.get("silhouette_score"),
                        "calinski_harabasz_score": training_results.get("calinski_harabasz_score"),
                        "davies_bouldin_score": training_results.get("davies_bouldin_score")
                    }
                )
                
            elif mode == "inference":
                # Train model first
                training_results = self.train_model(X_scaled)
                
                # Inference benchmark
                inference_metrics = self.run_inference_benchmark(X_scaled, [1, 10, 100])
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.CLASSICAL_ML,
                    model_name=f"{algorithm}_clustering",
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
                        loss=None,
                        f1_score=None,
                        precision=None,
                        recall=None
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
    parser = argparse.ArgumentParser(description="Enhanced Python Clustering Benchmark")
    parser.add_argument("--mode", default="training", choices=["training", "inference"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--hyperparams", default="{}", type=str)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--enable-profiling", action="store_true")
    
    args = parser.parse_args()
    
    try:
        # Parse hyperparameters
        hyperparams = json.loads(args.hyperparams)
        
        # Create benchmark instance
        benchmark = EnhancedClusteringBenchmark(enable_profiling=args.enable_profiling)
        
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
        
        logger.info(f"Clustering benchmark completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Clustering benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 