#!/usr/bin/env python3
"""
Enhanced Python CNN Benchmark Implementation

This module implements comprehensive CNN benchmarks using PyTorch for comparison
with Rust implementations. Features include advanced architectures, comprehensive
monitoring, and reproducible results.
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
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

# Import CNN models via absolute package path
from src.python.deep_learning.cnn_models import (
    ResNet18, VGG16, MobileNet, EnhancedLeNet, EnhancedSimpleCNN, AttentionCNN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedCNNBenchmark:
    """Enhanced CNN benchmark implementation with comprehensive monitoring."""
    
    def __init__(self, framework: str = "pytorch", enable_profiling: bool = True):
        self.framework = framework
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.resource_monitor = EnhancedResourceMonitor()
        self.enable_profiling = enable_profiling
        self.profiling_data = {}
        
        # Set deterministic seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        
        logger.info(f"Initialized CNN benchmark on device: {self.device}")
    
    def load_dataset(self, dataset_name: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Load a CNN dataset with comprehensive error handling."""
        try:
            if dataset_name == "mnist":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
                train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
                test_dataset = datasets.MNIST('./data', train=False, transform=transform)
                
            elif dataset_name == "cifar10":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
                
                train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
                test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
                
            elif dataset_name == "cifar100":
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
                
                train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
                test_dataset = datasets.CIFAR100('./data', train=False, transform=transform)
                
            elif dataset_name == "synthetic":
                # Generate synthetic image data
                n_samples = 1000
                n_channels = 3
                height, width = 32, 32
                
                X_train = torch.randn(n_samples, n_channels, height, width)
                y_train = torch.randint(0, 10, (n_samples,))
                X_test = torch.randn(200, n_channels, height, width)
                y_test = torch.randint(0, 10, (200,))
                
                train_dataset = TensorDataset(X_train, y_train)
                test_dataset = TensorDataset(X_test, y_test)
                
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            logger.info(f"Loaded dataset {dataset_name}: {len(train_dataset)} train, {len(test_dataset)} test samples")
            return train_loader, test_loader
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def create_model(self, architecture: str, num_classes: int = 10, **kwargs) -> nn.Module:
        """Create CNN model with comprehensive architecture support."""
        try:
            if architecture == "resnet18":
                pretrained = kwargs.get("pretrained", False)
                self.model = ResNet18(num_classes=num_classes, pretrained=pretrained)
                
            elif architecture == "vgg16":
                pretrained = kwargs.get("pretrained", False)
                self.model = VGG16(num_classes=num_classes, pretrained=pretrained)
                
            elif architecture == "mobilenet":
                pretrained = kwargs.get("pretrained", False)
                self.model = MobileNet(num_classes=num_classes, pretrained=pretrained)
                
            elif architecture == "lenet":
                self.model = EnhancedLeNet(num_classes=num_classes)
                
            elif architecture == "simple_cnn":
                self.model = EnhancedSimpleCNN(num_classes=num_classes)
                
            elif architecture == "attention_cnn":
                self.model = AttentionCNN(num_classes=num_classes)
                
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Created {architecture} model: {total_params:,} total parameters, {trainable_params:,} trainable")
            logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to create model {architecture}: {e}")
            raise
    
    def train_model(self, train_loader: DataLoader, epochs: int = 10, 
                   learning_rate: float = 0.001, weight_decay: float = 1e-4) -> Dict[str, Any]:
        """Train the CNN model and return comprehensive metrics."""
        self.resource_monitor.start_monitoring()
        
        try:
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            # Training loop
            start_time = time.perf_counter()
            train_losses = []
            train_accuracies = []
            
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    if batch_idx % 100 == 0:
                        logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                
                scheduler.step()
                
                epoch_loss = running_loss / len(train_loader)
                epoch_accuracy = 100. * correct / total
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_accuracy)
                
                logger.info(f'Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
            
            training_time = time.perf_counter() - start_time
            resource_metrics = self.resource_monitor.stop_monitoring()
            
            return {
                "training_time_seconds": training_time,
                "final_loss": train_losses[-1],
                "final_accuracy": train_accuracies[-1],
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "resource_metrics": resource_metrics
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the CNN model with comprehensive metrics."""
        try:
            self.model.eval()
            test_loss = 0
            correct = 0
            total = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    
                    # Calculate loss
                    criterion = nn.CrossEntropyLoss()
                    test_loss += criterion(output, target).item()
                    
                    # Calculate accuracy
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    # Store predictions for additional metrics
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            # Calculate metrics
            accuracy = 100. * correct / total
            avg_loss = test_loss / len(test_loader)
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(all_targets, all_predictions, average='weighted')
            recall = recall_score(all_targets, all_predictions, average='weighted')
            f1 = f1_score(all_targets, all_predictions, average='weighted')
            
            return {
                "accuracy": accuracy,
                "loss": avg_loss,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_inference_benchmark(self, test_loader: DataLoader, batch_sizes: List[int]) -> Dict[str, Any]:
        """Run comprehensive inference benchmarks."""
        try:
            self.model.eval()
            latencies = []
            throughputs = []
            
            for batch_size in batch_sizes:
                batch_latencies = []
                
                # Warm-up runs
                for _ in range(10):
                    if batch_size == 1:
                        sample = next(iter(test_loader))[0][:1]
                        sample = sample.to(self.device)
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            _ = self.model(sample)
                        end_time = time.perf_counter()
                    else:
                        batch_data = next(iter(test_loader))[0][:batch_size]
                        batch_data = batch_data.to(self.device)
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            _ = self.model(batch_data)
                        end_time = time.perf_counter()
                
                # Benchmark runs
                for _ in range(100):
                    if batch_size == 1:
                        sample = next(iter(test_loader))[0][:1]
                        sample = sample.to(self.device)
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            _ = self.model(sample)
                        end_time = time.perf_counter()
                    else:
                        batch_data = next(iter(test_loader))[0][:batch_size]
                        batch_data = batch_data.to(self.device)
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            _ = self.model(batch_data)
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
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    'model': gpu_name,
                    'memory_gb': gpu_memory
                }
            
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
                     architecture: str, 
                     hyperparams: Dict[str, Any],
                     run_id: str,
                     mode: str = "training") -> BenchmarkResult:
        """Run comprehensive benchmark with full analysis."""
        try:
            logger.info(f"Starting CNN benchmark: {dataset}, {architecture}, {mode}")
            
            # Load dataset
            train_loader, test_loader = self.load_dataset(dataset, batch_size=hyperparams.get("batch_size", 32))
            
            # Create model
            self.create_model(architecture, num_classes=hyperparams.get("num_classes", 10), **hyperparams)
            
            # Get hardware configuration
            hardware_config = self.get_hardware_config()
            
            if mode == "training":
                # Training benchmark
                training_results = self.train_model(
                    train_loader, 
                    epochs=hyperparams.get("epochs", 10),
                    learning_rate=hyperparams.get("learning_rate", 0.001),
                    weight_decay=hyperparams.get("weight_decay", 1e-4)
                )
                evaluation_results = self.evaluate_model(test_loader)
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.DEEP_LEARNING,
                    model_name=f"{architecture}_cnn",
                    dataset=dataset,
                    run_id=run_id,
                    timestamp=datetime.utcnow(),
                    hardware_config=hardware_config,
                    performance_metrics=PerformanceMetrics(
                        training_time_seconds=training_results["training_time_seconds"],
                        inference_latency_ms=None,
                        throughput_samples_per_second=None,
                        convergence_epochs=hyperparams.get("epochs", 10)
                    ),
                    resource_metrics=training_results["resource_metrics"],
                    quality_metrics=QualityMetrics(
                        accuracy=evaluation_results.get("accuracy"),
                        f1_score=evaluation_results.get("f1_score"),
                        precision=evaluation_results.get("precision"),
                        recall=evaluation_results.get("recall"),
                        loss=evaluation_results.get("loss")
                    ),
                    metadata={
                        "architecture": architecture,
                        "hyperparameters": hyperparams,
                        "dataset_size": len(train_loader.dataset),
                        "final_loss": training_results.get("final_loss"),
                        "final_accuracy": training_results.get("final_accuracy")
                    }
                )
                
            elif mode == "inference":
                # Train model first
                training_results = self.train_model(
                    train_loader, 
                    epochs=hyperparams.get("epochs", 5),
                    learning_rate=hyperparams.get("learning_rate", 0.001)
                )
                
                # Inference benchmark
                inference_metrics = self.run_inference_benchmark(test_loader, [1, 10, 100])
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.DEEP_LEARNING,
                    model_name=f"{architecture}_cnn",
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
                        "architecture": architecture,
                        "hyperparameters": hyperparams,
                        "dataset_size": len(train_loader.dataset)
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
            if torch.cuda.is_available():
                return {
                    'peak_memory_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
                    'avg_memory_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                    'utilization_percent': None  # Would need nvidia-smi for this
                }
            
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
    parser = argparse.ArgumentParser(description="Enhanced Python CNN Benchmark")
    parser.add_argument("--mode", default="training", choices=["training", "inference"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--hyperparams", default="{}", type=str)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--enable-profiling", action="store_true")
    
    args = parser.parse_args()
    
    try:
        # Parse hyperparameters
        hyperparams = json.loads(args.hyperparams)
        
        # Create benchmark instance
        benchmark = EnhancedCNNBenchmark(enable_profiling=args.enable_profiling)
        
        # Run benchmark
        result = benchmark.run_benchmark(
            args.dataset,
            args.architecture,
            hyperparams,
            args.run_id,
            args.mode
        )
        
        # Save results
        output_file = f"{args.dataset}_{args.architecture}_{args.run_id}_{args.mode}_results.json"
        output_path = Path(args.output_dir) / output_file
        
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"CNN benchmark completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"CNN benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 