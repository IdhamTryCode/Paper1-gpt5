"""
Metrics schema definitions for the Rust vs Python ML Benchmark System.

This module defines the core data structures used throughout the benchmark system
for storing and processing benchmark results, performance metrics, and resource usage data.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class TaskType(Enum):
    """Enumeration of benchmark task types."""
    CLASSICAL_ML = "classical_ml"
    DEEP_LEARNING = "deep_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    LLM = "llm"


class Language(Enum):
    """Enumeration of programming languages."""
    PYTHON = "python"
    RUST = "rust"

# Backwards-compatibility aliases for legacy names used in some modules/tests
# These are simple references to keep import style stable without breaking enums
# Legacy camelCase-like variants
Language.Python = Language.PYTHON
Language.Rust = Language.RUST
TaskType.ClassicalMl = TaskType.CLASSICAL_ML
TaskType.DeepLearning = TaskType.DEEP_LEARNING
TaskType.ReinforcementLearning = TaskType.REINFORCEMENT_LEARNING
TaskType.Llm = TaskType.LLM


@dataclass
class HardwareConfig:
    """Hardware configuration for benchmark runs."""
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    memory_gb: float
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    storage_type: str = "NVMe SSD"
    storage_capacity_gb: float = 1000.0


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during benchmark runs."""
    training_time_seconds: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    throughput_samples_per_second: Optional[float] = None
    convergence_epochs: Optional[int] = None
    tokens_per_second: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None


@dataclass
class ResourceMetrics:
    """Resource usage metrics collected during benchmark runs."""
    peak_memory_mb: float
    average_memory_mb: float
    cpu_utilization_percent: float
    peak_gpu_memory_mb: Optional[float] = None
    average_gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    energy_consumption_joules: Optional[float] = None
    network_io_mb: Optional[float] = None


@dataclass
class QualityMetrics:
    """Quality metrics for model performance."""
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    perplexity: Optional[float] = None
    episode_reward: Optional[float] = None
    convergence_steps: Optional[int] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark result with all metrics and metadata."""
    framework: str
    language: Language
    task_type: TaskType
    model_name: str
    dataset: str
    run_id: str
    timestamp: datetime
    hardware_config: HardwareConfig
    performance_metrics: PerformanceMetrics
    resource_metrics: ResourceMetrics
    quality_metrics: QualityMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the benchmark result to a dictionary."""
        return {
            "framework": self.framework,
            "language": self.language.value,
            "task_type": self.task_type.value,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "hardware_config": {
                "cpu_model": self.hardware_config.cpu_model,
                "cpu_cores": self.hardware_config.cpu_cores,
                "cpu_threads": self.hardware_config.cpu_threads,
                "memory_gb": self.hardware_config.memory_gb,
                "gpu_model": self.hardware_config.gpu_model,
                "gpu_memory_gb": self.hardware_config.gpu_memory_gb,
                "storage_type": self.hardware_config.storage_type,
                "storage_capacity_gb": self.hardware_config.storage_capacity_gb,
            },
            "performance_metrics": {
                "training_time_seconds": self.performance_metrics.training_time_seconds,
                "inference_latency_ms": self.performance_metrics.inference_latency_ms,
                "throughput_samples_per_second": self.performance_metrics.throughput_samples_per_second,
                "convergence_epochs": self.performance_metrics.convergence_epochs,
                "tokens_per_second": self.performance_metrics.tokens_per_second,
                "latency_p50_ms": self.performance_metrics.latency_p50_ms,
                "latency_p95_ms": self.performance_metrics.latency_p95_ms,
                "latency_p99_ms": self.performance_metrics.latency_p99_ms,
            },
            "resource_metrics": {
                "peak_memory_mb": self.resource_metrics.peak_memory_mb,
                "average_memory_mb": self.resource_metrics.average_memory_mb,
                "peak_gpu_memory_mb": self.resource_metrics.peak_gpu_memory_mb,
                "average_gpu_memory_mb": self.resource_metrics.average_gpu_memory_mb,
                "cpu_utilization_percent": self.resource_metrics.cpu_utilization_percent,
                "gpu_utilization_percent": self.resource_metrics.gpu_utilization_percent,
                "energy_consumption_joules": self.resource_metrics.energy_consumption_joules,
                "network_io_mb": self.resource_metrics.network_io_mb,
            },
            "quality_metrics": {
                "accuracy": self.quality_metrics.accuracy,
                "loss": self.quality_metrics.loss,
                "f1_score": self.quality_metrics.f1_score,
                "precision": self.quality_metrics.precision,
                "recall": self.quality_metrics.recall,
                "rmse": self.quality_metrics.rmse,
                "mae": self.quality_metrics.mae,
                "r2_score": self.quality_metrics.r2_score,
                "perplexity": self.quality_metrics.perplexity,
                "episode_reward": self.quality_metrics.episode_reward,
                "convergence_steps": self.quality_metrics.convergence_steps,
            },
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create a BenchmarkResult from a dictionary."""
        return cls(
            framework=data["framework"],
            language=Language(data["language"]),
            task_type=TaskType(data["task_type"]),
            model_name=data["model_name"],
            dataset=data["dataset"],
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            hardware_config=HardwareConfig(**data["hardware_config"]),
            performance_metrics=PerformanceMetrics(**data["performance_metrics"]),
            resource_metrics=ResourceMetrics(**data["resource_metrics"]),
            quality_metrics=QualityMetrics(**data["quality_metrics"]),
            metadata=data.get("metadata", {}),
        )
    
    def to_json(self) -> str:
        """Convert the benchmark result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BenchmarkResult':
        """Create a BenchmarkResult from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class BenchmarkSummary:
    """Summary statistics for a set of benchmark results."""
    framework: str
    language: Language
    task_type: TaskType
    model_name: str
    dataset: str
    num_runs: int
    mean_performance: PerformanceMetrics
    std_performance: PerformanceMetrics
    mean_resource: ResourceMetrics
    std_resource: ResourceMetrics
    mean_quality: QualityMetrics
    std_quality: QualityMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the benchmark summary to a dictionary."""
        return {
            "framework": self.framework,
            "language": self.language.value,
            "task_type": self.task_type.value,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "num_runs": self.num_runs,
            "mean_performance": {
                "training_time_seconds": self.mean_performance.training_time_seconds,
                "inference_latency_ms": self.mean_performance.inference_latency_ms,
                "throughput_samples_per_second": self.mean_performance.throughput_samples_per_second,
                "convergence_epochs": self.mean_performance.convergence_epochs,
                "tokens_per_second": self.mean_performance.tokens_per_second,
                "latency_p50_ms": self.mean_performance.latency_p50_ms,
                "latency_p95_ms": self.mean_performance.latency_p95_ms,
                "latency_p99_ms": self.mean_performance.latency_p99_ms,
            },
            "std_performance": {
                "training_time_seconds": self.std_performance.training_time_seconds,
                "inference_latency_ms": self.std_performance.inference_latency_ms,
                "throughput_samples_per_second": self.std_performance.throughput_samples_per_second,
                "convergence_epochs": self.std_performance.convergence_epochs,
                "tokens_per_second": self.std_performance.tokens_per_second,
                "latency_p50_ms": self.std_performance.latency_p50_ms,
                "latency_p95_ms": self.std_performance.latency_p95_ms,
                "latency_p99_ms": self.std_performance.latency_p99_ms,
            },
            "mean_resource": {
                "peak_memory_mb": self.mean_resource.peak_memory_mb,
                "average_memory_mb": self.mean_resource.average_memory_mb,
                "peak_gpu_memory_mb": self.mean_resource.peak_gpu_memory_mb,
                "average_gpu_memory_mb": self.mean_resource.average_gpu_memory_mb,
                "cpu_utilization_percent": self.mean_resource.cpu_utilization_percent,
                "gpu_utilization_percent": self.mean_resource.gpu_utilization_percent,
                "energy_consumption_joules": self.mean_resource.energy_consumption_joules,
                "network_io_mb": self.mean_resource.network_io_mb,
            },
            "std_resource": {
                "peak_memory_mb": self.std_resource.peak_memory_mb,
                "average_memory_mb": self.std_resource.average_memory_mb,
                "peak_gpu_memory_mb": self.std_resource.peak_gpu_memory_mb,
                "average_gpu_memory_mb": self.std_resource.average_gpu_memory_mb,
                "cpu_utilization_percent": self.std_resource.cpu_utilization_percent,
                "gpu_utilization_percent": self.std_resource.gpu_utilization_percent,
                "energy_consumption_joules": self.std_resource.energy_consumption_joules,
                "network_io_mb": self.std_resource.network_io_mb,
            },
            "mean_quality": {
                "accuracy": self.mean_quality.accuracy,
                "loss": self.mean_quality.loss,
                "f1_score": self.mean_quality.f1_score,
                "precision": self.mean_quality.precision,
                "recall": self.mean_quality.recall,
                "rmse": self.mean_quality.rmse,
                "mae": self.mean_quality.mae,
                "r2_score": self.mean_quality.r2_score,
                "perplexity": self.mean_quality.perplexity,
                "episode_reward": self.mean_quality.episode_reward,
                "convergence_steps": self.mean_quality.convergence_steps,
            },
            "std_quality": {
                "accuracy": self.std_quality.accuracy,
                "loss": self.std_quality.loss,
                "f1_score": self.std_quality.f1_score,
                "precision": self.std_quality.precision,
                "recall": self.std_quality.recall,
                "rmse": self.std_quality.rmse,
                "mae": self.std_quality.mae,
                "r2_score": self.std_quality.r2_score,
                "perplexity": self.std_quality.perplexity,
                "episode_reward": self.std_quality.episode_reward,
                "convergence_steps": self.std_quality.convergence_steps,
            },
        } 