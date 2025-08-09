#!/usr/bin/env python3
"""
Enhanced Python DQN Benchmark Implementation

This module implements comprehensive DQN benchmarks using stable-baselines3 for comparison
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
import gymnasium as gym
from stable_baselines3 import DQN, DDQN, DuelingDQN, PrioritizedReplayDQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import psutil
import os
import sys
import platform
import subprocess
from dataclasses import asdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from src.shared.schemas.metrics import (
    BenchmarkResult, PerformanceMetrics, ResourceMetrics, QualityMetrics,
    HardwareConfig, Language, TaskType
)

# Import RL models
from src.python.reinforcement_learning.rl_models import (
    DQNAgent, DDQNAgent, DuelingDQNAgent, PrioritizedDQNAgent, RainbowDQNAgent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedDQNBenchmark:
    """Enhanced DQN benchmark implementation with comprehensive monitoring."""
    
    def __init__(self, framework: str = "stable-baselines3", enable_profiling: bool = True):
        self.framework = framework
        self.model = None
        self.resource_monitor = EnhancedResourceMonitor()
        self.enable_profiling = enable_profiling
        self.profiling_data = {}
        
        # Set deterministic seeds for reproducibility
        np.random.seed(42)
        if hasattr(np.random, 'default_rng'):
            self.rng = np.random.default_rng(42)
        else:
            self.rng = np.random.RandomState(42)
    
    def create_environment(self, env_name: str) -> gym.Env:
        """Create a reinforcement learning environment."""
        try:
            # Create environment with monitoring
            env = gym.make(env_name, render_mode=None)
            env = Monitor(env)
            
            # Set seed for reproducibility
            env.reset(seed=42)
            
            logger.info(f"Created environment: {env_name}")
            return env
            
        except Exception as e:
            logger.error(f"Failed to create environment {env_name}: {e}")
            raise
    
    def create_model(self, algorithm: str, env: gym.Env, hyperparams: Dict[str, Any]):
        """Create a DQN model with comprehensive algorithm support."""
        try:
            # Common hyperparameters
            learning_rate = hyperparams.get("learning_rate", 1e-4)
            buffer_size = hyperparams.get("buffer_size", 100000)
            learning_starts = hyperparams.get("learning_starts", 1000)
            batch_size = hyperparams.get("batch_size", 32)
            tau = hyperparams.get("tau", 1.0)
            gamma = hyperparams.get("gamma", 0.99)
            train_freq = hyperparams.get("train_freq", 4)
            gradient_steps = hyperparams.get("gradient_steps", 1)
            target_update_interval = hyperparams.get("target_update_interval", 1000)
            exploration_fraction = hyperparams.get("exploration_fraction", 0.1)
            exploration_initial_eps = hyperparams.get("exploration_initial_eps", 1.0)
            exploration_final_eps = hyperparams.get("exploration_final_eps", 0.05)
            
            if algorithm == "dqn":
                self.model = DQN(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    target_update_interval=target_update_interval,
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=exploration_initial_eps,
                    exploration_final_eps=exploration_final_eps,
                    verbose=0,
                    seed=42
                )
                
            elif algorithm == "ddqn":
                self.model = DDQN(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    target_update_interval=target_update_interval,
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=exploration_initial_eps,
                    exploration_final_eps=exploration_final_eps,
                    verbose=0,
                    seed=42
                )
                
            elif algorithm == "dueling_dqn":
                self.model = DuelingDQN(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    target_update_interval=target_update_interval,
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=exploration_initial_eps,
                    exploration_final_eps=exploration_final_eps,
                    verbose=0,
                    seed=42
                )
                
            elif algorithm == "prioritized_dqn":
                self.model = PrioritizedReplayDQN(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    target_update_interval=target_update_interval,
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=exploration_initial_eps,
                    exploration_final_eps=exploration_final_eps,
                    verbose=0,
                    seed=42
                )
                
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            logger.info(f"Created {algorithm} model with hyperparameters: {hyperparams}")
            
        except Exception as e:
            logger.error(f"Failed to create model {algorithm}: {e}")
            raise
    
    def train_model(self, env: gym.Env, total_timesteps: int = 10000) -> Dict[str, Any]:
        """Train the DQN model and return comprehensive metrics."""
        self.resource_monitor.start_monitoring()
        
        try:
            start_time = time.perf_counter()
            
            # Train the model
            self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
            
            training_time = time.perf_counter() - start_time
            resource_metrics = self.resource_monitor.stop_monitoring()
            
            # Get training statistics
            training_stats = self.model.get_env().get_attr('episode_rewards')[0] if hasattr(self.model.get_env(), 'get_attr') else []
            
            return {
                "training_time_seconds": training_time,
                "total_timesteps": total_timesteps,
                "episode_rewards": training_stats,
                "resource_metrics": resource_metrics
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, env: gym.Env, n_eval_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the DQN model with comprehensive metrics."""
        try:
            # Evaluate the model
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                env, 
                n_eval_episodes=n_eval_episodes,
                deterministic=True
            )
            
            # Run additional evaluation for detailed metrics
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            
            for episode in range(n_eval_episodes):
                obs, _ = env.reset()
                done = False
                truncated = False
                episode_reward = 0
                episode_length = 0
                
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Check if episode was successful (reward > 0 for most environments)
                if episode_reward > 0:
                    success_count += 1
            
            success_rate = success_count / n_eval_episodes
            
            return {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "success_rate": success_rate,
                "mean_episode_length": np.mean(episode_lengths),
                "std_episode_length": np.std(episode_lengths),
                "min_reward": np.min(episode_rewards),
                "max_reward": np.max(episode_rewards)
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_inference_benchmark(self, env: gym.Env, n_episodes: int = 100) -> Dict[str, Any]:
        """Run comprehensive inference benchmarks."""
        try:
            latencies = []
            throughputs = []
            
            # Warm-up runs
            for _ in range(10):
                obs, _ = env.reset()
                start_time = time.perf_counter()
                action, _ = self.model.predict(obs, deterministic=True)
                end_time = time.perf_counter()
            
            # Benchmark runs
            for _ in range(n_episodes):
                obs, _ = env.reset()
                episode_latencies = []
                
                done = False
                truncated = False
                
                while not (done or truncated):
                    start_time = time.perf_counter()
                    action, _ = self.model.predict(obs, deterministic=True)
                    end_time = time.perf_counter()
                    
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    episode_latencies.append(latency)
                    
                    obs, reward, done, truncated, info = env.step(action)
                
                avg_episode_latency = np.mean(episode_latencies)
                latencies.append(avg_episode_latency)
                throughputs.append(len(episode_latencies) / (avg_episode_latency / 1000))  # actions per second
            
            # Calculate percentiles
            all_latencies = np.concatenate([latencies] * 10)  # Approximate
            p50 = np.percentile(all_latencies, 50)
            p95 = np.percentile(all_latencies, 95)
            p99 = np.percentile(all_latencies, 99)
            
            return {
                "inference_latency_ms": np.mean(latencies),
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
                "throughput_actions_per_second": np.mean(throughputs),
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
                     environment: str, 
                     algorithm: str, 
                     hyperparams: Dict[str, Any],
                     run_id: str,
                     mode: str = "training") -> BenchmarkResult:
        """Run comprehensive benchmark with full analysis."""
        try:
            logger.info(f"Starting DQN benchmark: {environment}, {algorithm}, {mode}")
            
            # Create environment
            env = self.create_environment(environment)
            
            # Create model
            self.create_model(algorithm, env, hyperparams)
            
            # Get hardware configuration
            hardware_config = self.get_hardware_config()
            
            if mode == "training":
                # Training benchmark
                training_results = self.train_model(
                    env, 
                    total_timesteps=hyperparams.get("total_timesteps", 10000)
                )
                evaluation_results = self.evaluate_model(env, n_eval_episodes=10)
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.Python,
                    task_type=TaskType.ReinforcementLearning,
                    model_name=f"{algorithm}_dqn",
                    dataset=environment,
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
                        f1_score=None,
                        precision=None,
                        recall=None,
                        loss=None,
                        mean_reward=evaluation_results.get("mean_reward"),
                        success_rate=evaluation_results.get("success_rate")
                    ),
                    metadata={
                        "algorithm": algorithm,
                        "hyperparameters": hyperparams,
                        "environment": environment,
                        "total_timesteps": training_results.get("total_timesteps"),
                        "episode_rewards": training_results.get("episode_rewards")
                    }
                )
                
            elif mode == "inference":
                # Train model first
                training_results = self.train_model(
                    env, 
                    total_timesteps=hyperparams.get("total_timesteps", 5000)
                )
                
                # Inference benchmark
                inference_metrics = self.run_inference_benchmark(env, n_episodes=50)
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.Python,
                    task_type=TaskType.ReinforcementLearning,
                    model_name=f"{algorithm}_dqn",
                    dataset=environment,
                    run_id=run_id,
                    timestamp=datetime.utcnow(),
                    hardware_config=hardware_config,
                    performance_metrics=PerformanceMetrics(
                        training_time_seconds=None,
                        inference_latency_ms=inference_metrics.get("inference_latency_ms"),
                        throughput_samples_per_second=inference_metrics.get("throughput_actions_per_second"),
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
                        "environment": environment
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
    parser = argparse.ArgumentParser(description="Enhanced Python DQN Benchmark")
    parser.add_argument("--mode", default="training", choices=["training", "inference"])
    parser.add_argument("--environment", required=True)
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
        benchmark = EnhancedDQNBenchmark(enable_profiling=args.enable_profiling)
        
        # Run benchmark
        result = benchmark.run_benchmark(
            args.environment,
            args.algorithm,
            hyperparams,
            args.run_id,
            args.mode
        )
        
        # Save results
        output_file = f"{args.environment}_{args.algorithm}_{args.run_id}_{args.mode}_results.json"
        output_path = Path(args.output_dir) / output_file
        
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"DQN benchmark completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"DQN benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 