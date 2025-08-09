#!/usr/bin/env python3
"""
Enhanced Python Transformer LLM Benchmark Implementation

This module implements comprehensive Transformer benchmarks using Hugging Face Transformers
for comparison with Rust implementations. Features include advanced model architectures,
comprehensive monitoring, and production-ready deployment capabilities.
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
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
    AutoModelForTokenClassification, pipeline, GenerationConfig
)
from transformers.utils import logging as transformers_logging
import psutil
import os
import sys
import platform
import subprocess
from dataclasses import asdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

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


class EnhancedTransformerBenchmark:
    """Enhanced Transformer benchmark implementation using Hugging Face Transformers."""
    
    def __init__(self, framework: str = "transformers", enable_profiling: bool = True):
        self.framework = framework
        self.model = None
        self.tokenizer = None
        self.resource_monitor = EnhancedResourceMonitor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_profiling = enable_profiling
        self.profiling_data = {}
        
        # Set deterministic seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        
        logger.info(f"Using device: {self.device}")
        
    def load_model(self, model_name: str, task: str = "text-generation", 
                  use_half_precision: bool = False, use_quantization: bool = False):
        """Load a pre-trained transformer model with comprehensive error handling."""
        try:
            logger.info(f"Loading model: {model_name} for task: {task}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on task
            if task == "text-generation":
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            elif task == "sequence-classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            elif task == "question-answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            elif task == "token-classification":
                self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            else:
                self.model = AutoModel.from_pretrained(model_name)
            
            # Apply optimizations
            if use_half_precision and self.device.type == 'cuda':
                self.model = self.model.half()
                logger.info("Applied half precision optimization")
            
            if use_quantization:
                # Apply dynamic quantization for CPU
                if self.device.type == 'cpu':
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("Applied dynamic quantization")
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model loaded: {total_params:,} total parameters, {trainable_params:,} trainable")
            
            # Store model info for profiling
            if self.enable_profiling:
                self.profiling_data["total_parameters"] = total_params
                self.profiling_data["trainable_parameters"] = trainable_params
                self.profiling_data["model_name"] = model_name
                self.profiling_data["task"] = task
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 50, num_return_sequences: int = 1,
                     temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
                     do_sample: bool = True, repetition_penalty: float = 1.0) -> List[str]:
        """Generate text using the loaded model with comprehensive parameters."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            generated_texts = []
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def classify_text(self, texts: List[str], return_probs: bool = True) -> List[Dict[str, Any]]:
        """Classify text using the loaded model with comprehensive metrics."""
        try:
            results = []
            
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=-1)
                    
                    # Get predictions
                    predicted_class = torch.argmax(logits, dim=-1).item()
                    confidence = torch.max(probabilities, dim=-1).values.item()
                    
                    result = {
                        "text": text,
                        "predicted_class": predicted_class,
                        "confidence": confidence
                    }
                    
                    if return_probs:
                        result["probabilities"] = probabilities.cpu().numpy().tolist()
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            raise
    
    def question_answering(self, questions: List[str], contexts: List[str]) -> List[Dict[str, Any]]:
        """Perform question answering using the loaded model."""
        try:
            results = []
            
            for question, context in zip(questions, contexts):
                inputs = self.tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Get answer span
                    answer_start = torch.argmax(outputs.start_logits)
                    answer_end = torch.argmax(outputs.end_logits) + 1
                    
                    # Decode answer
                    answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
                    answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                    
                    # Calculate confidence
                    start_confidence = torch.softmax(outputs.start_logits, dim=-1).max().item()
                    end_confidence = torch.softmax(outputs.end_logits, dim=-1).max().item()
                    confidence = (start_confidence + end_confidence) / 2
                    
                    result = {
                        "question": question,
                        "context": context,
                        "answer": answer,
                        "confidence": confidence,
                        "start_position": answer_start.item(),
                        "end_position": answer_end.item()
                    }
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise
    
    def run_training_benchmark(self, dataset: List[str], epochs: int = 1, 
                             learning_rate: float = 1e-5, batch_size: int = 4) -> Dict[str, Any]:
        """Run training benchmark with comprehensive monitoring."""
        try:
            self.resource_monitor.start_monitoring()
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            self.model.train()
            total_loss = 0.0
            learning_curve = []
            
            start_time = time.perf_counter()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                # Process data in batches
                for i in range(0, len(dataset), batch_size):
                    batch_texts = dataset[i:i+batch_size]
                    
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_epoch_loss = epoch_loss / (len(dataset) // batch_size)
                learning_curve.append({
                    'epoch': epoch + 1,
                    'loss': avg_epoch_loss,
                    'learning_rate': learning_rate
                })
                
                logger.info(f'Epoch {epoch+1}/{epochs}: Loss: {avg_epoch_loss:.4f}')
            
            training_time = time.perf_counter() - start_time
            resource_metrics = self.resource_monitor.stop_monitoring()
            
            return {
                "training_time_seconds": training_time,
                "final_loss": avg_epoch_loss,
                "resource_metrics": resource_metrics,
                "learning_curve": learning_curve,
                "epochs_completed": epochs
            }
            
        except Exception as e:
            logger.error(f"Training benchmark failed: {e}")
            raise
    
    def run_inference_benchmark(self, prompts: List[str], batch_sizes: List[int],
                               sequence_lengths: List[int]) -> Dict[str, Any]:
        """Run comprehensive inference benchmarks."""
        try:
            self.model.eval()
            latencies = []
            throughputs = []
            memory_usage = []
            token_counts = []
            
            for batch_size in batch_sizes:
                for seq_length in sequence_lengths:
                    batch_latencies = []
                    batch_memory = []
                    batch_tokens = []
                    
                    # Create batch of prompts
                    batch_prompts = prompts[:batch_size]
                    
                    # Warm-up runs
                    for _ in range(5):
                        inputs = self.tokenizer(
                            batch_prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=seq_length
                        ).to(self.device)
                        
                        with torch.no_grad():
                            _ = self.model.generate(
                                **inputs,
                                max_length=seq_length + 20,
                                do_sample=False,
                                num_return_sequences=1
                            )
                    
                    # Benchmark runs
                    for _ in range(50):
                        inputs = self.tokenizer(
                            batch_prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=seq_length
                        ).to(self.device)
                        
                        # Measure memory before
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                            memory_before = torch.cuda.memory_allocated()
                        
                        start_time = time.perf_counter()
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_length=seq_length + 20,
                                do_sample=False,
                                num_return_sequences=1
                            )
                        
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                            memory_after = torch.cuda.memory_allocated()
                            batch_memory.append((memory_after - memory_before) / (1024 * 1024))
                        
                        end_time = time.perf_counter()
                        latency = (end_time - start_time) * 1000  # Convert to ms
                        
                        batch_latencies.append(latency)
                        batch_tokens.append(outputs.shape[1])  # Number of tokens generated
                    
                    avg_latency = np.mean(batch_latencies)
                    avg_tokens = np.mean(batch_tokens)
                    
                    latencies.append(avg_latency)
                    throughputs.append(batch_size / (avg_latency / 1000))  # samples per second
                    memory_usage.append(np.mean(batch_memory) if batch_memory else 0)
                    token_counts.append(avg_tokens)
            
            # Calculate percentiles
            all_latencies = np.concatenate([latencies] * 50)  # Approximate
            p50 = np.percentile(all_latencies, 50)
            p95 = np.percentile(all_latencies, 95)
            p99 = np.percentile(all_latencies, 99)
            
            return {
                "inference_latency_ms": np.mean(latencies),
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
                "throughput_samples_per_second": np.mean(throughputs),
                "latency_std_ms": np.std(latencies),
                "memory_usage_mb": np.mean(memory_usage),
                "avg_tokens_per_sample": np.mean(token_counts),
                "tokens_per_second": np.mean(throughputs) * np.mean(token_counts)
            }
            
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            raise
    
    def evaluate_model(self, test_prompts: List[str], test_responses: List[str]) -> Dict[str, float]:
        """Evaluate the model with comprehensive metrics."""
        try:
            # Generate responses
            generated_responses = []
            for prompt in test_prompts:
                responses = self.generate_text(prompt, max_length=50, num_return_sequences=1)
                generated_responses.extend(responses)
            
            # Calculate metrics
            from sklearn.metrics import bleu_score
            from nltk.translate.bleu_score import sentence_bleu
            import nltk
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            # Calculate BLEU scores
            bleu_scores = []
            for ref, hyp in zip(test_responses, generated_responses):
                ref_tokens = nltk.word_tokenize(ref.lower())
                hyp_tokens = nltk.word_tokenize(hyp.lower())
                bleu_scores.append(sentence_bleu([ref_tokens], hyp_tokens))
            
            avg_bleu = np.mean(bleu_scores)
            
            # Calculate perplexity (if applicable)
            perplexity = None
            try:
                total_loss = 0.0
                total_tokens = 0
                
                for prompt in test_prompts:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        loss = outputs.loss
                        total_loss += loss.item()
                        total_tokens += inputs['input_ids'].shape[1]
                
                perplexity = torch.exp(torch.tensor(total_loss / len(test_prompts)))
            except:
                pass
            
            return {
                "bleu_score": avg_bleu,
                "perplexity": perplexity.item() if perplexity is not None else None,
                "avg_response_length": np.mean([len(r.split()) for r in generated_responses])
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
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
                     model_name: str, 
                     task: str,
                     hyperparams: Dict[str, Any],
                     run_id: str,
                     mode: str = "inference") -> BenchmarkResult:
        """Run comprehensive benchmark with full analysis."""
        try:
            logger.info(f"Starting Transformer benchmark: {model_name}, {task}, {mode}")
            
            # Load model
            use_half_precision = hyperparams.get("use_half_precision", False)
            use_quantization = hyperparams.get("use_quantization", False)
            self.load_model(model_name, task, use_half_precision, use_quantization)
            
            # Get hardware configuration
            hardware_config = self.get_hardware_config()
            
            if mode == "training":
                # Training benchmark
                dataset = hyperparams.get("dataset", ["Sample text for training"] * 100)
                epochs = hyperparams.get("epochs", 1)
                learning_rate = hyperparams.get("learning_rate", 1e-5)
                batch_size = hyperparams.get("batch_size", 4)
                
                training_results = self.run_training_benchmark(dataset, epochs, learning_rate, batch_size)
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.LLM,
                    model_name=f"{model_name}_{task}",
                    dataset="custom",
                    run_id=run_id,
                    timestamp=datetime.utcnow(),
                    hardware_config=hardware_config,
                    performance_metrics=PerformanceMetrics(
                        training_time_seconds=training_results["training_time_seconds"],
                        inference_latency_ms=None,
                        throughput_samples_per_second=None,
                        convergence_epochs=training_results.get("epochs_completed")
                    ),
                    resource_metrics=training_results["resource_metrics"],
                    quality_metrics=QualityMetrics(
                        accuracy=None,
                        loss=training_results.get("final_loss"),
                        f1_score=None,
                        precision=None,
                        recall=None
                    ),
                    metadata={
                        "model_name": model_name,
                        "task": task,
                        "hyperparameters": hyperparams,
                        "total_parameters": self.profiling_data.get("total_parameters"),
                        "trainable_parameters": self.profiling_data.get("trainable_parameters")
                    }
                )
                
            elif mode == "inference":
                # Inference benchmark
                prompts = hyperparams.get("prompts", ["Hello, how are you?"] * 10)
                batch_sizes = hyperparams.get("batch_sizes", [1, 4, 8])
                sequence_lengths = hyperparams.get("sequence_lengths", [128, 256])
                
                inference_metrics = self.run_inference_benchmark(prompts, batch_sizes, sequence_lengths)
                
                return BenchmarkResult(
                    framework=self.framework,
                    language=Language.PYTHON,
                    task_type=TaskType.LLM,
                    model_name=f"{model_name}_{task}",
                    dataset="custom",
                    run_id=run_id,
                    timestamp=datetime.utcnow(),
                    hardware_config=hardware_config,
                    performance_metrics=PerformanceMetrics(
                        training_time_seconds=None,
                        inference_latency_ms=inference_metrics.get("inference_latency_ms"),
                        throughput_samples_per_second=inference_metrics.get("throughput_samples_per_second"),
                        convergence_epochs=None,
                        tokens_per_second=inference_metrics.get("tokens_per_second")
                    ),
                    resource_metrics=ResourceMetrics(
                        peak_memory_mb=0.0,
                        average_memory_mb=0.0,
                        cpu_utilization_percent=0.0,
                        peak_gpu_memory_mb=inference_metrics.get("memory_usage_mb"),
                        average_gpu_memory_mb=inference_metrics.get("memory_usage_mb"),
                        gpu_utilization_percent=None
                    ),
                    quality_metrics=QualityMetrics(
                        accuracy=None,
                        loss=None,
                        f1_score=None,
                        precision=None,
                        recall=None,
                        perplexity=None
                    ),
                    metadata={
                        "model_name": model_name,
                        "task": task,
                        "hyperparameters": hyperparams,
                        "total_parameters": self.profiling_data.get("total_parameters"),
                        "avg_tokens_per_sample": inference_metrics.get("avg_tokens_per_sample")
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
    parser = argparse.ArgumentParser(description="Enhanced Python Transformer LLM Benchmark")
    parser.add_argument("--mode", default="inference", choices=["training", "inference"])
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--hyperparams", default="{}", type=str)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--enable-profiling", action="store_true")
    
    args = parser.parse_args()
    
    try:
        # Parse hyperparameters
        hyperparams = json.loads(args.hyperparams)
        
        # Create benchmark instance
        benchmark = EnhancedTransformerBenchmark(enable_profiling=args.enable_profiling)
        
        # Run benchmark
        result = benchmark.run_benchmark(
            args.model_name,
            args.task,
            hyperparams,
            args.run_id,
            args.mode
        )
        
        # Save results
        output_file = f"{args.model_name}_{args.task}_{args.run_id}_{args.mode}_results.json"
        output_path = Path(args.output_dir) / output_file
        
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"Transformer benchmark completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Transformer benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 