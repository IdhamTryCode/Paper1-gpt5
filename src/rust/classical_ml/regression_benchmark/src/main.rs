use std::collections::HashMap;
use std::time::Instant;
use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use linfa_elasticnet::ElasticNet;
use ndarray::{Array1, Array2, s};
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use sysinfo::{System, SystemExt, CpuExt, ProcessExt};
use uuid::Uuid;
use log::info;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Benchmark mode
    #[arg(long, default_value = "training")]
    mode: String,
    
    /// Dataset name
    #[arg(long)]
    dataset: String,
    
    /// Regression algorithm
    #[arg(long)]
    algorithm: String,
    
    /// Hyperparameters JSON
    #[arg(long, default_value = "{}")]
    hyperparams: String,
    
    /// Run ID
    #[arg(long)]
    run_id: Option<String>,
    
    /// Output directory
    #[arg(long, default_value = ".")]
    output_dir: String,
    
    /// Enable resource monitoring
    #[arg(long)]
    enable_monitoring: bool,
    
    /// Capture detailed metrics
    #[arg(long)]
    capture_metrics: bool,
    
    /// Framework to use
    #[arg(long, default_value = "linfa")]
    framework: String,
    
    /// Number of samples to use
    #[arg(long)]
    n_samples: Option<usize>,
    
    /// Enable profiling
    #[arg(long)]
    enable_profiling: bool,
    
    /// Optional CSV paths for parity with Python: X and y files without header
    #[arg(long)]
    data_csv: Option<String>,
    #[arg(long)]
    targets_csv: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HardwareConfig {
    cpu_model: String,
    cpu_cores: usize,
    cpu_threads: usize,
    memory_gb: f64,
    gpu_model: Option<String>,
    gpu_memory_gb: Option<f64>,
    storage_type: String,
    storage_capacity_gb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceMetrics {
    training_time_seconds: Option<f64>,
    inference_latency_ms: Option<f64>,
    throughput_samples_per_second: Option<f64>,
    convergence_epochs: Option<u32>,
    tokens_per_second: Option<f64>,
    latency_p50_ms: Option<f64>,
    latency_p95_ms: Option<f64>,
    latency_p99_ms: Option<f64>,
    latency_std_ms: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ResourceMetrics {
    peak_memory_mb: f64,
    average_memory_mb: f64,
    peak_gpu_memory_mb: Option<f64>,
    average_gpu_memory_mb: Option<f64>,
    cpu_utilization_percent: f64,
    gpu_utilization_percent: Option<f64>,
    energy_consumption_joules: Option<f64>,
    network_io_mb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QualityMetrics {
    accuracy: Option<f64>,
    loss: Option<f64>,
    f1_score: Option<f64>,
    precision: Option<f64>,
    recall: Option<f64>,
    rmse: Option<f64>,
    mae: Option<f64>,
    r2_score: Option<f64>,
    perplexity: Option<f64>,
    episode_reward: Option<f64>,
    convergence_steps: Option<u32>,
    mape: Option<f64>,
    explained_variance: Option<f64>,
    residual_std: Option<f64>,
    residual_skew: Option<f64>,
    residual_kurtosis: Option<f64>,
    mse: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
enum Language {
    Python,
    Rust,
}

#[derive(Debug, Serialize, Deserialize)]
enum TaskType {
    ClassicalMl,
    DeepLearning,
    ReinforcementLearning,
    Llm,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResult {
    framework: String,
    language: Language,
    task_type: TaskType,
    model_name: String,
    dataset: String,
    run_id: String,
    timestamp: DateTime<Utc>,
    hardware_config: HardwareConfig,
    performance_metrics: PerformanceMetrics,
    resource_metrics: ResourceMetrics,
    quality_metrics: QualityMetrics,
    metadata: HashMap<String, serde_json::Value>,
}

enum BaseModel { Linear(LinearRegression), RidgeElasticNet { alpha: f64 } }

type FittedElastic = linfa_elasticnet::ElasticNet<f64>;
enum FittedModel { LinfaLinear(linfa_linear::FittedLinearRegression<f64>), LinfaElastic(FittedElastic) }

struct EnhancedRegressionBenchmark {
    framework: String,
    resource_monitor: EnhancedResourceMonitor,
    rng: StdRng,
    enable_profiling: bool,
    profiling_data: HashMap<String, f64>,
    y_train_mean: Option<f64>,
}

impl EnhancedRegressionBenchmark {
    fn new(framework: String, enable_profiling: bool) -> Self {
        // Set deterministic seed for reproducibility
        let rng = StdRng::seed_from_u64(42);
        
        Self {
            framework,
            resource_monitor: EnhancedResourceMonitor::new(),
            rng,
            enable_profiling,
            profiling_data: HashMap::new(),
            y_train_mean: None,
        }
    }

    fn load_dataset(&self, dataset_name: &str, n_samples: Option<usize>) -> Result<(Array2<f64>, Array1<f64>)> {
        match dataset_name {
            "boston_housing" => self.load_boston_dataset(n_samples),
            "california_housing" => self.load_california_dataset(n_samples),
            "synthetic_linear" => self.generate_synthetic_dataset(1000, 20, 10, 0.1, n_samples),
            "synthetic_nonlinear" => self.generate_synthetic_dataset(1000, 20, 10, 0.5, n_samples),
            "synthetic_sparse" => self.generate_synthetic_dataset(1000, 50, 5, 0.1, n_samples),
            _ => Err(anyhow::anyhow!("Unknown dataset: {}", dataset_name)),
        }
    }

    fn load_dataset_from_csv(&self, x_path: &str, y_path: &str) -> Result<(Array2<f64>, Array1<f64>)> {
        let x_content = std::fs::read_to_string(x_path)?;
        let y_content = std::fs::read_to_string(y_path)?;
        let mut x_rows: Vec<f64> = Vec::new();
        let mut ncols: Option<usize> = None;
        let mut nrows: usize = 0;
        for line in x_content.lines() {
            if line.trim().is_empty() { continue; }
            let cols: Vec<&str> = line.trim().split(',').collect();
            if let Some(exp) = ncols { if exp != cols.len() { return Err(anyhow::anyhow!("Inconsistent column count in {}", x_path)); } } else { ncols = Some(cols.len()); }
            for v in cols { x_rows.push(v.parse::<f64>().map_err(|e| anyhow::anyhow!("Failed parsing '{}': {}", v, e))?); }
            nrows += 1;
        }
        let nc = ncols.ok_or_else(|| anyhow::anyhow!("Empty CSV: {}", x_path))?;
        let x = Array2::from_shape_vec((nrows, nc), x_rows)?;
        let mut y_vals: Vec<f64> = Vec::new();
        for line in y_content.lines() {
            if line.trim().is_empty() { continue; }
            y_vals.push(line.trim().parse::<f64>().map_err(|e| anyhow::anyhow!("Failed parsing target '{}': {}", line, e))?);
        }
        if y_vals.len() != nrows { return Err(anyhow::anyhow!("Targets length {} does not match rows {}", y_vals.len(), nrows)); }
        let y = Array1::from_vec(y_vals);
        Ok((x, y))
    }

    fn load_boston_dataset(&self, n_samples: Option<usize>) -> Result<(Array2<f64>, Array1<f64>)> {
        // Fallback: generate synthetic if CSV not embedded
        let data = match std::fs::read_to_string("src/rust/data/boston_housing.csv") {
            Ok(s) => s,
            Err(_) => return self.generate_synthetic_dataset(506, 13, 8, 0.2, n_samples),
        };
        let mut lines = data.lines();
        lines.next(); // Skip header
        
        let mut features = Vec::new();
        let mut targets = Vec::new();
        
        for line in lines {
            let values: Vec<f64> = line.split(',')
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            
            if values.len() >= 14 {
                targets.push(values[13]); // Target is the last column
                features.push(values[..13].to_vec()); // Features are first 13 columns
            }
        }
        
        let n_samples_actual = n_samples.unwrap_or(features.len());
        let n_samples_actual = n_samples_actual.min(features.len());
        
        // Randomly sample if needed
        let mut indices: Vec<usize> = (0..features.len()).collect();
        indices.partial_shuffle(&mut self.rng.clone(), n_samples_actual);
        
        let selected_features: Vec<Vec<f64>> = indices[..n_samples_actual]
            .iter()
            .map(|&i| features[i].clone())
            .collect();
        let selected_targets: Vec<f64> = indices[..n_samples_actual]
            .iter()
            .map(|&i| targets[i])
            .collect();
        
        let x = Array2::from_shape_vec(
            (n_samples_actual, 13),
            selected_features.into_iter().flatten().collect()
        )?;
        let y = Array1::from_vec(selected_targets);
        
        info!("Loaded Boston housing dataset: {} samples, {} features", x.shape()[0], x.shape()[1]);
        Ok((x, y))
    }

    fn load_california_dataset(&self, n_samples: Option<usize>) -> Result<(Array2<f64>, Array1<f64>)> {
        // Fallback to synthetic if CSV not present
        let data = match std::fs::read_to_string("src/rust/data/california_housing.csv") {
            Ok(s) => s,
            Err(_) => return self.generate_synthetic_dataset(20640, 8, 6, 0.3, n_samples),
        };
        let mut lines = data.lines();
        lines.next(); // Skip header
        
        let mut features = Vec::new();
        let mut targets = Vec::new();
        
        for line in lines {
            let values: Vec<f64> = line.split(',')
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            
            if values.len() >= 9 {
                targets.push(values[8]); // Target is the last column
                features.push(values[..8].to_vec()); // Features are first 8 columns
            }
        }
        
        let n_samples_actual = n_samples.unwrap_or(features.len());
        let n_samples_actual = n_samples_actual.min(features.len());
        
        // Randomly sample if needed
        let mut indices: Vec<usize> = (0..features.len()).collect();
        indices.partial_shuffle(&mut self.rng.clone(), n_samples_actual);
        
        let selected_features: Vec<Vec<f64>> = indices[..n_samples_actual]
            .iter()
            .map(|&i| features[i].clone())
            .collect();
        let selected_targets: Vec<f64> = indices[..n_samples_actual]
            .iter()
            .map(|&i| targets[i])
            .collect();
        
        let x = Array2::from_shape_vec(
            (n_samples_actual, 8),
            selected_features.into_iter().flatten().collect()
        )?;
        let y = Array1::from_vec(selected_targets);
        
        info!("Loaded California housing dataset: {} samples, {} features", x.shape()[0], x.shape()[1]);
        Ok((x, y))
    }

    fn generate_synthetic_dataset(&self, n_samples: usize, n_features: usize, n_informative: usize, noise: f64, limit: Option<usize>) -> Result<(Array2<f64>, Array1<f64>)> {
        let n_samples_actual = limit.unwrap_or(n_samples);
        
        // Generate synthetic features
        let mut x = Array2::zeros((n_samples_actual, n_features));
        let mut y = Array1::zeros(n_samples_actual);
        
        // Generate informative features
        for i in 0..n_informative {
            let feature = self.rng.clone().gen_range(-1.0..1.0);
            let weight = self.rng.clone().gen_range(-2.0..2.0);
            
            for j in 0..n_samples_actual {
                x[[j, i]] = feature + self.rng.clone().gen_range(-0.1..0.1);
                y[j] += weight * x[[j, i]];
            }
        }
        
        // Add noise to target
        for i in 0..n_samples_actual {
            y[i] += self.rng.clone().gen_range(-noise..noise);
        }
        
        info!("Generated synthetic dataset: {} samples, {} features", x.shape()[0], x.shape()[1]);
        Ok((x, y))
    }

    fn create_model(&mut self, algorithm: &str, hyperparams: &HashMap<String, f64>) -> Result<BaseModel> {
        match algorithm {
            "linear" => Ok(BaseModel::Linear(LinearRegression::new())),
            // If alpha absent, set to negative to trigger CV selection later
            "ridge" => Ok(BaseModel::RidgeElasticNet { alpha: *hyperparams.get("alpha").unwrap_or(&-1.0) }),
            _ => Err(anyhow::anyhow!("Unknown algorithm: {}", algorithm)),
        }
    }

    fn preprocess_data(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>)> {
        // Shuffle with deterministic seed (42) then 80/20 split to mirror sklearn train_test_split(random_state=42)
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut self.rng);
        // Reorder X and y according to shuffled indices
        let mut x_shuf = Array2::zeros((n_samples, n_features));
        let mut y_shuf = Array1::zeros(n_samples);
        for (new_i, &old_i) in indices.iter().enumerate() {
            x_shuf.row_mut(new_i).assign(&x.row(old_i));
            y_shuf[new_i] = y[old_i];
        }
        let n_train = (n_samples as f64 * 0.8) as usize;
        let mut x_train = x_shuf.slice(s![..n_train, ..]).to_owned();
        let mut x_test = x_shuf.slice(s![n_train.., ..]).to_owned();
        let y_train = y_shuf.slice(s![..n_train]).to_owned();
        let y_test = y_shuf.slice(s![n_train..]).to_owned();
        // Center y by train mean for ridge stability; store mean to add back
        let y_mean = y_train.mean().unwrap();
        self.y_train_mean = Some(y_mean);
        // Standardize by train statistics
        let means = x_train.mean_axis(ndarray::Axis(0)).unwrap();
        let stds = x_train.std_axis(ndarray::Axis(0), 0.0);
        for j in 0..x_train.ncols() {
            let m = means[j];
            let s = if stds[j] == 0.0 { 1.0 } else { stds[j] };
            for i in 0..x_train.nrows() { x_train[[i,j]] = (x_train[[i,j]] - m) / s; }
            for i in 0..x_test.nrows() { x_test[[i,j]] = (x_test[[i,j]] - m) / s; }
        }
        info!("Data preprocessed: train={:?}, test={:?}", x_train.shape(), x_test.shape());
        Ok((x_train, x_test, &y_train - y_mean, &y_test - y_mean))
    }

    fn train_model(&mut self, model: &BaseModel, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(f64, ResourceMetrics, FittedModel)> {
        self.resource_monitor.start_monitoring();
        let start_time = Instant::now();

        let fitted = match model {
            BaseModel::Linear(lm) => {
                let dataset = Dataset::new(x_train.clone(), y_train.clone());
                match lm.fit(&dataset) {
                    Ok(fitted_model) => FittedModel::LinfaLinear(fitted_model),
                    Err(e) => {
                        info!(
                            "Linear regression fit failed ({}); falling back to Ridge (ElasticNet L2) with small alpha",
                            e
                        );
                        let ridge_alpha = 1e-6; // tiny L2 to stabilize when X'X is singular
                        let fitted = ElasticNet::params()
                            .penalty(ridge_alpha)
                            .l1_ratio(0.0)
                            .fit(&dataset)?;
                        FittedModel::LinfaElastic(fitted)
                    }
                }
            }
            BaseModel::RidgeElasticNet { alpha } => {
                let mut best_alpha = *alpha;
                if best_alpha <= 0.0 {
                    // Simple 5-fold CV on train to select alpha
                    let grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0];
                    let k = 5usize;
                    let n = x_train.nrows();
                    let fold_size = (n + k - 1) / k;
                    let mut best_mse = f64::INFINITY;
                    for &a in &grid {
                        let mut mses = Vec::new();
                        for fold in 0..k {
                            let start = fold * fold_size;
                            let end = ((fold + 1) * fold_size).min(n);
                            if start >= end { continue; }
                            let x_val = x_train.slice(s![start..end, ..]).to_owned();
                            let y_val = y_train.slice(s![start..end]).to_owned();
                            let x_trn_top = x_train.slice(s![..start, ..]).to_owned();
                            let y_trn_top = y_train.slice(s![..start]).to_owned();
                            let x_trn_bot = x_train.slice(s![end.., ..]).to_owned();
                            let y_trn_bot = y_train.slice(s![end..]).to_owned();
                            let x_trn = ndarray::concatenate(ndarray::Axis(0), &[x_trn_top.view(), x_trn_bot.view()]).unwrap();
                            let y_trn = ndarray::concatenate(ndarray::Axis(0), &[y_trn_top.view(), y_trn_bot.view()]).unwrap();
                            let ds = Dataset::new(x_trn, y_trn);
                            let m = ElasticNet::params().penalty(a).l1_ratio(0.0).fit(&ds)?;
                            let y_hat = m.predict(&x_val);
                            let mse = self.calculate_mse(&y_val, &y_hat);
                            mses.push(mse);
                        }
                        if !mses.is_empty() {
                            let mean_mse = mses.iter().sum::<f64>() / mses.len() as f64;
                            if mean_mse < best_mse { best_mse = mean_mse; best_alpha = a; }
                        }
                    }
                    // fall back if still invalid
                    if best_alpha <= 0.0 { best_alpha = 0.1; }
                }
                let dataset = Dataset::new(x_train.clone(), y_train.clone());
                let fitted = ElasticNet::params().penalty(best_alpha).l1_ratio(0.0).fit(&dataset)?;
                FittedModel::LinfaElastic(fitted)
            }
        };

        let training_time = start_time.elapsed().as_secs_f64();
        let resource_metrics = self.resource_monitor.stop_monitoring();

        // Store profiling data
        if self.enable_profiling {
            self.profiling_data.insert("training_time_seconds".to_string(), training_time);
        }

        Ok((training_time, resource_metrics, fitted))
    }

    fn evaluate_with_model(&self, fitted: &FittedModel, x_test: &Array2<f64>, y_test: &Array1<f64>) -> Result<HashMap<String, f64>> {
        let mut y_pred = match fitted {
            FittedModel::LinfaLinear(m) => m.predict(x_test),
            FittedModel::LinfaElastic(m) => m.predict(x_test),
        };
        // Add back mean to compare on original scale
        if let Some(mu) = self.y_train_mean { y_pred = y_pred + mu; }
        let y_test_orig = if let Some(mu) = self.y_train_mean { y_test + mu } else { y_test.clone() };
        
        // Calculate comprehensive metrics
        let mse = self.calculate_mse(&y_test_orig, &y_pred);
        let rmse = mse.sqrt();
        let mae = self.calculate_mae(&y_test_orig, &y_pred);
        let r2 = self.calculate_r2_score(&y_test_orig, &y_pred);
        
        // Calculate additional metrics
        let mape = self.calculate_mape(&y_test_orig, &y_pred);
        let explained_variance = self.calculate_explained_variance(&y_test_orig, &y_pred);
        
        // Calculate residuals statistics
        let residuals = &y_test_orig - &y_pred;
        let residual_std = residuals.std(0.0);
        let residual_skew = self.calculate_skewness(&residuals);
        let residual_kurtosis = self.calculate_kurtosis(&residuals);
        
        let mut metrics = HashMap::new();
        metrics.insert("rmse".to_string(), rmse);
        metrics.insert("mae".to_string(), mae);
        metrics.insert("r2_score".to_string(), r2);
        metrics.insert("mape".to_string(), mape);
        metrics.insert("explained_variance".to_string(), explained_variance);
        metrics.insert("residual_std".to_string(), residual_std);
        metrics.insert("residual_skew".to_string(), residual_skew);
        metrics.insert("residual_kurtosis".to_string(), residual_kurtosis);
        metrics.insert("mse".to_string(), mse);
        
        Ok(metrics)
    }

    fn calculate_mse(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let diff = y_true - y_pred;
        diff.dot(&diff) / y_true.len() as f64
    }

    fn calculate_mae(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let diff = y_true - y_pred;
        diff.mapv(|x| x.abs()).sum() / y_true.len() as f64
    }

    fn calculate_r2_score(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let ss_res = (y_true - y_pred).dot(&(y_true - y_pred));
        let mean = y_true.mean().unwrap();
        let centered = y_true - &Array1::from_elem(y_true.len(), mean);
        let ss_tot = centered.dot(&centered);
        
        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }

    fn calculate_mape(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in 0..y_true.len() {
            if y_true[i] != 0.0 {
                sum += ((y_true[i] - y_pred[i]).abs() / y_true[i].abs()) * 100.0;
                count += 1;
            }
        }
        
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    fn calculate_explained_variance(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let var_pred = y_pred.var(0.0);
        let var_true = y_true.var(0.0);
        
        if var_true == 0.0 {
            0.0
        } else {
            var_pred / var_true
        }
    }

    fn calculate_skewness(&self, data: &Array1<f64>) -> f64 {
        let mean = data.mean().unwrap();
        let std = data.std(0.0);
        
        if std == 0.0 {
            return 0.0;
        }
        
        let n = data.len() as f64;
        let skewness = data.iter()
            .map(|&x| ((x - mean) / std).powi(3))
            .sum::<f64>() / n;
        
        skewness
    }

    fn calculate_kurtosis(&self, data: &Array1<f64>) -> f64 {
        let mean = data.mean().unwrap();
        let std = data.std(0.0);
        
        if std == 0.0 {
            return 0.0;
        }
        
        let n = data.len() as f64;
        let kurtosis = data.iter()
            .map(|&x| ((x - mean) / std).powi(4))
            .sum::<f64>() / n;
        
        kurtosis - 3.0
    }

    fn run_inference_benchmark(&self, fitted: &FittedModel, x_test: &Array2<f64>, batch_sizes: &[usize]) -> Result<HashMap<String, f64>> {
        
        let mut latencies = Vec::new();
        let mut throughputs = Vec::new();
        
        for &batch_size in batch_sizes {
            let mut batch_latencies = Vec::new();
            
            // Warm-up runs
                for _ in 0..10 {
                    let _ = match fitted {
                        FittedModel::LinfaLinear(m) => m.predict(&x_test.slice(s![..batch_size, ..]).to_owned()),
                        FittedModel::LinfaElastic(m) => m.predict(&x_test.slice(s![..batch_size, ..]).to_owned()),
                    };
                }
            
            // Benchmark runs
            for _ in 0..100 {
                let start_time = Instant::now();
                let _ = match fitted {
                    FittedModel::LinfaLinear(m) => m.predict(&x_test.slice(s![..batch_size, ..]).to_owned()),
                    FittedModel::LinfaElastic(m) => m.predict(&x_test.slice(s![..batch_size, ..]).to_owned()),
                };
                let latency = start_time.elapsed().as_secs_f64() * 1000.0; // Convert to ms
                batch_latencies.push(latency);
            }
            
            let avg_latency = batch_latencies.iter().sum::<f64>() / batch_latencies.len() as f64;
            latencies.push(avg_latency);
            throughputs.push(batch_size as f64 / (avg_latency / 1000.0)); // samples per second
        }
        
        // Calculate percentiles
        let all_latencies: Vec<f64> = latencies.iter().flat_map(|&x| std::iter::repeat(x).take(100)).collect();
        let p50 = self.percentile(&all_latencies, 50.0);
        let p95 = self.percentile(&all_latencies, 95.0);
        let p99 = self.percentile(&all_latencies, 99.0);
        
        let mut metrics = HashMap::new();
        metrics.insert("inference_latency_ms".to_string(), latencies.iter().sum::<f64>() / latencies.len() as f64);
        metrics.insert("latency_p50_ms".to_string(), p50);
        metrics.insert("latency_p95_ms".to_string(), p95);
        metrics.insert("latency_p99_ms".to_string(), p99);
        metrics.insert("throughput_samples_per_second".to_string(), throughputs.iter().sum::<f64>() / throughputs.len() as f64);
        metrics.insert("latency_std_ms".to_string(), self.calculate_std(&latencies));
        
        Ok(metrics)
    }

    fn percentile(&self, values: &[f64], percentile: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    fn calculate_std(&self, values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    fn get_hardware_config(&self) -> HardwareConfig {
        let mut sys = System::new_all();
        sys.refresh_all();
        
        let cpu_name = sys.cpus().first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        
        let cpu_count = sys.cpus().len();
        let memory_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        
        // Try to get GPU information
        let gpu_info = self.get_gpu_info();
        
        HardwareConfig {
            cpu_model: cpu_name,
            cpu_cores: cpu_count,
            cpu_threads: cpu_count,
            memory_gb,
            gpu_model: None,
            gpu_memory_gb: gpu_info.get("peak_memory_mb").and_then(|v| v.clone()).map(|mb| mb / 1024.0),
            storage_type: "SSD".to_string(), // Default assumption
            storage_capacity_gb: 1000.0, // Default assumption
        }
    }

    fn get_gpu_info(&self) -> HashMap<String, Option<f64>> {
        let mut gpu_metrics = HashMap::new();
        
        // Try to get NVIDIA GPU metrics
        if let Ok(output) = Command::new("nvidia-smi")
            .args(&["--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
            .output() {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Some(line) = output_str.lines().next() {
                    let parts: Vec<&str> = line.split(", ").collect();
                    if parts.len() >= 2 {
                        if let Ok(memory_mb) = parts[0].trim().parse::<f64>() {
                            gpu_metrics.insert("peak_memory_mb".to_string(), Some(memory_mb));
                            gpu_metrics.insert("avg_memory_mb".to_string(), Some(memory_mb));
                        }
                        if let Ok(utilization) = parts[1].trim().parse::<f64>() {
                            gpu_metrics.insert("utilization_percent".to_string(), Some(utilization));
                        }
                    }
                }
            }
        }
        
        gpu_metrics
    }

    fn run_benchmark(&mut self, 
                     dataset: &str, 
                     algorithm: &str, 
                     hyperparams: &HashMap<String, f64>,
                     run_id: &str,
                     mode: &str,
                     data_csv: Option<&str>,
                     targets_csv: Option<&str>) -> Result<BenchmarkResult> {
        info!("Starting benchmark: {}, {}, {}", dataset, algorithm, mode);
        
        // Load and preprocess data
        let (x, y) = if let (Some(xp), Some(yp)) = (data_csv, targets_csv) {
            self.load_dataset_from_csv(xp, yp)?
        } else {
            self.load_dataset(dataset, None)?
        };
        let (x_train, x_test, y_train, y_test) = self.preprocess_data(&x, &y)?;
        
        // Create model
        let base_model = self.create_model(algorithm, hyperparams)?;
        
        // Get hardware configuration
        let hardware_config = self.get_hardware_config();
        
        if mode == "training" {
            // Training benchmark
            let (training_time, resource_metrics, fitted_model) = self.train_model(&base_model, &x_train, &y_train)?;
            let quality_metrics = self.evaluate_with_model(&fitted_model, &x_test, &y_test)?;
            
            // Combine quality metrics with training results
            let mut combined_quality = quality_metrics.clone();
            combined_quality.insert("cv_r2_mean".to_string(), self.profiling_data.get("cv_r2_mean").unwrap_or(&0.0).clone());
            combined_quality.insert("cv_r2_std".to_string(), self.profiling_data.get("cv_r2_std").unwrap_or(&0.0).clone());
            combined_quality.insert("model_sparsity".to_string(), self.profiling_data.get("model_sparsity").unwrap_or(&0.0).clone());
            combined_quality.insert("n_nonzero_coefficients".to_string(), self.profiling_data.get("n_nonzero_coefficients").unwrap_or(&0.0).clone());
            
            return Ok(BenchmarkResult {
                framework: self.framework.clone(),
                language: Language::Rust,
                task_type: TaskType::ClassicalMl,
                model_name: format!("{}_regression", algorithm),
                dataset: dataset.to_string(),
                run_id: run_id.to_string(),
                timestamp: Utc::now(),
                hardware_config,
                performance_metrics: PerformanceMetrics {
                    training_time_seconds: Some(training_time),
                    inference_latency_ms: None,
                    throughput_samples_per_second: None,
                    convergence_epochs: None,
                    tokens_per_second: None,
                    latency_p50_ms: None,
                    latency_p95_ms: None,
                    latency_p99_ms: None,
                    latency_std_ms: None,
                },
                resource_metrics,
                quality_metrics: QualityMetrics {
                    accuracy: None,
                    loss: None,
                    f1_score: None,
                    precision: None,
                    recall: None,
                    rmse: quality_metrics.get("rmse").cloned(),
                    mae: quality_metrics.get("mae").cloned(),
                    r2_score: quality_metrics.get("r2_score").cloned(),
                    perplexity: None,
                    episode_reward: None,
                    convergence_steps: None,
                    mape: quality_metrics.get("mape").cloned(),
                    explained_variance: quality_metrics.get("explained_variance").cloned(),
                    residual_std: quality_metrics.get("residual_std").cloned(),
                    residual_skew: quality_metrics.get("residual_skew").cloned(),
                    residual_kurtosis: quality_metrics.get("residual_kurtosis").cloned(),
                    mse: quality_metrics.get("mse").cloned(),
                },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("algorithm".to_string(), serde_json::Value::String(algorithm.to_string()));
                    meta.insert("hyperparameters".to_string(), serde_json::to_value(hyperparams)?);
                    meta.insert("dataset_size".to_string(), serde_json::Value::Number(serde_json::Number::from(x.shape()[0])));
                    meta.insert("features".to_string(), serde_json::Value::Number(serde_json::Number::from(x.shape()[1])));
                    meta.insert("cv_r2_mean".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.profiling_data.get("cv_r2_mean").unwrap_or(&0.0).clone()).unwrap()));
                    meta.insert("cv_r2_std".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.profiling_data.get("cv_r2_std").unwrap_or(&0.0).clone()).unwrap()));
                    meta.insert("model_sparsity".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.profiling_data.get("model_sparsity").unwrap_or(&0.0).clone()).unwrap()));
                    meta.insert("n_nonzero_coefficients".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.profiling_data.get("n_nonzero_coefficients").unwrap_or(&0.0).clone()).unwrap()));
                    meta
                },
            });
            
        } else if mode == "inference" {
            // Train model first
            let (_tt, _rm, fitted_model) = self.train_model(&base_model, &x_train, &y_train)?;
            
            // Inference benchmark
            let inference_metrics = self.run_inference_benchmark(&fitted_model, &x_test, &[1, 10, 100])?;
            
            return Ok(BenchmarkResult {
                framework: self.framework.clone(),
                language: Language::Rust,
                task_type: TaskType::ClassicalMl,
                model_name: format!("{}_regression", algorithm),
                dataset: dataset.to_string(),
                run_id: run_id.to_string(),
                timestamp: Utc::now(),
                hardware_config,
                performance_metrics: PerformanceMetrics {
                    training_time_seconds: None,
                    inference_latency_ms: inference_metrics.get("inference_latency_ms").cloned(),
                    throughput_samples_per_second: inference_metrics.get("throughput_samples_per_second").cloned(),
                    convergence_epochs: None,
                    tokens_per_second: None,
                    latency_p50_ms: inference_metrics.get("latency_p50_ms").cloned(),
                    latency_p95_ms: inference_metrics.get("latency_p95_ms").cloned(),
                    latency_p99_ms: inference_metrics.get("latency_p99_ms").cloned(),
                    latency_std_ms: inference_metrics.get("latency_std_ms").cloned(),
                },
                resource_metrics: ResourceMetrics {
                    peak_memory_mb: 0.0,
                    average_memory_mb: 0.0,
                    cpu_utilization_percent: 0.0,
                    peak_gpu_memory_mb: None,
                    average_gpu_memory_mb: None,
                    gpu_utilization_percent: None,
                    energy_consumption_joules: None,
                    network_io_mb: None,
                },
                quality_metrics: QualityMetrics {
                    accuracy: None,
                    loss: None,
                    f1_score: None,
                    precision: None,
                    recall: None,
                    rmse: None,
                    mae: None,
                    r2_score: None,
                    perplexity: None,
                    episode_reward: None,
                    convergence_steps: None,
                    mape: None,
                    explained_variance: None,
                    residual_std: None,
                    residual_skew: None,
                    residual_kurtosis: None,
                    mse: None,
                },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("algorithm".to_string(), serde_json::Value::String(algorithm.to_string()));
                    meta.insert("hyperparameters".to_string(), serde_json::to_value(hyperparams)?);
                    meta.insert("dataset_size".to_string(), serde_json::Value::Number(serde_json::Number::from(x.shape()[0])));
                    meta.insert("features".to_string(), serde_json::Value::Number(serde_json::Number::from(x.shape()[1])));
                    meta
                },
            });
            
        } else {
            return Err(anyhow::anyhow!("Unknown mode: {}", mode));
        }
    }
}

struct EnhancedResourceMonitor {
    start_memory: Option<usize>,
    peak_memory: usize,
    memory_samples: Vec<usize>,
    cpu_samples: Vec<f32>,
    start_time: Option<Instant>,
    process_id: u32,
}

impl EnhancedResourceMonitor {
    fn new() -> Self {
        Self {
            start_memory: None,
            peak_memory: 0,
            memory_samples: Vec::new(),
            cpu_samples: Vec::new(),
            start_time: None,
            process_id: std::process::id(),
        }
    }

    fn start_monitoring(&mut self) {
        self.start_time = Some(Instant::now());
        self.start_memory = Some(self.get_memory_usage());
        self.peak_memory = self.start_memory.unwrap();
        self.memory_samples = vec![self.start_memory.unwrap()];
        self.cpu_samples = vec![self.get_cpu_usage()];
    }

    fn stop_monitoring(&mut self) -> ResourceMetrics {
        let end_memory = self.get_memory_usage();
        let end_cpu = self.get_cpu_usage();
        
        self.memory_samples.push(end_memory);
        self.cpu_samples.push(end_cpu);
        
        // Calculate comprehensive metrics
        let peak_memory = self.memory_samples.iter().max().unwrap_or(&0);
        let avg_memory = self.memory_samples.iter().sum::<usize>() / self.memory_samples.len();
        let avg_cpu = self.cpu_samples.iter().sum::<f32>() / self.cpu_samples.len() as f32;
        
        // Try to get GPU metrics
        let gpu_metrics = self.get_gpu_metrics();
        
        ResourceMetrics {
            peak_memory_mb: *peak_memory as f64 / (1024.0 * 1024.0),
            average_memory_mb: avg_memory as f64 / (1024.0 * 1024.0),
            cpu_utilization_percent: avg_cpu as f64,
            peak_gpu_memory_mb: gpu_metrics.get("peak_memory_mb").and_then(|v| *v),
            average_gpu_memory_mb: gpu_metrics.get("avg_memory_mb").and_then(|v| *v),
            gpu_utilization_percent: gpu_metrics.get("utilization_percent").and_then(|v| *v),
            energy_consumption_joules: None, // Would need additional hardware monitoring
            network_io_mb: None, // Would need additional network monitoring
        }
    }

    fn get_memory_usage(&self) -> usize {
        let mut sys = System::new_all();
        let pid = sysinfo::Pid::from(self.process_id as usize);
        sys.refresh_process(pid);
        if let Some(p) = sys.process(pid) {
            // memory() returns KiB
            (p.memory() as usize) * 1024
        } else {
            0
        }
    }

    fn get_cpu_usage(&self) -> f32 {
        let mut sys = System::new_all();
        let pid = sysinfo::Pid::from(self.process_id as usize);
        sys.refresh_process(pid);
        if let Some(p) = sys.process(pid) {
            p.cpu_usage()
        } else {
            0.0
        }
    }

    fn get_gpu_metrics(&self) -> HashMap<String, Option<f64>> {
        let mut gpu_metrics = HashMap::new();
        
        // Try to get NVIDIA GPU metrics
        if let Ok(output) = Command::new("nvidia-smi")
            .args(&["--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
            .output() {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Some(line) = output_str.lines().next() {
                    let parts: Vec<&str> = line.split(", ").collect();
                    if parts.len() >= 2 {
                        if let Ok(memory_mb) = parts[0].trim().parse::<f64>() {
                            gpu_metrics.insert("peak_memory_mb".to_string(), Some(memory_mb));
                            gpu_metrics.insert("avg_memory_mb".to_string(), Some(memory_mb));
                        }
                        if let Ok(utilization) = parts[1].trim().parse::<f64>() {
                            gpu_metrics.insert("utilization_percent".to_string(), Some(utilization));
                        }
                    }
                }
            }
        }
        
        gpu_metrics
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            training_time_seconds: None,
            inference_latency_ms: None,
            throughput_samples_per_second: None,
            convergence_epochs: None,
            tokens_per_second: None,
            latency_p50_ms: None,
            latency_p95_ms: None,
            latency_p99_ms: None,
            latency_std_ms: None,
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            peak_memory_mb: 0.0,
            average_memory_mb: 0.0,
            peak_gpu_memory_mb: None,
            average_gpu_memory_mb: None,
            cpu_utilization_percent: 0.0,
            gpu_utilization_percent: None,
            energy_consumption_joules: None,
            network_io_mb: None,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            accuracy: None,
            loss: None,
            f1_score: None,
            precision: None,
            recall: None,
            rmse: None,
            mae: None,
            r2_score: None,
            perplexity: None,
            episode_reward: None,
            convergence_steps: None,
            mape: None,
            explained_variance: None,
            residual_std: None,
            residual_skew: None,
            residual_kurtosis: None,
            mse: None,
        }
    }
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    let args = Args::parse();
    
    // Generate run ID if not provided
    let run_id = args.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());
    
    // Parse hyperparameters
    let hyperparams: HashMap<String, f64> = if args.hyperparams.is_empty() {
        HashMap::new()
    } else {
        serde_json::from_str(&args.hyperparams)
            .context("Failed to parse hyperparameters")?
    };
    
    // Create benchmark instance
    let mut benchmark = EnhancedRegressionBenchmark::new(
        args.framework,
        args.enable_profiling
    );
    
    // Run benchmark
    let result = benchmark.run_benchmark(
        &args.dataset,
        &args.algorithm,
        &hyperparams,
        &run_id,
        &args.mode,
        args.data_csv.as_deref(),
        args.targets_csv.as_deref()
    )?;
    
    // Save results
    let output_file = format!("{}_{}_{}_{}_results.json", 
                             args.dataset, args.algorithm, run_id, args.mode);
    let output_path = Path::new(&args.output_dir).join(&output_file);
    
    let json_result = serde_json::to_string_pretty(&result)?;
    fs::write(&output_path, json_result)?;
    
    info!("Benchmark completed. Results saved to: {:?}", output_path);
    
    Ok(())
} 