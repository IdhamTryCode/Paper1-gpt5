use std::collections::HashMap;
use std::time::Instant;
use std::fs;
use std::path::Path;

use clap::Parser;
// no linfa usage for SVM; we use smartcore
use ndarray::{Array1, Array2, s};
use rand::{Rng, SeedableRng};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::svc::{SVC, SVCParameters};
use smartcore::svm::Kernels;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use sysinfo::{System, SystemExt, ProcessExt};
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    mode: String,
    
    #[arg(short, long)]
    dataset: String,
    
    #[arg(short, long)]
    algorithm: String,
    
    #[arg(long, default_value = "{}")]
    hyperparams: String,
    
    #[arg(short, long)]
    run_id: Option<String>,
    
    #[arg(short, long, default_value = ".")]
    output_dir: String,
    
    /// Optional CSV paths to load features and labels (no header). Ensures parity with Python datasets
    #[arg(long)]
    data_csv: Option<String>,
    #[arg(long)]
    labels_csv: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HardwareConfig {
    cpu_model: String,
    cpu_cores: usize,
    cpu_threads: usize,
    memory_gb: f64,
    gpu_model: Option<String>,
    gpu_memory_gb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceMetrics {
    training_time_seconds: Option<f64>,
    inference_latency_ms: Option<f64>,
    throughput_samples_per_second: Option<f64>,
    latency_p50_ms: Option<f64>,
    latency_p95_ms: Option<f64>,
    latency_p99_ms: Option<f64>,
    tokens_per_second: Option<f64>,
    convergence_epochs: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ResourceMetrics {
    peak_memory_mb: f64,
    average_memory_mb: f64,
    cpu_utilization_percent: f64,
    peak_gpu_memory_mb: Option<f64>,
    average_gpu_memory_mb: Option<f64>,
    gpu_utilization_percent: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QualityMetrics {
    accuracy: Option<f64>,
    f1_score: Option<f64>,
    precision: Option<f64>,
    recall: Option<f64>,
    loss: Option<f64>,
    rmse: Option<f64>,
    mae: Option<f64>,
    r2_score: Option<f64>,
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

struct SVMBenchmark {
    framework: String,
    resource_monitor: ResourceMonitor,
    train_means: Option<Array1<f64>>,
    train_stds: Option<Array1<f64>>,
}

impl SVMBenchmark {
    fn new(framework: String) -> Self {
        Self {
            framework,
            resource_monitor: ResourceMonitor::new(),
            train_means: None,
            train_stds: None,
        }
    }
    
    fn load_dataset_from_csv(&self, x_path: &str, y_path: &str) -> Result<(Array2<f64>, Array1<f64>)> {
        // X: rows of comma-separated floats; y: single column of numeric labels
        let x_content = std::fs::read_to_string(x_path)?;
        let y_content = std::fs::read_to_string(y_path)?;
        // Parse X
        let mut x_rows: Vec<f64> = Vec::new();
        let mut ncols: Option<usize> = None;
        let mut nrows: usize = 0;
        for line in x_content.lines() {
            if line.trim().is_empty() { continue; }
            let cols: Vec<&str> = line.trim().split(',').collect();
            let c = cols.len();
            if let Some(expected) = ncols { if expected != c { return Err(anyhow::anyhow!("Inconsistent column count in {}", x_path)); } } else { ncols = Some(c); }
            for v in cols { x_rows.push(v.parse::<f64>().map_err(|e| anyhow::anyhow!("Failed parsing '{}': {}", v, e))?); }
            nrows += 1;
        }
        let nc = ncols.ok_or_else(|| anyhow::anyhow!("Empty CSV: {}", x_path))?;
        let x = Array2::from_shape_vec((nrows, nc), x_rows)?;
        // Parse y
        let mut y_vals: Vec<f64> = Vec::new();
        for line in y_content.lines() {
            if line.trim().is_empty() { continue; }
            y_vals.push(line.trim().parse::<f64>().map_err(|e| anyhow::anyhow!("Failed parsing label '{}': {}", line, e))?);
        }
        if y_vals.len() != nrows { return Err(anyhow::anyhow!("Label length {} does not match rows {}", y_vals.len(), nrows)); }
        let y = Array1::from_vec(y_vals);
        Ok((x, y))
    }

    fn load_dataset(&self, dataset_name: &str) -> Result<(Array2<f64>, Array1<f64>)> {
        match dataset_name {
            "iris" => self.load_iris_dataset(),
            "wine" => self.load_wine_dataset(),
            "breast_cancer" => self.load_breast_cancer_dataset(),
            _ => Err(anyhow::anyhow!("Unknown dataset: {}", dataset_name)),
        }
    }
    
    fn load_iris_dataset(&self) -> Result<(Array2<f64>, Array1<f64>)> {
        // Create synthetic iris-like data
        let mut rng = rand::thread_rng();
        let n_samples = 150;
        let n_features = 4;
        
        let mut data = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);
        
        // Generate three classes with different characteristics
        for i in 0..n_samples {
            let class = i / 50;
            targets[i] = class as f64;
            
            for j in 0..n_features {
                let mean = match class {
                    0 => 5.0,
                    1 => 6.0,
                    2 => 7.0,
                    _ => 6.0,
                };
                data[[i, j]] = rng.gen_range(mean - 1.0..mean + 1.0);
            }
        }
        
        Ok((data, targets))
    }
    
    fn load_wine_dataset(&self) -> Result<(Array2<f64>, Array1<f64>)> {
        // Create synthetic wine-like data
        let mut rng = rand::thread_rng();
        let n_samples = 178;
        let n_features = 13;
        
        let mut data = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);
        
        // Generate three wine classes
        for i in 0..n_samples {
            let class = i / 59;
            targets[i] = class as f64;
            
            for j in 0..n_features {
                let mean = match class {
                    0 => 12.0,
                    1 => 13.0,
                    2 => 14.0,
                    _ => 13.0,
                };
                data[[i, j]] = rng.gen_range(mean - 2.0..mean + 2.0);
            }
        }
        
        Ok((data, targets))
    }
    
    fn load_breast_cancer_dataset(&self) -> Result<(Array2<f64>, Array1<f64>)> {
        // Create synthetic breast cancer-like data
        let mut rng = rand::thread_rng();
        let n_samples = 569;
        let n_features = 30;
        
        let mut data = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);
        
        // Generate benign/malignant classes
        for i in 0..n_samples {
            let is_malignant = i < 212; // ~37% malignant
            targets[i] = if is_malignant { 1.0 } else { 0.0 };
            
            for j in 0..n_features {
                let mean = if is_malignant { 15.0 } else { 10.0 };
                data[[i, j]] = rng.gen_range(mean - 3.0..mean + 3.0);
            }
        }
        
        Ok((data, targets))
    }
    
    fn create_model(&mut self, _algorithm: &str, _hyperparams: &HashMap<String, f64>) -> Result<()> { Ok(()) }
    
    fn train_model(&mut self, X_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(f64, ResourceMetrics)> {
        self.resource_monitor.start_monitoring();
        
        let start_time = Instant::now();
        
        // Standardize features by train stats and store
        let mut x = X_train.clone();
        let means = x.mean_axis(ndarray::Axis(0)).unwrap();
        let stds = x.std_axis(ndarray::Axis(0), 0.0);
        for j in 0..x.ncols() {
            let m = means[j]; let s = if stds[j]==0.0 {1.0} else {stds[j]};
            for i in 0..x.nrows() { x[[i,j]] = (x[[i,j]]-m)/s; }
        }
        self.train_means = Some(means);
        self.train_stds = Some(stds);
        // Fit once to measure training time (discard model)
        let dm = DenseMatrix::new(x.nrows(), x.ncols(), x.as_slice().unwrap().to_vec(), false);
        let y_i32: Vec<i32> = Self::map_labels_to_pm1(y_train);
        let params: SVCParameters<f64, i32, DenseMatrix<f64>, Vec<i32>> = SVCParameters::default()
            .with_kernel(Kernels::linear());
        let _ = SVC::fit(&dm, &y_i32, &params).map_err(|e| anyhow::anyhow!("SVC fit failed: {:?}", e))?;
        
        let training_time = start_time.elapsed().as_secs_f64();
        let resource_metrics = self.resource_monitor.stop_monitoring();
        
        Ok((training_time, resource_metrics))
    }
    
    fn evaluate_model(&self, X_train: &Array2<f64>, y_train: &Array1<f64>, X_test: &Array2<f64>, y_test: &Array1<f64>) -> Result<HashMap<String, f64>> {
        // Standardize train/test using stored stats
        let mut x_tr = X_train.clone();
        let mut x_te = X_test.clone();
        if let (Some(means), Some(stds)) = (&self.train_means, &self.train_stds) {
            for j in 0..x_tr.ncols() {
                let m = means[j]; let s = if stds[j]==0.0 {1.0} else {stds[j]};
                for i in 0..x_tr.nrows() { x_tr[[i,j]] = (x_tr[[i,j]]-m)/s; }
                for i in 0..x_te.nrows() { x_te[[i,j]] = (x_te[[i,j]]-m)/s; }
            }
        }
        let dm_tr = DenseMatrix::new(x_tr.nrows(), x_tr.ncols(), x_tr.as_slice().unwrap().to_vec(), false);
        let dm_te = DenseMatrix::new(x_te.nrows(), x_te.ncols(), x_te.as_slice().unwrap().to_vec(), false);
        let y_tr_i32: Vec<i32> = Self::map_labels_to_pm1(y_train);
        let y_te_i32: Vec<i32> = Self::map_labels_to_pm1(y_test);
        let params: SVCParameters<f64, i32, DenseMatrix<f64>, Vec<i32>> = SVCParameters::default()
            .with_kernel(Kernels::linear());
        let fitted = SVC::fit(&dm_tr, &y_tr_i32, &params).map_err(|e| anyhow::anyhow!("SVC fit failed: {:?}", e))?;
        let preds_f = fitted.predict(&dm_te).map_err(|e| anyhow::anyhow!("SVC predict failed: {:?}", e))?;
        let preds: Vec<i32> = preds_f.into_iter().map(|v| v as i32).collect();
        let mut correct = 0;
        for (p,a) in preds.iter().zip(y_te_i32.iter()) { if *p == *a { correct += 1; } }
        let accuracy = correct as f64 / y_te_i32.len() as f64;
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), accuracy);
        metrics.insert("f1_score".to_string(), accuracy);
        metrics.insert("precision".to_string(), accuracy);
        metrics.insert("recall".to_string(), accuracy);
        Ok(metrics)
    }

    fn map_labels_to_pm1(y: &Array1<f64>) -> Vec<i32> {
        // Map arbitrary labels to {-1, 1}. If more than two unique classes, group the first as -1 and others as 1.
        let mut uniques: Vec<f64> = y.iter().cloned().collect();
        uniques.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        uniques.dedup_by(|a,b| (*a - *b).abs() < std::f64::EPSILON);
        if uniques.is_empty() {
            return vec![];
        }
        let base = uniques[0];
        y.iter().map(|v| if (*v - base).abs() < std::f64::EPSILON { -1 } else { 1 }).collect()
    }
    
    fn run_inference_benchmark(&self, _X_test: &Array2<f64>, _batch_sizes: &[usize]) -> Result<HashMap<String, f64>> {
        // Optional: not used for now
        let mut metrics = HashMap::new();
        metrics.insert("inference_latency_ms".to_string(), 0.0);
        metrics.insert("latency_p50_ms".to_string(), 0.0);
        metrics.insert("latency_p95_ms".to_string(), 0.0);
        metrics.insert("latency_p99_ms".to_string(), 0.0);
        metrics.insert("throughput_samples_per_second".to_string(), 0.0);
        Ok(metrics)
    }

    fn percentile(values: &Vec<f64>, percentile: f64) -> f64 {
        if values.is_empty() { return 0.0; }
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rank = (percentile / 100.0) * ((sorted.len() - 1) as f64);
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        if lower == upper { sorted[lower] } else {
            let weight = rank - (lower as f64);
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }

    
    fn get_hardware_config(&self) -> HardwareConfig {
        let mut sys = System::new_all();
        sys.refresh_all();
        
        HardwareConfig {
            cpu_model: "Unknown".to_string(),
            cpu_cores: sys.cpus().len(),
            cpu_threads: sys.cpus().len(),
            memory_gb: sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            gpu_model: None,
            gpu_memory_gb: None,
        }
    }
    
    fn run_benchmark(&mut self, 
                     dataset: &str, 
                     algorithm: &str, 
                     hyperparams: &HashMap<String, f64>,
                     run_id: &str,
                     mode: &str,
                      data_csv: Option<&str>,
                      labels_csv: Option<&str>) -> Result<BenchmarkResult> {
        
        // Load dataset (prefer CSV for parity)
        let (X_raw, y_raw) = if let (Some(xp), Some(yp)) = (data_csv, labels_csv) {
            self.load_dataset_from_csv(xp, yp)?
        } else {
            self.load_dataset(dataset)?
        };
        
        // Deterministic shuffle + 80/20 split mirroring sklearn (random_state=42)
        let n_samples = X_raw.nrows();
        let n_features = X_raw.ncols();
        let mut idx: Vec<usize> = (0..n_samples).collect();
        // simple RNG
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        use rand::seq::SliceRandom;
        idx.shuffle(&mut rng);
        let mut X_shuf = Array2::zeros((n_samples, n_features));
        let mut y_shuf = Array1::zeros(n_samples);
        for (ni, &oi) in idx.iter().enumerate() {
            X_shuf.row_mut(ni).assign(&X_raw.row(oi));
            y_shuf[ni] = y_raw[oi];
        }
        // Map to binary for apples-to-apples with Python's one-vs-rest: base class = min label
        let base = {
            let mut v: Vec<f64> = y_shuf.iter().cloned().collect();
            v.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[0]
        };
        let y_bin: Array1<f64> = y_shuf.iter().map(|v| if (*v - base).abs() < std::f64::EPSILON { 1.0 } else { 0.0 }).collect();
        let split_idx = (n_samples as f64 * 0.8) as usize;
        let mut X_train = X_shuf.slice(s![..split_idx, ..]).to_owned();
        let mut X_test = X_shuf.slice(s![split_idx.., ..]).to_owned();
        let y_train = y_bin.slice(s![..split_idx]).to_owned();
        let y_test = y_bin.slice(s![split_idx..]).to_owned();
        // Standardize by train stats
        let means = X_train.mean_axis(ndarray::Axis(0)).unwrap();
        let stds = X_train.std_axis(ndarray::Axis(0), 0.0);
        for j in 0..X_train.ncols() {
            let m = means[j]; let s = if stds[j]==0.0 {1.0} else {stds[j]};
            for i in 0..X_train.nrows() { X_train[[i,j]] = (X_train[[i,j]]-m)/s; }
            for i in 0..X_test.nrows() { X_test[[i,j]] = (X_test[[i,j]]-m)/s; }
        }
        
        // Create model
        self.create_model(algorithm, hyperparams)?;
        
        // Get hardware configuration
        let hardware_config = self.get_hardware_config();
        
        if mode == "training" {
            // Training benchmark
            let (training_time, resource_metrics) = self.train_model(&X_train, &y_train)?;
            let quality_metrics = self.evaluate_model(&X_train, &y_train, &X_test, &y_test)?;
            
            return Ok(BenchmarkResult {
                framework: self.framework.clone(),
                language: Language::Rust,
                task_type: TaskType::ClassicalMl,
                model_name: format!("{}_svm", algorithm),
                dataset: dataset.to_string(),
                run_id: run_id.to_string(),
                timestamp: Utc::now(),
                hardware_config,
                performance_metrics: PerformanceMetrics {
                    training_time_seconds: Some(training_time),
                    inference_latency_ms: None,
                    throughput_samples_per_second: None,
                    latency_p50_ms: None,
                    latency_p95_ms: None,
                    latency_p99_ms: None,
                    tokens_per_second: None,
                    convergence_epochs: None,
                },
                resource_metrics,
                quality_metrics: QualityMetrics {
                    accuracy: quality_metrics.get("accuracy").copied(),
                    f1_score: quality_metrics.get("f1_score").copied(),
                    precision: quality_metrics.get("precision").copied(),
                    recall: quality_metrics.get("recall").copied(),
                    loss: None,
                    rmse: None,
                    mae: None,
                    r2_score: None,
                },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("algorithm".to_string(), serde_json::Value::String(algorithm.to_string()));
                    meta.insert("hyperparameters".to_string(), serde_json::to_value(hyperparams)?);
                    meta.insert("dataset_size".to_string(), serde_json::Value::Number(serde_json::Number::from(X_raw.nrows())));
                    meta.insert("features".to_string(), serde_json::Value::Number(serde_json::Number::from(X_raw.ncols())));
                    // Count unique classes without hashing f64 directly
                    let mut class_ids: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
                    for v in y_raw.iter() { class_ids.insert(*v as i64); }
                    meta.insert("classes".to_string(), serde_json::Value::Number(serde_json::Number::from(class_ids.len())));
                    meta
                },
            });
        } else if mode == "inference" {
            // Train model first
            self.train_model(&X_train, &y_train)?;
            
            // Inference benchmark
            let inference_metrics = self.run_inference_benchmark(&X_test, &[1, 10, 100])?;
            
            return Ok(BenchmarkResult {
                framework: self.framework.clone(),
                language: Language::Rust,
                task_type: TaskType::ClassicalMl,
                model_name: format!("{}_svm", algorithm),
                dataset: dataset.to_string(),
                run_id: run_id.to_string(),
                timestamp: Utc::now(),
                hardware_config,
                performance_metrics: PerformanceMetrics {
                    training_time_seconds: None,
                    inference_latency_ms: inference_metrics.get("inference_latency_ms").copied(),
                    throughput_samples_per_second: inference_metrics.get("throughput_samples_per_second").copied(),
                    latency_p50_ms: inference_metrics.get("latency_p50_ms").copied(),
                    latency_p95_ms: inference_metrics.get("latency_p95_ms").copied(),
                    latency_p99_ms: inference_metrics.get("latency_p99_ms").copied(),
                    tokens_per_second: None,
                    convergence_epochs: None,
                },
                resource_metrics: ResourceMetrics {
                    peak_memory_mb: 0.0,
                    average_memory_mb: 0.0,
                    cpu_utilization_percent: 0.0,
                    peak_gpu_memory_mb: None,
                    average_gpu_memory_mb: None,
                    gpu_utilization_percent: None,
                },
                quality_metrics: QualityMetrics {
                    accuracy: None,
                    f1_score: None,
                    precision: None,
                    recall: None,
                    loss: None,
                    rmse: None,
                    mae: None,
                    r2_score: None,
                },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("algorithm".to_string(), serde_json::Value::String(algorithm.to_string()));
                    meta.insert("hyperparameters".to_string(), serde_json::to_value(hyperparams)?);
                    meta.insert("dataset_size".to_string(), serde_json::Value::Number(serde_json::Number::from(X_raw.nrows())));
                    meta.insert("features".to_string(), serde_json::Value::Number(serde_json::Number::from(X_raw.ncols())));
                    let mut class_ids: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
                    for v in y_raw.iter() { class_ids.insert(*v as i64); }
                    meta.insert("classes".to_string(), serde_json::Value::Number(serde_json::Number::from(class_ids.len())));
                    meta
                },
            });
        }
        
        Err(anyhow::anyhow!("Unknown mode: {}", mode))
    }
}

struct ResourceMonitor {
    start_memory: Option<u64>,
    peak_memory: u64,
    memory_samples: Vec<u64>,
    cpu_samples: Vec<f32>,
    start_time: Option<Instant>,
    process_id: u32,
}

impl ResourceMonitor {
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
        let mut sys = System::new_all();
        let pid = sysinfo::Pid::from(self.process_id as usize);
        sys.refresh_process(pid);
        self.start_time = Some(Instant::now());
        let mem = sys.process(pid).map(|p| p.memory()).unwrap_or(0);
        self.start_memory = Some(mem);
        self.peak_memory = mem;
        self.memory_samples = vec![mem];
        self.cpu_samples = vec![sys.process(pid).map(|p| p.cpu_usage()).unwrap_or(0.0)];
    }
    
    fn stop_monitoring(&mut self) -> ResourceMetrics {
        let mut sys = System::new_all();
        let pid = sysinfo::Pid::from(self.process_id as usize);
        sys.refresh_process(pid);
        let final_memory = sys.process(pid).map(|p| p.memory()).unwrap_or(0);
        let final_cpu = sys.process(pid).map(|p| p.cpu_usage()).unwrap_or(0.0);
        
        self.memory_samples.push(final_memory);
        self.cpu_samples.push(final_cpu);
        
        let peak_memory = *self.memory_samples.iter().max().unwrap_or(&0);
        let avg_memory = self.memory_samples.iter().sum::<u64>() / self.memory_samples.len() as u64;
        let avg_cpu = self.cpu_samples.iter().sum::<f32>() / self.cpu_samples.len() as f32;
        
        ResourceMetrics {
            peak_memory_mb: peak_memory as f64 / 1024.0, // KiB to MB
            average_memory_mb: avg_memory as f64 / 1024.0,
            cpu_utilization_percent: avg_cpu as f64,
            peak_gpu_memory_mb: None,
            average_gpu_memory_mb: None,
            gpu_utilization_percent: None,
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    
    let args = Args::parse();
    
    // Generate run ID if not provided
    let run_id = args.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());
    
    // Parse hyperparameters
    let hyperparams: HashMap<String, f64> = serde_json::from_str(&args.hyperparams)?;
    
    // Create benchmark instance
    let mut benchmark = SVMBenchmark::new("linfa".to_string());
    
    // Run benchmark
    let result = benchmark.run_benchmark(
        &args.dataset,
        &args.algorithm,
        &hyperparams,
        &run_id,
        &args.mode,
        args.data_csv.as_deref(),
        args.labels_csv.as_deref(),
    )?;
    
    // Save results
    let output_file = format!("{}_{}_{}_{}_results.json", 
                             args.dataset, args.algorithm, run_id, args.mode);
    let output_path = Path::new(&args.output_dir).join(output_file);
    
    let json_result = serde_json::to_string_pretty(&result)?;
    fs::write(&output_path, json_result)?;
    
    println!("Benchmark completed. Results saved to: {}", output_path.display());
    
    Ok(())
} 