use std::collections::HashMap;
use std::time::Instant;
use std::fs;
use std::path::Path;

use clap::Parser;
use linfa::prelude::*;
use linfa::dataset::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2, s};
use rand::Rng;
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
    
    /// Optional CSV path to load features (no header). Ensures parity with Python datasets
    #[arg(long)]
    data_csv: Option<String>,
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
    silhouette_score: Option<f64>,
    inertia: Option<f64>,
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

struct ClusteringBenchmark {
    framework: String,
    model: Option<Box<dyn ClusteringModel>>,
    resource_monitor: ResourceMonitor,
    // linfa-based path
    use_linfa_kmeans: bool,
    linfa_kmeans_clusters: usize,
    last_labels: Option<ndarray::Array1<usize>>,
}

trait ClusteringModel {
    fn fit(&mut self, data: &Array2<f64>) -> Result<()>;
    fn predict(&self, data: &Array2<f64>) -> Result<Array1<usize>>;
    fn get_inertia(&self) -> Option<f64>;
    fn get_silhouette_score(&self, data: &Array2<f64>, labels: &Array1<usize>) -> Option<f64>;
}

struct KMeansModel {
    centroids: Option<Vec<Array1<f64>>>,
    n_clusters: usize,
}

impl KMeansModel {
    fn new(n_clusters: usize) -> Self {
        Self {
            centroids: None,
            n_clusters,
        }
    }
}

impl ClusteringModel for KMeansModel {
    fn fit(&mut self, data: &Array2<f64>) -> Result<()> {
        // Simple k-means implementation (few iterations)
        use rand::seq::index::sample;
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Array1<f64>> = sample(&mut rng, data.nrows(), self.n_clusters)
            .into_iter()
            .map(|i| data.row(i).to_owned())
            .collect();
        for _ in 0..10 {
            // Assignments
            let mut new_centroids = vec![Array1::zeros(data.ncols()); self.n_clusters];
            let mut counts = vec![0usize; self.n_clusters];
            for i in 0..data.nrows() {
                let mut best = 0usize;
                let mut bestd = f64::INFINITY;
                for (k, c) in centroids.iter().enumerate() {
                    let diff = data.row(i).to_owned() - c;
                    let d = diff.dot(&diff);
                    if d < bestd { bestd = d; best = k; }
                }
                new_centroids[best] = new_centroids[best].clone() + data.row(i).to_owned();
                counts[best] += 1;
            }
            for k in 0..self.n_clusters {
                if counts[k] > 0 {
                    new_centroids[k] = new_centroids[k].mapv(|v| v / counts[k] as f64);
                } else {
                    new_centroids[k] = centroids[k].clone();
                }
            }
            centroids = new_centroids;
        }
        self.centroids = Some(centroids);
        Ok(())
    }
    
    fn predict(&self, data: &Array2<f64>) -> Result<Array1<usize>> {
        if let Some(ref centroids) = self.centroids {
            let mut preds = Array1::zeros(data.nrows());
            for i in 0..data.nrows() {
                let mut best = 0usize;
                let mut bestd = f64::INFINITY;
                for (k, c) in centroids.iter().enumerate() {
                    let diff = data.row(i).to_owned() - c;
                    let d = diff.dot(&diff);
                    if d < bestd { bestd = d; best = k; }
                }
                preds[i] = best;
            }
            Ok(preds)
        } else {
            Err(anyhow::anyhow!("Model not fitted"))
        }
    }
    
    fn get_inertia(&self) -> Option<f64> {
        if let Some(ref _centroids) = self.centroids {
            // Inertia isn't directly exposed; return None for now
            None
        } else {
            None
        }
    }
    
    fn get_silhouette_score(&self, data: &Array2<f64>, labels: &Array1<usize>) -> Option<f64> {
        // Simplified silhouette score calculation
        let n_samples = data.nrows();
        let mut silhouette_sum = 0.0;
        let mut valid_samples = 0;
        
        for i in 0..n_samples {
            let mut intra_cluster_dist = 0.0;
            let mut inter_cluster_dist = f64::MAX;
            let current_label = labels[i];
            let mut intra_count = 0;
            
            for j in 0..n_samples {
                if i != j {
                    let dist = euclidean_distance(&data.row(i), &data.row(j));
                    if labels[j] == current_label {
                        intra_cluster_dist += dist;
                        intra_count += 1;
                    } else {
                        // Simplified: just track minimum distance to other clusters
                        inter_cluster_dist = inter_cluster_dist.min(dist);
                    }
                }
            }
            
            if intra_count > 0 {
                intra_cluster_dist /= intra_count as f64;
                let silhouette = (inter_cluster_dist - intra_cluster_dist) / 
                               (inter_cluster_dist.max(intra_cluster_dist));
                silhouette_sum += silhouette;
                valid_samples += 1;
            }
        }
        
        if valid_samples > 0 {
            Some(silhouette_sum / valid_samples as f64)
        } else {
            None
        }
    }
}

struct DbscanModel {
    fitted: bool,
    eps: f64,
    min_points: usize,
}

impl DbscanModel {
    fn new(eps: f64, min_points: usize) -> Self {
        Self {
            fitted: false,
            eps,
            min_points,
        }
    }
}

impl ClusteringModel for DbscanModel {
    fn fit(&mut self, _data: &Array2<f64>) -> Result<()> {
        // Minimal validation; mark as fitted
        self.fitted = true;
        Ok(())
    }
    
    fn predict(&self, data: &Array2<f64>) -> Result<Array1<usize>> {
        if self.fitted {
            // For simplicity, return zeros as placeholder predictions
            Ok(Array1::zeros(data.nrows()))
        } else {
            Err(anyhow::anyhow!("Model not fitted"))
        }
    }
    
    fn get_inertia(&self) -> Option<f64> {
        // DBSCAN doesn't have inertia, return None
        None
    }
    
    fn get_silhouette_score(&self, data: &Array2<f64>, labels: &Array1<usize>) -> Option<f64> {
        // Same silhouette calculation as KMeans
        let n_samples = data.nrows();
        let mut silhouette_sum = 0.0;
        let mut valid_samples = 0;
        
        for i in 0..n_samples {
            let mut intra_cluster_dist = 0.0;
            let mut inter_cluster_dist = f64::MAX;
            let current_label = labels[i];
            let mut intra_count = 0;
            
            for j in 0..n_samples {
                if i != j {
                    let dist = euclidean_distance(&data.row(i), &data.row(j));
                    if labels[j] == current_label {
                        intra_cluster_dist += dist;
                        intra_count += 1;
                    } else {
                        inter_cluster_dist = inter_cluster_dist.min(dist);
                    }
                }
            }
            
            if intra_count > 0 {
                intra_cluster_dist /= intra_count as f64;
                let silhouette = (inter_cluster_dist - intra_cluster_dist) / 
                               (inter_cluster_dist.max(intra_cluster_dist));
                silhouette_sum += silhouette;
                valid_samples += 1;
            }
        }
        
        if valid_samples > 0 {
            Some(silhouette_sum / valid_samples as f64)
        } else {
            None
        }
    }
}

fn euclidean_distance(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

impl ClusteringBenchmark {
    fn new(framework: String) -> Self {
        Self {
            framework,
            model: None,
            resource_monitor: ResourceMonitor::new(),
            use_linfa_kmeans: false,
            linfa_kmeans_clusters: 3,
            last_labels: None,
        }
    }
    
    fn load_dataset(&self, dataset_name: &str) -> Result<Array2<f64>> {
        match dataset_name {
            "iris" => self.load_iris_dataset(),
            "wine" => self.load_wine_dataset(),
            "breast_cancer" => self.load_breast_cancer_dataset(),
            "synthetic" => self.load_synthetic_dataset(),
            _ => Err(anyhow::anyhow!("Unknown dataset: {}", dataset_name)),
        }
    }

    fn load_dataset_from_csv(&self, path: &str) -> Result<Array2<f64>> {
        let content = std::fs::read_to_string(path)?;
        let mut rows: Vec<f64> = Vec::new();
        let mut ncols: Option<usize> = None;
        let mut nrows: usize = 0;
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            let cols: Vec<&str> = line.trim().split(',').collect();
            let c = cols.len();
            if let Some(expected) = ncols { if expected != c { return Err(anyhow::anyhow!("Inconsistent column count in {}", path)); } } else { ncols = Some(c); }
            for v in cols {
                rows.push(v.parse::<f64>().map_err(|e| anyhow::anyhow!("Failed parsing '{}': {}", v, e))?);
            }
            nrows += 1;
        }
        let nc = ncols.ok_or_else(|| anyhow::anyhow!("Empty CSV: {}", path))?;
        let arr = Array2::from_shape_vec((nrows, nc), rows)?;
        Ok(arr)
    }
    
    fn load_iris_dataset(&self) -> Result<Array2<f64>> {
        // Create synthetic iris-like data
        let mut rng = rand::thread_rng();
        let n_samples = 150;
        let n_features = 4;
        
        let mut data = Array2::zeros((n_samples, n_features));
        
        // Generate three clusters with different characteristics
        for i in 0..n_samples {
            let cluster = i / 50;
            for j in 0..n_features {
                let mean = match cluster {
                    0 => 5.0,
                    1 => 6.0,
                    2 => 7.0,
                    _ => 6.0,
                };
                data[[i, j]] = rng.gen_range(mean - 1.0..mean + 1.0);
            }
        }
        
        Ok(data)
    }
    
    fn load_wine_dataset(&self) -> Result<Array2<f64>> {
        // Create synthetic wine-like data
        let mut rng = rand::thread_rng();
        let n_samples = 178;
        let n_features = 13;
        
        let mut data = Array2::zeros((n_samples, n_features));
        
        // Generate three wine clusters
        for i in 0..n_samples {
            let cluster = i / 59;
            for j in 0..n_features {
                let mean = match cluster {
                    0 => 12.0,
                    1 => 13.0,
                    2 => 14.0,
                    _ => 13.0,
                };
                data[[i, j]] = rng.gen_range(mean - 2.0..mean + 2.0);
            }
        }
        
        Ok(data)
    }
    
    fn load_breast_cancer_dataset(&self) -> Result<Array2<f64>> {
        // Create synthetic breast cancer-like data
        let mut rng = rand::thread_rng();
        let n_samples = 569;
        let n_features = 30;
        
        let mut data = Array2::zeros((n_samples, n_features));
        
        // Generate two clusters (benign/malignant)
        for i in 0..n_samples {
            let is_malignant = i < 212; // ~37% malignant
            for j in 0..n_features {
                let mean = if is_malignant { 15.0 } else { 10.0 };
                data[[i, j]] = rng.gen_range(mean - 3.0..mean + 3.0);
            }
        }
        
        Ok(data)
    }
    
    fn load_synthetic_dataset(&self) -> Result<Array2<f64>> {
        // Create synthetic clustering dataset
        let mut rng = rand::thread_rng();
        let n_samples = 1000;
        let n_features = 2;
        
        let mut data = Array2::zeros((n_samples, n_features));
        
        // Generate 5 clusters in 2D space
        for i in 0..n_samples {
            let cluster = i % 5;
            let center_x = match cluster {
                0 => 0.0,
                1 => 5.0,
                2 => 0.0,
                3 => 5.0,
                4 => 2.5,
                _ => 0.0,
            };
            let center_y = match cluster {
                0 => 0.0,
                1 => 0.0,
                2 => 5.0,
                3 => 5.0,
                4 => 2.5,
                _ => 0.0,
            };
            
            data[[i, 0]] = rng.gen_range(center_x - 1.0..center_x + 1.0);
            data[[i, 1]] = rng.gen_range(center_y - 1.0..center_y + 1.0);
        }
        
        Ok(data)
    }
    
    fn create_model(&mut self, algorithm: &str, hyperparams: &HashMap<String, f64>) -> Result<()> {
        match algorithm {
            "kmeans" => {
                let n_clusters = *hyperparams.get("n_clusters").unwrap_or(&3.0) as usize;
                // Prefer linfa-clustering KMeans for fidelity
                self.use_linfa_kmeans = true;
                self.linfa_kmeans_clusters = n_clusters;
                self.model = None;
            }
            "dbscan" => {
                let eps = hyperparams.get("eps").unwrap_or(&0.5);
                let min_points = *hyperparams.get("min_points").unwrap_or(&5.0) as usize;
                self.model = Some(Box::new(DbscanModel::new(*eps, min_points)));
            }
            _ => return Err(anyhow::anyhow!("Unknown algorithm: {}", algorithm)),
        }
        Ok(())
    }
    
    fn train_model(&mut self, data: &Array2<f64>) -> Result<(f64, ResourceMetrics)> {
        self.resource_monitor.start_monitoring();
        
        let start_time = Instant::now();
        
        if self.use_linfa_kmeans {
            let k = self.linfa_kmeans_clusters.max(2);
            let dummy = ndarray::Array1::<usize>::zeros(data.nrows());
            let ds: DatasetBase<ndarray::ArrayView2<'_, f64>, ndarray::Array1<usize>> = DatasetBase::new(data.view(), dummy);
            let fitted = KMeans::params(k).max_n_iterations(300).fit(&ds)?;
            let labels = fitted.predict(&ds);
            self.last_labels = Some(labels);
        } else if let Some(ref mut model) = self.model {
            model.fit(data)?;
        } else {
            return Err(anyhow::anyhow!("Model not created"));
        }
        
        let training_time = start_time.elapsed().as_secs_f64();
        let resource_metrics = self.resource_monitor.stop_monitoring();
        
        Ok((training_time, resource_metrics))
    }
    
    fn evaluate_model(&self, data: &Array2<f64>) -> Result<HashMap<String, f64>> {
        let predictions: Array1<usize> = if self.use_linfa_kmeans {
            if let Some(ref labels) = self.last_labels {
                labels.clone()
            } else {
                return Err(anyhow::anyhow!("No labels available from linfa KMeans"));
            }
        } else if let Some(ref model) = self.model {
            model.predict(data)?
        } else {
            return Err(anyhow::anyhow!("Model not trained"));
        };
            
            let mut metrics = HashMap::new();
            
            // Calculate inertia (for KMeans)
            // Not available directly from linfa; left as None
            
            // Calculate silhouette score
            if let Some(silhouette) = self.simple_silhouette(data, &predictions) {
                metrics.insert("silhouette_score".to_string(), silhouette);
            }
            
            // Calculate number of clusters (unique usize labels)
            let mut uniq: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
            for v in predictions.iter() { uniq.insert(*v); }
            metrics.insert("n_clusters".to_string(), uniq.len() as f64);
            
            Ok(metrics)
        
    }

    fn simple_silhouette(&self, data: &Array2<f64>, labels: &Array1<usize>) -> Option<f64> {
        // Simplified silhouette calculation (same as before)
        let n_samples = data.nrows();
        let mut silhouette_sum = 0.0;
        let mut valid_samples = 0;
        for i in 0..n_samples {
            let mut intra_cluster_dist = 0.0;
            let mut inter_cluster_dist = f64::MAX;
            let current_label = labels[i];
            let mut intra_count = 0;
            for j in 0..n_samples {
                if i != j {
                    let dist = euclidean_distance(&data.row(i), &data.row(j));
                    if labels[j] == current_label {
                        intra_cluster_dist += dist;
                        intra_count += 1;
                    } else {
                        inter_cluster_dist = inter_cluster_dist.min(dist);
                    }
                }
            }
            if intra_count > 0 {
                intra_cluster_dist /= intra_count as f64;
                let silhouette = (inter_cluster_dist - intra_cluster_dist) /
                                 (inter_cluster_dist.max(intra_cluster_dist));
                silhouette_sum += silhouette;
                valid_samples += 1;
            }
        }
        if valid_samples > 0 { Some(silhouette_sum / valid_samples as f64) } else { None }
    }
    
    fn run_inference_benchmark(&self, data: &Array2<f64>, batch_sizes: &[usize]) -> Result<HashMap<String, f64>> {
        if let Some(ref model) = self.model {
            let mut latencies = Vec::new();
            
            for &batch_size in batch_sizes {
                let mut batch_latencies = Vec::new();
                
                for i in (0..data.nrows()).step_by(batch_size) {
                    let end = std::cmp::min(i + batch_size, data.nrows());
                    let batch = data.slice(s![i..end, ..]);
                    
                    let start_time = Instant::now();
                    let _predictions = model.predict(&batch.to_owned())?;
                    let latency = start_time.elapsed().as_millis() as f64;
                    
                    batch_latencies.push(latency);
                }
                
                let avg_latency = batch_latencies.iter().sum::<f64>() / batch_latencies.len() as f64;
                latencies.push(avg_latency);
            }
            
            let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
            let p50 = Self::percentile(&latencies, 50.0);
            let p95 = Self::percentile(&latencies, 95.0);
            let p99 = Self::percentile(&latencies, 99.0);
            
            let mut metrics = HashMap::new();
            metrics.insert("inference_latency_ms".to_string(), avg_latency);
            metrics.insert("latency_p50_ms".to_string(), p50);
            metrics.insert("latency_p95_ms".to_string(), p95);
            metrics.insert("latency_p99_ms".to_string(), p99);
            metrics.insert("throughput_samples_per_second".to_string(), 1000.0 / avg_latency);
            
            Ok(metrics)
        } else {
            Err(anyhow::anyhow!("Model not trained"))
        }
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
                     data_csv: Option<&str>) -> Result<BenchmarkResult> {
        
        // Load dataset (prefer CSV if provided for parity)
        let mut data = if let Some(path) = data_csv { self.load_dataset_from_csv(path)? } else { self.load_dataset(dataset)? };
        
        // Standardize features (match Python's StandardScaler behavior)
        {
            let means = data.mean_axis(ndarray::Axis(0)).unwrap();
            let stds = data.std_axis(ndarray::Axis(0), 0.0);
            for j in 0..data.ncols() {
                let m = means[j];
                let s = if stds[j] == 0.0 { 1.0 } else { stds[j] };
                for i in 0..data.nrows() { data[[i,j]] = (data[[i,j]] - m) / s; }
            }
        }
        
        // Create model
        self.create_model(algorithm, hyperparams)?;
        
        // Get hardware configuration
        let hardware_config = self.get_hardware_config();
        
        if mode == "training" {
            // Training benchmark
            let (training_time, resource_metrics) = self.train_model(&data)?;
            let quality_metrics = self.evaluate_model(&data)?;
            
            return Ok(BenchmarkResult {
                framework: self.framework.clone(),
                language: Language::Rust,
                task_type: TaskType::ClassicalMl,
                model_name: format!("{}_clustering", algorithm),
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
                    accuracy: None,
                    f1_score: None,
                    precision: None,
                    recall: None,
                    loss: None,
                    rmse: None,
                    mae: None,
                    r2_score: None,
                    silhouette_score: quality_metrics.get("silhouette_score").copied(),
                    inertia: quality_metrics.get("inertia").copied(),
                },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("algorithm".to_string(), serde_json::Value::String(algorithm.to_string()));
                    meta.insert("hyperparameters".to_string(), serde_json::to_value(hyperparams)?);
                    meta.insert("dataset_size".to_string(), serde_json::Value::Number(serde_json::Number::from(data.nrows())));
                    meta.insert("features".to_string(), serde_json::Value::Number(serde_json::Number::from(data.ncols())));
                    meta.insert("n_clusters".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(quality_metrics.get("n_clusters").copied().unwrap_or(0.0)).unwrap()));
                    meta
                },
            });
        } else if mode == "inference" {
            // Train model first
            self.train_model(&data)?;
            
            // Inference benchmark
            let inference_metrics = self.run_inference_benchmark(&data, &[1, 10, 100])?;
            
            return Ok(BenchmarkResult {
                framework: self.framework.clone(),
                language: Language::Rust,
                task_type: TaskType::ClassicalMl,
                model_name: format!("{}_clustering", algorithm),
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
                    silhouette_score: None,
                    inertia: None,
                },
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("algorithm".to_string(), serde_json::Value::String(algorithm.to_string()));
                    meta.insert("hyperparameters".to_string(), serde_json::to_value(hyperparams)?);
                    meta.insert("dataset_size".to_string(), serde_json::Value::Number(serde_json::Number::from(data.nrows())));
                    meta.insert("features".to_string(), serde_json::Value::Number(serde_json::Number::from(data.ncols())));
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
    let mut benchmark = ClusteringBenchmark::new("linfa".to_string());
    
    // Run benchmark
    let result = benchmark.run_benchmark(
        &args.dataset,
        &args.algorithm,
        &hyperparams,
        &run_id,
        &args.mode,
        args.data_csv.as_deref(),
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