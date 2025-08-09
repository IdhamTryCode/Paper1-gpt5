#!/usr/bin/env python3
"""
Visualization Script

This script creates comprehensive visualizations for benchmark results
in the Rust vs Python ML Benchmark System.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BenchmarkVisualizer:
    """Creates visualizations for benchmark results."""
    
    def __init__(self, statistical_results_path: str):
        self.statistical_results_path = statistical_results_path
        self.statistical_results = self._load_statistical_results()
        self.output_plots_dir = "plots"
        
        # Create output directory
        os.makedirs(self.output_plots_dir, exist_ok=True)
    
    def _load_statistical_results(self) -> Dict[str, Any]:
        """Load statistical analysis results from JSON file."""
        try:
            with open(self.statistical_results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load statistical results: {e}")
            sys.exit(1)
    
    def create_performance_comparison_plot(self) -> str:
        """Create performance comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Comparison: Rust vs Python', fontsize=16, fontweight='bold')
        
        metrics = ['training_time_seconds', 'inference_latency_ms', 
                  'throughput_samples_per_second', 'tokens_per_second']
        titles = ['Training Time (seconds)', 'Inference Latency (ms)', 
                 'Throughput (samples/sec)', 'Tokens per Second']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Extract data for this metric
            rust_means = []
            python_means = []
            task_names = []
            
            for task_type, analysis in self.statistical_results.get('task_analyses', {}).items():
                if metric in analysis.get('performance_metrics', {}):
                    comparison = analysis['performance_metrics'][metric]
                    if 'rust_stats' in comparison and 'python_stats' in comparison:
                        rust_means.append(comparison['rust_stats']['mean'])
                        python_means.append(comparison['python_stats']['mean'])
                        task_names.append(task_type.replace('_', ' ').title())
            
            if rust_means and python_means:
                x = np.arange(len(task_names))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, rust_means, width, label='Rust', alpha=0.8)
                bars2 = ax.bar(x + width/2, python_means, width, label='Python', alpha=0.8)
                
                ax.set_xlabel('Task Type')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.set_xticks(x)
                ax.set_xticklabels(task_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_plots_dir, 'performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_resource_usage_plot(self) -> str:
        """Create resource usage comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resource Usage Comparison: Rust vs Python', fontsize=16, fontweight='bold')
        
        metrics = ['peak_memory_mb', 'average_memory_mb', 'cpu_utilization_percent']
        titles = ['Peak Memory (MB)', 'Average Memory (MB)', 'CPU Utilization (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Extract data for this metric
            rust_means = []
            python_means = []
            task_names = []
            
            for task_type, analysis in self.statistical_results.get('task_analyses', {}).items():
                if metric in analysis.get('resource_metrics', {}):
                    comparison = analysis['resource_metrics'][metric]
                    if 'rust_stats' in comparison and 'python_stats' in comparison:
                        rust_means.append(comparison['rust_stats']['mean'])
                        python_means.append(comparison['python_stats']['mean'])
                        task_names.append(task_type.replace('_', ' ').title())
            
            if rust_means and python_means:
                x = np.arange(len(task_names))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, rust_means, width, label='Rust', alpha=0.8)
                bars2 = ax.bar(x + width/2, python_means, width, label='Python', alpha=0.8)
                
                ax.set_xlabel('Task Type')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.set_xticks(x)
                ax.set_xticklabels(task_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Add summary statistics in the last subplot
        ax = axes[1, 1]
        summary = self.statistical_results.get('summary', {})
        
        stats_text = f"""
        Summary Statistics:
        
        Total Comparisons: {summary.get('total_comparisons', 0)}
        Significant Comparisons: {summary.get('significant_comparisons', 0)}
        Significance Rate: {summary.get('significance_rate', 0):.1%}
        
        Rust Wins: {summary.get('rust_wins', 0)} ({summary.get('rust_win_rate', 0):.1%})
        Python Wins: {summary.get('python_wins', 0)} ({summary.get('python_win_rate', 0):.1%})
        Ties: {summary.get('ties', 0)}
        
        Average Effect Size: {summary.get('average_effect_size', 0):.3f}
        """
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Summary Statistics')
        ax.axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_plots_dir, 'resource_usage_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_quality_metrics_plot(self) -> str:
        """Create quality metrics comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quality Metrics Comparison: Rust vs Python', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        titles = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Extract data for this metric
            rust_means = []
            python_means = []
            task_names = []
            
            for task_type, analysis in self.statistical_results.get('task_analyses', {}).items():
                if metric in analysis.get('quality_metrics', {}):
                    comparison = analysis['quality_metrics'][metric]
                    if 'rust_stats' in comparison and 'python_stats' in comparison:
                        rust_means.append(comparison['rust_stats']['mean'])
                        python_means.append(comparison['python_stats']['mean'])
                        task_names.append(task_type.replace('_', ' ').title())
            
            if rust_means and python_means:
                x = np.arange(len(task_names))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, rust_means, width, label='Rust', alpha=0.8)
                bars2 = ax.bar(x + width/2, python_means, width, label='Python', alpha=0.8)
                
                ax.set_xlabel('Task Type')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.set_xticks(x)
                ax.set_xticklabels(task_names, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_plots_dir, 'quality_metrics_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_interactive_dashboard(self) -> str:
        """Create an interactive Plotly dashboard."""
        # Prepare data for the dashboard
        dashboard_data = []
        
        for task_type, analysis in self.statistical_results.get('task_analyses', {}).items():
            for metric_category, metrics in analysis.items():
                for metric_name, comparison in metrics.items():
                    if isinstance(comparison, dict) and 'rust_stats' in comparison and 'python_stats' in comparison:
                        dashboard_data.append({
                            'task_type': task_type,
                            'metric_category': metric_category,
                            'metric_name': metric_name,
                            'rust_mean': comparison['rust_stats']['mean'],
                            'python_mean': comparison['python_stats']['mean'],
                            'rust_std': comparison['rust_stats']['std'],
                            'python_std': comparison['python_stats']['std'],
                            'significant': comparison['comparison'].get('significant', False),
                            'p_value': comparison['comparison'].get('p_value', 1.0),
                            'cohens_d': comparison['comparison'].get('effect_size', {}).get('cohens_d', 0),
                            'improvement_ratio': comparison['comparison'].get('improvement_ratio', 1.0)
                        })
        
        df = pd.DataFrame(dashboard_data)
        
        # Create interactive scatter plot
        fig = px.scatter(df, x='python_mean', y='rust_mean', 
                        color='metric_category', size='cohens_d',
                        hover_data=['task_type', 'metric_name', 'significant', 'p_value'],
                        title='Rust vs Python Performance Comparison',
                        labels={'python_mean': 'Python Performance', 'rust_mean': 'Rust Performance'})
        
        # Add diagonal line for equal performance
        max_val = max(df['python_mean'].max(), df['rust_mean'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], 
                                mode='lines', name='Equal Performance',
                                line=dict(dash='dash', color='gray')))
        
        fig.update_layout(width=1000, height=600)
        
        dashboard_path = os.path.join(self.output_plots_dir, 'interactive_dashboard.html')
        fig.write_html(dashboard_path)
        
        return dashboard_path
    
    def create_effect_size_heatmap(self) -> str:
        """Create a heatmap of effect sizes."""
        # Prepare data for heatmap
        heatmap_data = []
        
        for task_type, analysis in self.statistical_results.get('task_analyses', {}).items():
            for metric_category, metrics in analysis.items():
                for metric_name, comparison in metrics.items():
                    if isinstance(comparison, dict) and 'comparison' in comparison:
                        cohens_d = comparison['comparison'].get('effect_size', {}).get('cohens_d', 0)
                        significant = comparison['comparison'].get('significant', False)
                        
                        heatmap_data.append({
                            'task_type': task_type.replace('_', ' ').title(),
                            'metric': f"{metric_category}_{metric_name}",
                            'effect_size': abs(cohens_d),
                            'significant': significant
                        })
        
        if heatmap_data:
            df = pd.DataFrame(heatmap_data)
            pivot_df = df.pivot(index='task_type', columns='metric', values='effect_size')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, cmap='RdYlBu_r', center=0.5,
                       cbar_kws={'label': 'Absolute Effect Size (|Cohen\'s d|)'})
            plt.title('Effect Size Heatmap: Rust vs Python Comparisons')
            plt.tight_layout()
            
            heatmap_path = os.path.join(self.output_plots_dir, 'effect_size_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return heatmap_path
        
        return None
    
    def create_all_visualizations(self) -> Dict[str, str]:
        """Create all visualizations and return paths."""
        logger.info("Creating visualizations...")
        
        visualization_paths = {}
        
        # Create performance comparison plot
        try:
            performance_path = self.create_performance_comparison_plot()
            visualization_paths['performance_comparison'] = performance_path
            logger.info(f"Created performance comparison plot: {performance_path}")
        except Exception as e:
            logger.error(f"Failed to create performance comparison plot: {e}")
        
        # Create resource usage plot
        try:
            resource_path = self.create_resource_usage_plot()
            visualization_paths['resource_usage'] = resource_path
            logger.info(f"Created resource usage plot: {resource_path}")
        except Exception as e:
            logger.error(f"Failed to create resource usage plot: {e}")
        
        # Create quality metrics plot
        try:
            quality_path = self.create_quality_metrics_plot()
            visualization_paths['quality_metrics'] = quality_path
            logger.info(f"Created quality metrics plot: {quality_path}")
        except Exception as e:
            logger.error(f"Failed to create quality metrics plot: {e}")
        
        # Create interactive dashboard
        try:
            dashboard_path = self.create_interactive_dashboard()
            visualization_paths['interactive_dashboard'] = dashboard_path
            logger.info(f"Created interactive dashboard: {dashboard_path}")
        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {e}")
        
        # Create effect size heatmap
        try:
            heatmap_path = self.create_effect_size_heatmap()
            if heatmap_path:
                visualization_paths['effect_size_heatmap'] = heatmap_path
                logger.info(f"Created effect size heatmap: {heatmap_path}")
        except Exception as e:
            logger.error(f"Failed to create effect size heatmap: {e}")
        
        return visualization_paths


def main():
    """Main function for creating visualizations."""
    parser = argparse.ArgumentParser(description="Create visualizations for benchmark results")
    parser.add_argument("--statistical-results", required=True, 
                       help="Path to statistical analysis results JSON file")
    parser.add_argument("--output-visualizations", required=True, 
                       help="Output file for visualization metadata")
    parser.add_argument("--output-plots", default="plots", 
                       help="Output directory for plot files")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BenchmarkVisualizer(args.statistical_results)
    visualizer.output_plots_dir = args.output_plots
    
    # Create all visualizations
    visualization_paths = visualizer.create_all_visualizations()
    
    # Save visualization metadata
    with open(args.output_visualizations, 'w') as f:
        json.dump(visualization_paths, f, indent=2)
    
    logger.info(f"Visualization metadata saved to: {args.output_visualizations}")
    logger.info(f"Plot files saved to: {args.output_plots}")
    
    # Print summary
    logger.info("Visualization Summary:")
    for viz_type, path in visualization_paths.items():
        logger.info(f"  {viz_type}: {path}")


if __name__ == "__main__":
    main() 