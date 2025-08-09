#!/usr/bin/env python3
"""
Statistical Analysis Script

This script performs comprehensive statistical analysis on benchmark results
for the Rust vs Python ML Benchmark System.
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Performs statistical analysis on benchmark results."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def load_benchmark_results(self, results_path: str) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load benchmark results: {e}")
            sys.exit(1)
    
    def compare_frameworks(self, 
                          rust_results: List[Dict], 
                          python_results: List[Dict],
                          metric: str) -> Dict[str, Any]:
        """Compare Rust and Python results for a specific metric."""
        
        rust_values = [r['performance_metrics'].get(metric) for r in rust_results 
                      if r['performance_metrics'].get(metric) is not None]
        python_values = [p['performance_metrics'].get(metric) for p in python_results 
                        if p['performance_metrics'].get(metric) is not None]
        
        if not rust_values or not python_values:
            return {'error': f'Insufficient data for metric {metric}'}
        
        # Normality tests
        rust_normal = self._test_normality(rust_values)
        python_normal = self._test_normality(python_values)
        
        # Choose appropriate statistical test
        if rust_normal and python_normal:
            # Use t-test for normal distributions
            statistic, p_value = stats.ttest_ind(rust_values, python_values)
            test_type = 'independent_t_test'
        else:
            # Use Mann-Whitney U test for non-normal distributions
            statistic, p_value = stats.mannwhitneyu(rust_values, python_values, 
                                                   alternative='two-sided')
            test_type = 'mann_whitney_u'
        
        # Effect size calculations
        cohens_d = self._calculate_cohens_d(rust_values, python_values)
        cliffs_delta = self._calculate_cliffs_delta(rust_values, python_values)
        
        # Confidence intervals
        rust_ci = self._calculate_confidence_interval(rust_values)
        python_ci = self._calculate_confidence_interval(python_values)
        
        return {
            'metric': metric,
            'rust_stats': {
                'mean': np.mean(rust_values),
                'std': np.std(rust_values, ddof=1),
                'median': np.median(rust_values),
                'count': len(rust_values),
                'confidence_interval': rust_ci,
                'normal_distribution': rust_normal
            },
            'python_stats': {
                'mean': np.mean(python_values),
                'std': np.std(python_values, ddof=1),
                'median': np.median(python_values),
                'count': len(python_values),
                'confidence_interval': python_ci,
                'normal_distribution': python_normal
            },
            'comparison': {
                'test_type': test_type,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'effect_size': {
                    'cohens_d': cohens_d,
                    'cliffs_delta': cliffs_delta
                },
                'improvement_ratio': np.mean(rust_values) / np.mean(python_values)
                                   if np.mean(python_values) != 0 else None
            }
        }
    
    def _test_normality(self, values: List[float]) -> bool:
        """Test for normality using Shapiro-Wilk test."""
        if len(values) < 3:
            return False
        statistic, p_value = stats.shapiro(values)
        return p_value > self.alpha
    
    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _calculate_cliffs_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's delta effect size."""
        n1, n2 = len(group1), len(group2)
        greater = sum(1 for x1 in group1 for x2 in group2 if x1 > x2)
        less = sum(1 for x1 in group1 for x2 in group2 if x1 < x2)
        
        return (greater - less) / (n1 * n2)
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for the mean."""
        n = len(values)
        mean = np.mean(values)
        sem = stats.sem(values)  # Standard error of the mean
        
        # Use t-distribution for small samples
        t_value = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        margin_error = t_value * sem
        
        return (mean - margin_error, mean + margin_error)
    
    def analyze_resource_usage(self, rust_results: List[Dict], python_results: List[Dict]) -> Dict[str, Any]:
        """Analyze resource usage differences."""
        metrics = ['peak_memory_mb', 'average_memory_mb', 'cpu_utilization_percent']
        analysis = {}
        
        for metric in metrics:
            rust_values = [r['resource_metrics'].get(metric) for r in rust_results 
                          if r['resource_metrics'].get(metric) is not None]
            python_values = [p['resource_metrics'].get(metric) for p in python_results 
                            if p['resource_metrics'].get(metric) is not None]
            
            if rust_values and python_values:
                analysis[metric] = self.compare_frameworks(
                    [{'performance_metrics': {metric: v}} for v in rust_values],
                    [{'performance_metrics': {metric: v}} for v in python_values],
                    metric
                )
        
        return analysis
    
    def analyze_quality_metrics(self, rust_results: List[Dict], python_results: List[Dict]) -> Dict[str, Any]:
        """Analyze quality metrics differences."""
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'rmse', 'mae', 'r2_score']
        analysis = {}
        
        for metric in metrics:
            rust_values = [r['quality_metrics'].get(metric) for r in rust_results 
                          if r['quality_metrics'].get(metric) is not None]
            python_values = [p['quality_metrics'].get(metric) for p in python_results 
                            if p['quality_metrics'].get(metric) is not None]
            
            if rust_values and python_values:
                analysis[metric] = self.compare_frameworks(
                    [{'performance_metrics': {metric: v}} for v in rust_values],
                    [{'performance_metrics': {metric: v}} for v in python_values],
                    metric
                )
        
        return analysis
    
    def perform_comprehensive_analysis(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        logger.info("Starting comprehensive statistical analysis...")
        
        # Separate Rust and Python results
        rust_results = []
        python_results = []
        
        # Process results by task type
        task_analyses = {}
        
        for task_type, results in benchmark_results.items():
            if isinstance(results, dict):
                rust_task_results = results.get('rust', [])
                python_task_results = results.get('python', [])
                
                if rust_task_results and python_task_results:
                    task_analysis = {
                        'performance_metrics': {},
                        'resource_metrics': {},
                        'quality_metrics': {}
                    }
                    
                    # Analyze performance metrics
                    performance_metrics = ['training_time_seconds', 'inference_latency_ms', 
                                        'throughput_samples_per_second', 'tokens_per_second']
                    
                    for metric in performance_metrics:
                        rust_values = [r.get('performance_metrics', {}).get(metric) 
                                     for r in rust_task_results 
                                     if r.get('performance_metrics', {}).get(metric) is not None]
                        python_values = [p.get('performance_metrics', {}).get(metric) 
                                       for p in python_task_results 
                                       if p.get('performance_metrics', {}).get(metric) is not None]
                        
                        if rust_values and python_values:
                            task_analysis['performance_metrics'][metric] = self.compare_frameworks(
                                [{'performance_metrics': {metric: v}} for v in rust_values],
                                [{'performance_metrics': {metric: v}} for v in python_values],
                                metric
                            )
                    
                    # Analyze resource metrics
                    task_analysis['resource_metrics'] = self.analyze_resource_usage(
                        rust_task_results, python_task_results
                    )
                    
                    # Analyze quality metrics
                    task_analysis['quality_metrics'] = self.analyze_quality_metrics(
                        rust_task_results, python_task_results
                    )
                    
                    task_analyses[task_type] = task_analysis
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(task_analyses)
        
        return {
            'task_analyses': task_analyses,
            'summary': summary,
            'confidence_level': self.confidence_level,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _generate_summary_statistics(self, task_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all tasks."""
        total_comparisons = 0
        significant_comparisons = 0
        rust_wins = 0
        python_wins = 0
        ties = 0
        
        effect_sizes = []
        
        for task_type, analysis in task_analyses.items():
            for metric_category, metrics in analysis.items():
                for metric_name, comparison in metrics.items():
                    if isinstance(comparison, dict) and 'comparison' in comparison:
                        total_comparisons += 1
                        
                        if comparison['comparison'].get('significant', False):
                            significant_comparisons += 1
                        
                        improvement_ratio = comparison['comparison'].get('improvement_ratio')
                        if improvement_ratio is not None:
                            if improvement_ratio > 1.0:
                                rust_wins += 1
                            elif improvement_ratio < 1.0:
                                python_wins += 1
                            else:
                                ties += 1
                        
                        cohens_d = comparison['comparison'].get('effect_size', {}).get('cohens_d')
                        if cohens_d is not None:
                            effect_sizes.append(abs(cohens_d))
        
        return {
            'total_comparisons': total_comparisons,
            'significant_comparisons': significant_comparisons,
            'significance_rate': significant_comparisons / total_comparisons if total_comparisons > 0 else 0,
            'rust_wins': rust_wins,
            'python_wins': python_wins,
            'ties': ties,
            'rust_win_rate': rust_wins / total_comparisons if total_comparisons > 0 else 0,
            'python_win_rate': python_wins / total_comparisons if total_comparisons > 0 else 0,
            'average_effect_size': np.mean(effect_sizes) if effect_sizes else 0,
            'median_effect_size': np.median(effect_sizes) if effect_sizes else 0
        }


def main():
    """Main function for statistical analysis."""
    parser = argparse.ArgumentParser(description="Perform statistical analysis on benchmark results")
    parser.add_argument("--results", required=True, help="Path to benchmark results JSON file")
    parser.add_argument("--output", required=True, help="Output file for statistical analysis results")
    parser.add_argument("--confidence-level", type=float, default=0.95, 
                       help="Confidence level for statistical tests")
    
    args = parser.parse_args()
    
    # Load benchmark results
    benchmark_results = {}
    try:
        with open(args.results, 'r') as f:
            benchmark_results = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load benchmark results: {e}")
        sys.exit(1)
    
    # Perform statistical analysis
    analyzer = StatisticalAnalyzer(args.confidence_level)
    analysis_results = analyzer.perform_comprehensive_analysis(benchmark_results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Print summary
    summary = analysis_results['summary']
    logger.info("Statistical Analysis Summary:")
    logger.info(f"  Total comparisons: {summary['total_comparisons']}")
    logger.info(f"  Significant comparisons: {summary['significant_comparisons']}")
    logger.info(f"  Significance rate: {summary['significance_rate']:.2%}")
    logger.info(f"  Rust wins: {summary['rust_wins']} ({summary['rust_win_rate']:.2%})")
    logger.info(f"  Python wins: {summary['python_wins']} ({summary['python_win_rate']:.2%})")
    logger.info(f"  Ties: {summary['ties']}")
    logger.info(f"  Average effect size: {summary['average_effect_size']:.3f}")
    
    logger.info(f"Statistical analysis results saved to: {args.output}")


if __name__ == "__main__":
    main() 