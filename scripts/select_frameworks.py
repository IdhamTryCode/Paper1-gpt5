#!/usr/bin/env python3
"""
Framework Selection Script

This script selects frameworks based on availability, functionality,
and selection criteria for the Rust vs Python ML Benchmark System.
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameworkSelector:
    """Selects frameworks based on criteria and availability."""
    
    def __init__(self, available_frameworks_path: str):
        self.available_frameworks_path = available_frameworks_path
        self.available_frameworks = self._load_available_frameworks()
        self.selection_criteria = {
            "maturity": self._evaluate_maturity,
            "performance": self._evaluate_performance,
            "ecosystem": self._evaluate_ecosystem,
            "documentation": self._evaluate_documentation,
            "community": self._evaluate_community
        }
    
    def _load_available_frameworks(self) -> Dict[str, Any]:
        """Load available frameworks from JSON file."""
        try:
            with open(self.available_frameworks_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load available frameworks: {e}")
            sys.exit(1)
    
    def _evaluate_maturity(self, framework_info: Dict[str, Any]) -> float:
        """Evaluate framework maturity (0-1 scale)."""
        framework_name = framework_info["framework"]
        category = framework_info["category"]
        
        # Maturity scores based on framework age, stability, and adoption
        maturity_scores = {
            "python": {
                "scikit-learn": 0.95,  # Very mature
                "pytorch": 0.85,       # Mature
                "tensorflow": 0.90,    # Very mature
                "transformers": 0.80,  # Mature
                "numpy": 0.95,         # Very mature
                "pandas": 0.90,        # Very mature
            },
            "rust": {
                "linfa": 0.60,         # Growing
                "smartcore": 0.50,     # Growing
                "tch": 0.70,           # Mature for Rust
                "burn": 0.40,          # New
                "candle": 0.65,        # Growing
                "candle-transformers": 0.55,  # Growing
            }
        }
        
        language = "python" if category in ["classical_ml", "deep_learning", "llm"] else "rust"
        return maturity_scores.get(language, {}).get(framework_name, 0.50)
    
    def _evaluate_performance(self, framework_info: Dict[str, Any]) -> float:
        """Evaluate framework performance potential (0-1 scale)."""
        framework_name = framework_info["framework"]
        category = framework_info["category"]
        
        # Performance scores based on language and framework characteristics
        performance_scores = {
            "python": {
                "scikit-learn": 0.75,  # Good performance
                "pytorch": 0.85,       # High performance
                "tensorflow": 0.80,    # High performance
                "transformers": 0.80,  # High performance
                "numpy": 0.70,         # Good performance
                "pandas": 0.65,        # Moderate performance
            },
            "rust": {
                "linfa": 0.85,         # High performance (Rust)
                "smartcore": 0.80,     # High performance (Rust)
                "tch": 0.90,           # Very high performance
                "burn": 0.85,          # High performance
                "candle": 0.90,        # Very high performance
                "candle-transformers": 0.85,  # High performance
            }
        }
        
        language = "python" if category in ["classical_ml", "deep_learning", "llm"] else "rust"
        return performance_scores.get(language, {}).get(framework_name, 0.70)
    
    def _evaluate_ecosystem(self, framework_info: Dict[str, Any]) -> float:
        """Evaluate framework ecosystem (0-1 scale)."""
        framework_name = framework_info["framework"]
        category = framework_info["category"]
        
        # Ecosystem scores based on available tools, libraries, and integrations
        ecosystem_scores = {
            "python": {
                "scikit-learn": 0.95,  # Excellent ecosystem
                "pytorch": 0.90,       # Excellent ecosystem
                "tensorflow": 0.95,    # Excellent ecosystem
                "transformers": 0.85,  # Good ecosystem
                "numpy": 0.90,         # Excellent ecosystem
                "pandas": 0.85,        # Good ecosystem
            },
            "rust": {
                "linfa": 0.60,         # Growing ecosystem
                "smartcore": 0.50,     # Limited ecosystem
                "tch": 0.75,           # Good ecosystem (PyTorch bindings)
                "burn": 0.40,          # New ecosystem
                "candle": 0.70,        # Growing ecosystem
                "candle-transformers": 0.65,  # Growing ecosystem
            }
        }
        
        language = "python" if category in ["classical_ml", "deep_learning", "llm"] else "rust"
        return ecosystem_scores.get(language, {}).get(framework_name, 0.60)
    
    def _evaluate_documentation(self, framework_info: Dict[str, Any]) -> float:
        """Evaluate framework documentation quality (0-1 scale)."""
        framework_name = framework_info["framework"]
        category = framework_info["category"]
        
        # Documentation scores
        documentation_scores = {
            "python": {
                "scikit-learn": 0.95,  # Excellent documentation
                "pytorch": 0.90,       # Excellent documentation
                "tensorflow": 0.95,    # Excellent documentation
                "transformers": 0.85,  # Good documentation
                "numpy": 0.90,         # Excellent documentation
                "pandas": 0.85,        # Good documentation
            },
            "rust": {
                "linfa": 0.70,         # Good documentation
                "smartcore": 0.60,     # Moderate documentation
                "tch": 0.80,           # Good documentation
                "burn": 0.50,          # Limited documentation
                "candle": 0.75,        # Good documentation
                "candle-transformers": 0.70,  # Good documentation
            }
        }
        
        language = "python" if category in ["classical_ml", "deep_learning", "llm"] else "rust"
        return documentation_scores.get(language, {}).get(framework_name, 0.60)
    
    def _evaluate_community(self, framework_info: Dict[str, Any]) -> float:
        """Evaluate framework community support (0-1 scale)."""
        framework_name = framework_info["framework"]
        category = framework_info["category"]
        
        # Community scores
        community_scores = {
            "python": {
                "scikit-learn": 0.95,  # Excellent community
                "pytorch": 0.90,       # Excellent community
                "tensorflow": 0.95,    # Excellent community
                "transformers": 0.85,  # Good community
                "numpy": 0.90,         # Excellent community
                "pandas": 0.85,        # Good community
            },
            "rust": {
                "linfa": 0.60,         # Growing community
                "smartcore": 0.50,     # Limited community
                "tch": 0.70,           # Good community
                "burn": 0.40,          # New community
                "candle": 0.65,        # Growing community
                "candle-transformers": 0.60,  # Growing community
            }
        }
        
        language = "python" if category in ["classical_ml", "deep_learning", "llm"] else "rust"
        return community_scores.get(language, {}).get(framework_name, 0.60)
    
    def select_frameworks(self, criteria: List[str], min_score: float = 0.5) -> Dict[str, Any]:
        """Select frameworks based on criteria."""
        selected_frameworks = {
            "python": {"classical_ml": [], "deep_learning": [], "llm": []},
            "rust": {"classical_ml": [], "deep_learning": [], "llm": []}
        }
        
        # Process available frameworks
        for language, categories in self.available_frameworks.items():
            for category, frameworks in categories.items():
                if isinstance(frameworks, dict):
                    for framework_name, framework_info in frameworks.items():
                        if framework_info.get("available", False) and framework_info.get("functional", False):
                            # Calculate overall score
                            scores = {}
                            for criterion in criteria:
                                if criterion in self.selection_criteria:
                                    scores[criterion] = self.selection_criteria[criterion](framework_info)
                            
                            # Calculate weighted average score
                            if scores:
                                overall_score = sum(scores.values()) / len(scores)
                                
                                if overall_score >= min_score:
                                    selected_framework = {
                                        "framework": framework_name,
                                        "category": category,
                                        "config": framework_info.get("config", {}),
                                        "scores": scores,
                                        "overall_score": overall_score,
                                        "version": framework_info.get("version", "unknown")
                                    }
                                    
                                    # Map category to appropriate section
                                    if category == "classical_ml":
                                        selected_frameworks[language]["classical_ml"].append(selected_framework)
                                    elif category in ["deep_learning", "pytorch", "tensorflow"]:
                                        selected_frameworks[language]["deep_learning"].append(selected_framework)
                                    elif category in ["llm", "transformers"]:
                                        selected_frameworks[language]["llm"].append(selected_framework)
        
        # Sort frameworks by overall score within each category
        for language in selected_frameworks:
            for category in selected_frameworks[language]:
                selected_frameworks[language][category].sort(
                    key=lambda x: x["overall_score"], reverse=True
                )
        
        return selected_frameworks
    
    def generate_selection_report(self, selected_frameworks: Dict[str, Any]) -> str:
        """Generate a human-readable selection report."""
        report = []
        report.append("# Framework Selection Report\n")
        
        total_selected = 0
        for language, categories in selected_frameworks.items():
            report.append(f"## {language.title()} Frameworks\n")
            
            for category, frameworks in categories.items():
                if frameworks:
                    report.append(f"### {category.replace('_', ' ').title()}\n")
                    
                    for framework in frameworks:
                        report.append(f"- **{framework['framework']}** (Score: {framework['overall_score']:.2f})")
                        report.append(f"  - Version: {framework['version']}")
                        report.append(f"  - Scores: {', '.join([f'{k}: {v:.2f}' for k, v in framework['scores'].items()])}")
                        report.append("")
                        total_selected += 1
                else:
                    report.append(f"### {category.replace('_', ' ').title()}\n")
                    report.append("- No frameworks selected\n")
        
        report.append(f"\n## Summary\n")
        report.append(f"- Total frameworks selected: {total_selected}")
        report.append(f"- Python frameworks: {sum(len(frameworks) for frameworks in selected_frameworks['python'].values())}")
        report.append(f"- Rust frameworks: {sum(len(frameworks) for frameworks in selected_frameworks['rust'].values())}")
        
        return "\n".join(report)


def main():
    """Main function for framework selection."""
    parser = argparse.ArgumentParser(description="Select frameworks based on criteria")
    parser.add_argument("--available", required=True, help="Path to available frameworks JSON file")
    parser.add_argument("--output", required=True, help="Output file for selected frameworks")
    parser.add_argument("--selection-criteria", default="maturity,performance,ecosystem", 
                       help="Comma-separated list of selection criteria")
    parser.add_argument("--min-score", type=float, default=0.5, 
                       help="Minimum score threshold for selection")
    
    args = parser.parse_args()
    
    # Parse selection criteria
    criteria = [c.strip() for c in args.selection_criteria.split(",")]
    
    # Select frameworks
    selector = FrameworkSelector(args.available)
    selected_frameworks = selector.select_frameworks(criteria, args.min_score)
    
    # Generate report
    report = selector.generate_selection_report(selected_frameworks)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(selected_frameworks, f, indent=2)
    
    # Print report
    print(report)
    
    logger.info(f"Selected frameworks saved to: {args.output}")
    
    # Exit with error code if no frameworks were selected
    total_selected = sum(len(frameworks) for language in selected_frameworks.values() 
                        for frameworks in language.values())
    if total_selected == 0:
        logger.error("No frameworks were selected!")
        sys.exit(1)


if __name__ == "__main__":
    main() 