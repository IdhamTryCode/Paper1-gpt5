#!/usr/bin/env python3
"""
Framework Availability Check Script

This script checks the availability and functionality of frameworks
for the Rust vs Python ML Benchmark System.
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any
import importlib
import subprocess
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameworkAvailabilityChecker:
    """Checks framework availability and basic functionality."""
    
    def __init__(self, configs_path: str):
        self.configs_path = configs_path
        self.configs = self._load_configs()
        self.availability_results = {}
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load framework configurations from JSON file."""
        try:
            with open(self.configs_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            sys.exit(1)
    
    def check_python_framework_availability(self) -> Dict[str, Any]:
        """Check availability of Python frameworks."""
        results = {
            "available": [],
            "unavailable": [],
            "functional": [],
            "non_functional": []
        }
        
        if "python" not in self.configs:
            logger.warning("No Python frameworks in configuration")
            return results
        
        python_configs = self.configs["python"]
        
        for category, frameworks in python_configs.items():
            if isinstance(frameworks, dict):
                for framework_name, framework_config in frameworks.items():
                    availability = self._check_python_framework_availability(framework_name, framework_config)
                    if availability["available"]:
                        results["available"].append({
                            "category": category,
                            "framework": framework_name,
                            "config": framework_config,
                            "version": availability.get("version", "unknown")
                        })
                        
                        # Check functionality
                        if self._test_python_framework_functionality(framework_name, category):
                            results["functional"].append({
                                "category": category,
                                "framework": framework_name,
                                "config": framework_config
                            })
                        else:
                            results["non_functional"].append({
                                "category": category,
                                "framework": framework_name,
                                "config": framework_config,
                                "reason": "Functionality test failed"
                            })
                    else:
                        results["unavailable"].append({
                            "category": category,
                            "framework": framework_name,
                            "config": framework_config,
                            "reason": availability.get("reason", "Unknown error")
                        })
        
        return results
    
    def check_rust_framework_availability(self) -> Dict[str, Any]:
        """Check availability of Rust frameworks."""
        results = {
            "available": [],
            "unavailable": [],
            "functional": [],
            "non_functional": []
        }
        
        if "rust" not in self.configs:
            logger.warning("No Rust frameworks in configuration")
            return results
        
        rust_configs = self.configs["rust"]
        
        for category, frameworks in rust_configs.items():
            if isinstance(frameworks, dict):
                for framework_name, framework_config in frameworks.items():
                    availability = self._check_rust_framework_availability(framework_name, framework_config)
                    if availability["available"]:
                        results["available"].append({
                            "category": category,
                            "framework": framework_name,
                            "config": framework_config,
                            "version": availability.get("version", "unknown")
                        })
                        
                        # Check functionality
                        if self._test_rust_framework_functionality(framework_name, category):
                            results["functional"].append({
                                "category": category,
                                "framework": framework_name,
                                "config": framework_config
                            })
                        else:
                            results["non_functional"].append({
                                "category": category,
                                "framework": framework_name,
                                "config": framework_config,
                                "reason": "Functionality test failed"
                            })
                    else:
                        results["unavailable"].append({
                            "category": category,
                            "framework": framework_name,
                            "config": framework_config,
                            "reason": availability.get("reason", "Unknown error")
                        })
        
        return results
    
    def _check_python_framework_availability(self, framework_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check availability of a specific Python framework."""
        try:
            if framework_name == "scikit-learn":
                import sklearn
                return {
                    "available": True,
                    "version": sklearn.__version__
                }
            elif framework_name == "pytorch":
                import torch
                return {
                    "available": True,
                    "version": torch.__version__
                }
            elif framework_name == "tensorflow":
                import tensorflow as tf
                return {
                    "available": True,
                    "version": tf.__version__
                }
            elif framework_name == "transformers":
                import transformers
                return {
                    "available": True,
                    "version": transformers.__version__
                }
            else:
                # Try to import the framework
                module = importlib.import_module(framework_name)
                version = getattr(module, '__version__', 'unknown')
                return {
                    "available": True,
                    "version": version
                }
        except ImportError:
            return {
                "available": False,
                "reason": "Module not found"
            }
        except Exception as e:
            return {
                "available": False,
                "reason": str(e)
            }
    
    def _check_rust_framework_availability(self, framework_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check availability of a specific Rust framework."""
        try:
            # Check if framework is in Cargo.toml
            cargo_toml_path = Path("Cargo.toml")
            if cargo_toml_path.exists():
                with open(cargo_toml_path, 'r') as f:
                    cargo_content = f.read()
                    if framework_name.lower() in cargo_content:
                        # Try to build the project to check if it compiles
                        result = subprocess.run(
                            ["cargo", "check", "--quiet"],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            return {
                                "available": True,
                                "version": "compiled successfully"
                            }
                        else:
                            return {
                                "available": False,
                                "reason": f"Compilation failed: {result.stderr}"
                            }
                    else:
                        return {
                            "available": False,
                            "reason": "Framework not found in Cargo.toml"
                        }
            else:
                return {
                    "available": False,
                    "reason": "Cargo.toml not found"
                }
        except subprocess.TimeoutExpired:
            return {
                "available": False,
                "reason": "Compilation timeout"
            }
        except Exception as e:
            return {
                "available": False,
                "reason": str(e)
            }
    
    def _test_python_framework_functionality(self, framework_name: str, category: str) -> bool:
        """Test basic functionality of a Python framework."""
        try:
            if category == "classical_ml" and framework_name == "scikit-learn":
                # Test basic sklearn functionality
                from sklearn.datasets import make_classification
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LogisticRegression
                
                X, y = make_classification(n_samples=100, n_features=20, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = LogisticRegression()
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                logger.info(f"✓ {framework_name} functionality test passed (accuracy: {score:.3f})")
                return True
                
            elif category == "deep_learning" and framework_name == "pytorch":
                # Test basic PyTorch functionality
                import torch
                import torch.nn as nn
                
                model = nn.Linear(10, 1)
                x = torch.randn(5, 10)
                y = model(x)
                
                logger.info(f"✓ {framework_name} functionality test passed")
                return True
                
            elif category == "deep_learning" and framework_name == "tensorflow":
                # Test basic TensorFlow functionality
                import tensorflow as tf
                
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(1, input_shape=(10,))
                ])
                x = tf.random.normal((5, 10))
                y = model(x)
                
                logger.info(f"✓ {framework_name} functionality test passed")
                return True
                
            elif category == "llm" and framework_name == "transformers":
                # Test basic transformers functionality
                from transformers import AutoTokenizer, AutoModel
                
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                model = AutoModel.from_pretrained("bert-base-uncased")
                
                text = "Hello, world!"
                inputs = tokenizer(text, return_tensors="pt")
                outputs = model(**inputs)
                
                logger.info(f"✓ {framework_name} functionality test passed")
                return True
                
            else:
                # Generic test - just try to import and create a simple object
                module = importlib.import_module(framework_name)
                logger.info(f"✓ {framework_name} basic import test passed")
                return True
                
        except Exception as e:
            logger.warning(f"✗ {framework_name} functionality test failed: {e}")
            return False
    
    def _test_rust_framework_functionality(self, framework_name: str, category: str) -> bool:
        """Test basic functionality of a Rust framework."""
        try:
            # For now, we'll just check if the framework can be compiled
            # In a real implementation, you would run actual functionality tests
            
            if category == "classical_ml" and framework_name == "linfa":
                # Test basic Linfa functionality
                result = subprocess.run(
                    ["cargo", "test", "--lib", "--quiet"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    logger.info(f"✓ {framework_name} functionality test passed")
                    return True
                else:
                    logger.warning(f"✗ {framework_name} functionality test failed: {result.stderr}")
                    return False
                    
            else:
                # Generic test - just check if it compiles
                result = subprocess.run(
                    ["cargo", "check", "--quiet"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    logger.info(f"✓ {framework_name} compilation test passed")
                    return True
                else:
                    logger.warning(f"✗ {framework_name} compilation test failed: {result.stderr}")
                    return False
                    
        except subprocess.TimeoutExpired:
            logger.warning(f"✗ {framework_name} functionality test timed out")
            return False
        except Exception as e:
            logger.warning(f"✗ {framework_name} functionality test failed: {e}")
            return False
    
    def check_all(self) -> Dict[str, Any]:
        """Check availability of all frameworks."""
        logger.info("Starting framework availability check...")
        
        python_results = self.check_python_framework_availability()
        rust_results = self.check_rust_framework_availability()
        
        # Combine results
        combined_results = {
            "python": python_results,
            "rust": rust_results,
            "summary": {
                "total_available": len(python_results["available"]) + len(rust_results["available"]),
                "total_unavailable": len(python_results["unavailable"]) + len(rust_results["unavailable"]),
                "total_functional": len(python_results["functional"]) + len(rust_results["functional"]),
                "total_non_functional": len(python_results["non_functional"]) + len(rust_results["non_functional"])
            }
        }
        
        # Log summary
        logger.info(f"Availability check complete:")
        logger.info(f"  Available frameworks: {combined_results['summary']['total_available']}")
        logger.info(f"  Unavailable frameworks: {combined_results['summary']['total_unavailable']}")
        logger.info(f"  Functional frameworks: {combined_results['summary']['total_functional']}")
        logger.info(f"  Non-functional frameworks: {combined_results['summary']['total_non_functional']}")
        
        return combined_results


def main():
    """Main function for framework availability check."""
    parser = argparse.ArgumentParser(description="Check framework availability")
    parser.add_argument("--configs", required=True, help="Path to framework configurations JSON file")
    parser.add_argument("--output", required=True, help="Output file for availability results")
    
    args = parser.parse_args()
    
    # Check framework availability
    checker = FrameworkAvailabilityChecker(args.configs)
    results = checker.check_all()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Availability results saved to: {args.output}")
    
    # Exit with error code if there are no available frameworks
    if results["summary"]["total_available"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main() 