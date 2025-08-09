#!/usr/bin/env python3
"""
Framework Validation Script

This script validates framework configurations and checks availability
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameworkValidator:
    """Validates framework configurations and availability."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.validation_results = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load framework configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def validate_python_frameworks(self) -> Dict[str, Any]:
        """Validate Python framework configurations."""
        results = {
            "valid": [],
            "invalid": [],
            "missing": []
        }
        
        if "python" not in self.config:
            logger.warning("No Python frameworks configured")
            return results
        
        python_config = self.config["python"]
        
        for category, frameworks in python_config.items():
            logger.info(f"Validating Python {category} frameworks...")
            
            if isinstance(frameworks, dict):
                for framework_name, framework_config in frameworks.items():
                    if self._validate_python_framework(framework_name, framework_config):
                        results["valid"].append({
                            "category": category,
                            "framework": framework_name,
                            "config": framework_config
                        })
                    else:
                        results["invalid"].append({
                            "category": category,
                            "framework": framework_name,
                            "config": framework_config,
                            "reason": "Validation failed"
                        })
            elif isinstance(frameworks, str):
                # Handle simple string configurations
                if self._validate_python_framework(frameworks, {}):
                    results["valid"].append({
                        "category": category,
                        "framework": frameworks,
                        "config": {}
                    })
                else:
                    results["invalid"].append({
                        "category": category,
                        "framework": frameworks,
                        "config": {},
                        "reason": "Validation failed"
                    })
        
        return results
    
    def validate_rust_frameworks(self) -> Dict[str, Any]:
        """Validate Rust framework configurations."""
        results = {
            "valid": [],
            "invalid": [],
            "missing": []
        }
        
        if "rust" not in self.config:
            logger.warning("No Rust frameworks configured")
            return results
        
        rust_config = self.config["rust"]
        
        for category, frameworks in rust_config.items():
            logger.info(f"Validating Rust {category} frameworks...")
            
            if isinstance(frameworks, dict):
                for framework_name, framework_config in frameworks.items():
                    if self._validate_rust_framework(framework_name, framework_config):
                        results["valid"].append({
                            "category": category,
                            "framework": framework_name,
                            "config": framework_config
                        })
                    else:
                        results["invalid"].append({
                            "category": category,
                            "framework": framework_name,
                            "config": framework_config,
                            "reason": "Validation failed"
                        })
            elif isinstance(frameworks, str):
                # Handle simple string configurations
                if self._validate_rust_framework(frameworks, {}):
                    results["valid"].append({
                        "category": category,
                        "framework": frameworks,
                        "config": {}
                    })
                else:
                    results["invalid"].append({
                        "category": category,
                        "framework": frameworks,
                        "config": {},
                        "reason": "Validation failed"
                    })
        
        return results
    
    def _validate_python_framework(self, framework_name: str, config: Dict[str, Any]) -> bool:
        """Validate a specific Python framework."""
        try:
            # Check if framework can be imported
            if framework_name == "scikit-learn":
                import sklearn
                logger.info(f"✓ {framework_name} is available (version: {sklearn.__version__})")
                return True
            elif framework_name == "pytorch":
                import torch
                logger.info(f"✓ {framework_name} is available (version: {torch.__version__})")
                return True
            elif framework_name == "tensorflow":
                import tensorflow as tf
                logger.info(f"✓ {framework_name} is available (version: {tf.__version__})")
                return True
            elif framework_name == "transformers":
                import transformers
                logger.info(f"✓ {framework_name} is available (version: {transformers.__version__})")
                return True
            else:
                # Try to import the framework
                importlib.import_module(framework_name)
                logger.info(f"✓ {framework_name} is available")
                return True
        except ImportError:
            logger.warning(f"✗ {framework_name} is not available")
            return False
        except Exception as e:
            logger.error(f"✗ Error validating {framework_name}: {e}")
            return False
    
    def _validate_rust_framework(self, framework_name: str, config: Dict[str, Any]) -> bool:
        """Validate a specific Rust framework."""
        try:
            # Check if framework is available in Cargo.toml
            cargo_toml_path = Path("Cargo.toml")
            if cargo_toml_path.exists():
                with open(cargo_toml_path, 'r') as f:
                    cargo_content = f.read()
                    if framework_name.lower() in cargo_content:
                        logger.info(f"✓ {framework_name} is configured in Cargo.toml")
                        return True
                    else:
                        logger.warning(f"✗ {framework_name} is not configured in Cargo.toml")
                        return False
            else:
                logger.warning("Cargo.toml not found, cannot validate Rust frameworks")
                return False
        except Exception as e:
            logger.error(f"✗ Error validating {framework_name}: {e}")
            return False
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate all frameworks and return results."""
        logger.info("Starting framework validation...")
        
        python_results = self.validate_python_frameworks()
        rust_results = self.validate_rust_frameworks()
        
        # Combine results
        combined_results = {
            "python": python_results,
            "rust": rust_results,
            "summary": {
                "total_valid": len(python_results["valid"]) + len(rust_results["valid"]),
                "total_invalid": len(python_results["invalid"]) + len(rust_results["invalid"]),
                "total_missing": len(python_results["missing"]) + len(rust_results["missing"])
            }
        }
        
        # Log summary
        logger.info(f"Validation complete:")
        logger.info(f"  Valid frameworks: {combined_results['summary']['total_valid']}")
        logger.info(f"  Invalid frameworks: {combined_results['summary']['total_invalid']}")
        logger.info(f"  Missing frameworks: {combined_results['summary']['total_missing']}")
        
        return combined_results


def main():
    """Main function for framework validation."""
    parser = argparse.ArgumentParser(description="Validate framework configurations")
    parser.add_argument("--config", required=True, help="Path to frameworks configuration file")
    parser.add_argument("--output", required=True, help="Output file for validation results")
    
    args = parser.parse_args()
    
    # Validate frameworks
    validator = FrameworkValidator(args.config)
    results = validator.validate_all()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Validation results saved to: {args.output}")
    
    # Exit with error code if there are invalid frameworks
    if results["summary"]["total_invalid"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main() 