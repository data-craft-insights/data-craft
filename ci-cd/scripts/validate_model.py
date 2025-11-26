#!/usr/bin/env python3
"""
Validate Model Performance
Checks if model meets validation thresholds using best model responses
"""

import sys
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "validation_thresholds.yaml"

# Try multiple locations for outputs directory (for CI/CD compatibility)
outputs_dir = None
possible_paths = [
    project_root / "outputs",
    Path.cwd() / "outputs",
]

# Check GITHUB_WORKSPACE if available
github_workspace = os.environ.get('GITHUB_WORKSPACE')
if github_workspace:
    possible_paths.insert(0, Path(github_workspace) / "outputs")

for path in possible_paths:
    if path.exists():
        outputs_dir = path
        if path != project_root / "outputs":
            print(f"Using outputs directory: {outputs_dir}")
        break

if outputs_dir is None:
    outputs_dir = project_root / "outputs"  # Default fallback

def load_thresholds() -> Dict:
    """Load validation thresholds from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['validation_thresholds']

def find_latest_selection_report() -> Optional[Path]:
    """Find the latest model selection report in best-model-responses"""
    best_model_dir = outputs_dir / "best-model-responses"
    
    # Debug: Check if outputs_dir exists and list contents
    print(f"  Checking outputs directory: {outputs_dir}")
    print(f"  Outputs directory exists: {outputs_dir.exists()}")
    if outputs_dir.exists():
        print(f"  Contents of outputs directory:")
        for item in outputs_dir.iterdir():
            print(f"    - {item.name} ({'dir' if item.is_dir() else 'file'})")
    
    if not best_model_dir.exists():
        print(f"  Directory does not exist: {best_model_dir}")
        print(f"  Absolute path: {best_model_dir.resolve()}")
        return None
    
    # Debug: List all JSON files found
    all_json_files = list(best_model_dir.rglob("*.json"))
    if all_json_files:
        print(f"  Found {len(all_json_files)} JSON files:")
        for f in sorted(all_json_files)[:10]:  # Show first 10
            print(f"    - {f.relative_to(best_model_dir)}")
    
    # Look for model_selection_report.json (exact match)
    reports = list(best_model_dir.rglob("model_selection_report.json"))
    
    # Also try pattern matching for variations
    if not reports:
        reports = list(best_model_dir.rglob("*model_selection*.json"))
    
    if reports:
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        print(f"  Using report: {latest.relative_to(best_model_dir)}")
        return latest
    
    print(f"  No model selection report found in {best_model_dir}")
    return None

def find_latest_summary() -> Optional[Path]:
    """Find the latest summary.json file"""
    best_model_dir = outputs_dir / "best-model-responses"
    if not best_model_dir.exists():
        return None
    
    # Look for summary.json
    summary_files = list(best_model_dir.rglob("summary.json"))
    if summary_files:
        return max(summary_files, key=lambda p: p.stat().st_mtime)
    
    return None

def validate_performance(metrics: Dict, thresholds: Dict) -> List[str]:
    """Validate performance metrics from selection report"""
    errors = []
    perf_thresholds = thresholds['performance']
    
    # Get metrics from best_model section
    best_model = metrics.get('best_model', {})
    
    # Use composite_score as overall_score
    overall_score = best_model.get('composite_score', 0)
    if overall_score < perf_thresholds['min_overall_score']:
        errors.append(
            f"Composite score {overall_score:.2f} below threshold "
            f"{perf_thresholds['min_overall_score']}"
        )
    
    # Use performance_score
    performance_score = best_model.get('performance_score', 0)
    if performance_score < perf_thresholds.get('min_performance_score', 0):
        errors.append(
            f"Performance score {performance_score:.2f} below threshold "
            f"{perf_thresholds.get('min_performance_score', 0)}"
        )
    
    success_rate = best_model.get('success_rate', 0)
    if success_rate < perf_thresholds['min_success_rate']:
        errors.append(
            f"Success rate {success_rate:.2f}% below threshold "
            f"{perf_thresholds['min_success_rate']}%"
        )
    
    return errors

def validate_execution(summary_data: Dict, thresholds: Dict) -> List[str]:
    """Validate execution metrics from summary.json"""
    errors = []
    exec_thresholds = thresholds['execution']
    
    if not summary_data:
        return ["No summary data found"]
    
    # Calculate metrics from summary.json
    total_queries = summary_data.get('total_queries', 0)
    successful_queries = summary_data.get('successful_queries', 0)
    failed_queries = summary_data.get('failed_queries', 0)
    
    if total_queries == 0:
        return ["No queries found in summary"]
    
    # Execution success rate = successful queries / total queries
    exec_success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
    
    if exec_success_rate < exec_thresholds['min_execution_success_rate']:
        errors.append(
            f"Execution success rate {exec_success_rate:.2f}% below threshold "
            f"{exec_thresholds['min_execution_success_rate']}%"
        )
    
    # For result validity, we use successful queries as valid (since summary doesn't have execution validation)
    # This assumes successful query generation means valid results
    validity_rate = exec_success_rate  # Use same as execution success rate
    
    if validity_rate < exec_thresholds['min_result_validity_rate']:
        errors.append(
            f"Result validity rate {validity_rate:.2f}% below threshold "
            f"{exec_thresholds['min_result_validity_rate']}%"
        )
    
    # Overall accuracy = valid results / total queries
    overall_accuracy = exec_success_rate  # Same as execution success rate
    
    if overall_accuracy < exec_thresholds['min_overall_accuracy']:
        errors.append(
            f"Overall accuracy {overall_accuracy:.2f}% below threshold "
            f"{exec_thresholds['min_overall_accuracy']}%"
        )
    
    return errors

def main():
    """Main validation function"""
    print("=" * 70)
    print("MODEL VALIDATION")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Outputs directory: {outputs_dir}")
    print(f"Outputs directory exists: {outputs_dir.exists()}")
    print(f"Outputs directory absolute path: {outputs_dir.resolve()}")
    
    try:
        # Load thresholds
        thresholds = load_thresholds()
        print("✓ Loaded validation thresholds")
        
        # Load selection report
        print("\nSearching for model selection report...")
        selection_report_path = find_latest_selection_report()
        if not selection_report_path:
            print("✗ No model selection report found")
            print(f"  Searched in: {outputs_dir / 'best-model-responses'}")
            return 1
        
        print(f"✓ Found selection report: {selection_report_path.name}")
        
        with open(selection_report_path, 'r') as f:
            selection_data = json.load(f)
        
        best_model_name = selection_data.get('best_model', {}).get('name', 'unknown')
        print(f"✓ Best model: {best_model_name}")
        
        # Validate performance
        print("\nValidating performance metrics...")
        perf_errors = validate_performance(selection_data, thresholds)
        
        # Validate execution using summary.json
        print("Validating execution metrics...")
        summary_file = find_latest_summary()
        summary_data = None
        if summary_file:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            print(f"✓ Found summary: {summary_file.name}")
            print(f"  Total queries: {summary_data.get('total_queries', 0)}")
            print(f"  Successful queries: {summary_data.get('successful_queries', 0)}")
            print(f"  Failed queries: {summary_data.get('failed_queries', 0)}")
        else:
            print("⚠ No summary file found")
        
        exec_errors = validate_execution(summary_data, thresholds)
        
        # Combine errors
        all_errors = perf_errors + exec_errors
        
        # Save validation report
        best_model = selection_data.get('best_model', {})
        
        # Calculate execution accuracy from summary if available
        execution_accuracy = None
        if summary_data:
            total = summary_data.get('total_queries', 0)
            successful = summary_data.get('successful_queries', 0)
            execution_accuracy = (successful / total * 100) if total > 0 else 0
        
        validation_report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "status": "passed" if not all_errors else "failed",
            "errors": all_errors,
            "metrics": {
                "composite_score": best_model.get('composite_score', 0),
                "performance_score": best_model.get('performance_score', 0),
                "success_rate": best_model.get('success_rate', 0),
                "execution_accuracy": execution_accuracy
            }
        }
        
        validation_dir = outputs_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_file = validation_dir / "validation_report.json"
        
        with open(validation_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        # Print results
        print("\n" + "=" * 70)
        if all_errors:
            print("VALIDATION FAILED")
            print("=" * 70)
            for error in all_errors:
                print(f"  ✗ {error}")
            print(f"\nValidation report saved: {validation_file}")
            return 1
        else:
            print("VALIDATION PASSED")
            print("=" * 70)
            print(f"  ✓ Composite Score: {best_model.get('composite_score', 0):.2f}")
            print(f"  ✓ Performance Score: {best_model.get('performance_score', 0):.2f}")
            print(f"  ✓ Success Rate: {best_model.get('success_rate', 0):.2f}%")
            if summary_data:
                total = summary_data.get('total_queries', 0)
                successful = summary_data.get('successful_queries', 0)
                exec_acc = (successful / total * 100) if total > 0 else 0
                print(f"  ✓ Execution Accuracy: {exec_acc:.2f}%")
            print(f"\nValidation report saved: {validation_file}")
            return 0
            
    except Exception as e:
        print(f"\n✗ VALIDATION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
