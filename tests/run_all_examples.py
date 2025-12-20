#!/usr/bin/env python3
"""
Run All BayeSED3 Python Interface Examples

This script runs all the Python interface examples extracted from README.md.
Each example demonstrates different aspects of BayeSED3 functionality.
"""

import sys
import traceback
from pathlib import Path

# Add the parent directory to the path so we can import bayesed
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_example(example_name, example_function):
    """Run a single example with error handling."""
    print(f"\n{'='*60}")
    print(f"Running {example_name}")
    print(f"{'='*60}")
    
    try:
        result = example_function()
        print(f"✅ {example_name} completed successfully!")
        return True
    except Exception as e:
        print(f"❌ {example_name} failed with error:")
        print(f"   {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all examples."""
    print("BayeSED3 Python Interface Examples")
    print("=" * 60)
    
    examples = []
    results = []
    
    # Import and run each example
    try:
        from quick_start import test_quick_start
        examples.append(("Quick Start", test_quick_start))
    except ImportError as e:
        print(f"Could not import quick_start: {e}")
    
    try:
        from test_agn_fitting import test_agn_fitting
        examples.append(("AGN Fitting", test_agn_fitting))
    except ImportError as e:
        print(f"Could not import test_agn_fitting: {e}")
    
    try:
        from test_data_arrays import test_data_arrays
        examples.append(("Data Arrays", test_data_arrays))
    except ImportError as e:
        print(f"Could not import test_data_arrays: {e}")
    
    try:
        from test_custom_model import test_custom_model_dust_emission, test_custom_model_full_agn
        examples.append(("Custom Model - Dust Emission", test_custom_model_dust_emission))
        examples.append(("Custom Model - Full AGN", test_custom_model_full_agn))
    except ImportError as e:
        print(f"Could not import test_custom_model: {e}")
    
    try:
        from test_multi_model_comparison import test_multi_model_comparison
        examples.append(("Multi-Model Comparison", test_multi_model_comparison))
    except ImportError as e:
        print(f"Could not import test_multi_model_comparison: {e}")
    
    try:
        from test_advanced_analytics import test_advanced_analytics, test_evidence_analysis
        examples.append(("Advanced Analytics", test_advanced_analytics))
        examples.append(("Evidence Analysis", test_evidence_analysis))
    except ImportError as e:
        print(f"Could not import test_advanced_analytics: {e}")
    
    # Run all examples
    for example_name, example_function in examples:
        success = run_example(example_name, example_function)
        results.append((example_name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for example_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {example_name}")
    
    print(f"\nTotal: {successful}/{total} examples passed")
    
    if successful == total:
        print("🎉 All examples completed successfully!")
        return 0
    else:
        print("⚠️  Some examples failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())