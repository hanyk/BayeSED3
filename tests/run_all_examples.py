#!/usr/bin/env python3
"""
Run All BayeSED3 Python Interface Examples

This script runs all test examples sequentially and provides a summary
of successes and failures.
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path to ensure bayesed can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_example(name, test_function):
    """Run a single test example and catch any errors."""
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"{'='*80}\n")
    
    try:
        test_function()
        print(f"\n✓ {name} completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ {name} failed with error:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all examples and provide summary."""
    
    results = {}
    
    # Import and run each test
    print("Starting BayeSED3 Python Interface Examples")
    print("=" * 80)
    
    # 1. Quick Start
    try:
        from quick_start import test_quick_start
        results['Quick Start'] = run_example('Quick Start', test_quick_start)
    except ImportError as e:
        print(f"✗ Could not import quick_start: {e}")
        results['Quick Start'] = False
    
    # 2. AGN Fitting
    try:
        from test_agn_fitting import test_agn_fitting
        results['AGN Fitting'] = run_example('AGN Fitting', test_agn_fitting)
    except ImportError as e:
        print(f"✗ Could not import test_agn_fitting: {e}")
        results['AGN Fitting'] = False
    
    # 3. Data Arrays
    try:
        from test_data_arrays import test_data_arrays
        results['Data Arrays'] = run_example('Data Arrays', test_data_arrays)
    except ImportError as e:
        print(f"✗ Could not import test_data_arrays: {e}")
        results['Data Arrays'] = False
    
    # 4. Custom Model
    try:
        from test_custom_model import test_custom_model_dust_emission
        results['Custom Model'] = run_example('Custom Model', test_custom_model_dust_emission)
    except ImportError as e:
        print(f"✗ Could not import test_custom_model: {e}")
        results['Custom Model'] = False
    
    # 5. Multi-Model Comparison
    try:
        from test_multi_model_comparison import test_multi_model_comparison
        results['Multi-Model Comparison'] = run_example('Multi-Model Comparison', test_multi_model_comparison)
    except ImportError as e:
        print(f"✗ Could not import test_multi_model_comparison: {e}")
        results['Multi-Model Comparison'] = False
    
    # 6. Advanced Analytics
    try:
        from test_advanced_analytics import test_advanced_analytics
        results['Advanced Analytics'] = run_example('Advanced Analytics', test_advanced_analytics)
    except ImportError as e:
        print(f"✗ Could not import test_advanced_analytics: {e}")
        results['Advanced Analytics'] = False
    
    # 7. Prior Management
    try:
        from test_prior_management import test_prior_management
        results['Prior Management'] = run_example('Prior Management', test_prior_management)
    except ImportError as e:
        print(f"✗ Could not import test_prior_management: {e}")
        results['Prior Management'] = False
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:12} {name}")
    
    print(f"\n{'-'*80}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"{'-'*80}\n")
    
    if failed > 0:
        print("Some examples failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("All examples completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
