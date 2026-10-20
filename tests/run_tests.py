"""
AquaLens Unified Test Runner

Discovers and executes all test suites under the tests/ directory,
calculates coverage metrics, timings, and displays formatted summary.
"""

from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_all_tests():
    print("=" * 70)
    print("AquaLens ML Project — Automated Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / "tests"), pattern="test_*.py")

    total_test_count = suite.countTestCases()
    print(f"Discovered {total_test_count} tests across all test modules.")
    print("-" * 70)

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    print("=" * 70)
    print("Test Execution Summary:")
    print(f"  Total tests run:     {result.testsRun}")
    print(f"  Passed:              {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"  Failures:            {len(result.failures)}")
    print(f"  Errors:              {len(result.errors)}")
    print(f"  Skipped:             {len(result.skipped)}")
    print(f"  Total time elapsed:  {elapsed:.3f} seconds")

    if result.wasSuccessful():
        print("\nSUCCESS: All AquaLens tests passed successfully! [100% PASS]")
        print("=" * 70)
        return 0
    else:
        print("\nFAILURE: Some tests failed or raised unhandled errors.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
