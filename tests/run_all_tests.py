"""
Siren AI v2 - Master Test Runner
═════════════════════════════════════════════
Comprehensive test execution and reporting suite

Runs all test suites and generates detailed reports

Usage:
    python run_all_tests.py
    python run_all_tests.py --verbose
    python run_all_tests.py --report
"""

import unittest
import sys
import os
import time
import json
from io import StringIO
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test modules
try:
    from test_sarsa_engine import (
        TestSARSAAgentInitialization,
        TestEpsilonDecay,
        TestQValueManagement,
        TestActionSelection,
        TestRewardFunction,
        TestStateManagement
    )
    SARSA_TESTS_AVAILABLE = True
except ImportError:
    SARSA_TESTS_AVAILABLE = False
    print("⚠ Warning: SARSA engine tests not found")

try:
    from test_safety_wrapper import (
        TestAggressionOverride,
        TestCooldownEnforcement,
        TestBudgetEnforcement,
        TestSafetyScenarios,
        TestEdgeCases
    )
    SAFETY_TESTS_AVAILABLE = True
except ImportError:
    SAFETY_TESTS_AVAILABLE = False
    print("⚠ Warning: Safety wrapper tests not found")


class TestReportGenerator:
    """Generate comprehensive test reports"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def generate_summary_report(self, result):
        """Generate summary report"""
        total_tests = result.testsRun
        passed = total_tests - len(result.failures) - len(result.errors)
        failed = len(result.failures)
        errors = len(result.errors)
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        report = f"""
{'='*70}
  SIREN AI v2 - COMPREHENSIVE TEST REPORT
{'='*70}
  
  Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Duration: {self.end_time - self.start_time:.2f} seconds
  
  TEST STATISTICS:
  ────────────────────────────────────────────────────────────────────
  Total Tests Run:        {total_tests}
  Tests Passed:           {passed} ({success_rate:.1f}%)
  Tests Failed:           {failed}
  Tests with Errors:      {errors}
  
  SUCCESS RATE:           {success_rate:.1f}%
  
  TEST CATEGORIES:
  ────────────────────────────────────────────────────────────────────
"""
        
        # Add category breakdown
        categories = {
            'SARSA Engine': SARSA_TESTS_AVAILABLE,
            'Safety Wrapper': SAFETY_TESTS_AVAILABLE,
        }
        
        for category, available in categories.items():
            status = "✓ TESTED" if available else "✗ NOT AVAILABLE"
            report += f"  {category:30s} {status}\n"
        
        report += f"\n{'='*70}\n"
        
        return report
    
    def generate_failure_report(self, result):
        """Generate detailed failure report"""
        if not result.failures and not result.errors:
            return "\n  ✓ NO FAILURES - ALL TESTS PASSED\n"
        
        report = "\n  FAILURE DETAILS:\n"
        report += "  " + "─"*66 + "\n"
        
        # Report failures
        for test, traceback in result.failures:
            report += f"\n  ✗ FAILED: {test}\n"
            report += f"  {traceback}\n"
        
        # Report errors
        for test, traceback in result.errors:
            report += f"\n  ✗ ERROR: {test}\n"
            report += f"  {traceback}\n"
        
        return report
    
    def save_json_report(self, result, filename="test_report.json"):
        """Save test results to JSON file"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': self.end_time - self.start_time,
            'total_tests': result.testsRun,
            'passed': result.testsRun - len(result.failures) - len(result.errors),
            'failed': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0,
            'failures': [str(test) for test, _ in result.failures],
            'errors': [str(test) for test, _ in result.errors],
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n  📄 JSON report saved: {filename}")


def print_banner():
    """Print test suite banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║              SIREN AI v2 - COMPREHENSIVE TEST SUITE              ║
║                                                                   ║
║        Bio-Inspired Acoustic Deterrent System Testing            ║
║                  WildWatch 360 Project                           ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_all_tests(verbose=False, generate_report=True):
    """
    Run all available test suites
    
    Args:
        verbose: If True, print detailed test output
        generate_report: If True, generate comprehensive reports
    
    Returns:
        tuple: (result, report_generator)
    """
    print_banner()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add SARSA tests if available
    if SARSA_TESTS_AVAILABLE:
        print("  ✓ Loading SARSA Engine Tests...")
        suite.addTests(loader.loadTestsFromTestCase(TestSARSAAgentInitialization))
        suite.addTests(loader.loadTestsFromTestCase(TestEpsilonDecay))
        suite.addTests(loader.loadTestsFromTestCase(TestQValueManagement))
        suite.addTests(loader.loadTestsFromTestCase(TestActionSelection))
        suite.addTests(loader.loadTestsFromTestCase(TestRewardFunction))
        suite.addTests(loader.loadTestsFromTestCase(TestStateManagement))
    
    # Add Safety tests if available
    if SAFETY_TESTS_AVAILABLE:
        print("  ✓ Loading Safety Wrapper Tests...")
        suite.addTests(loader.loadTestsFromTestCase(TestAggressionOverride))
        suite.addTests(loader.loadTestsFromTestCase(TestCooldownEnforcement))
        suite.addTests(loader.loadTestsFromTestCase(TestBudgetEnforcement))
        suite.addTests(loader.loadTestsFromTestCase(TestSafetyScenarios))
        suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    print(f"\n  Total test cases loaded: {suite.countTestCases()}")
    print("\n  Starting test execution...\n")
    
    # Create report generator
    report_gen = TestReportGenerator()
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    report_gen.start_time = time.time()
    result = runner.run(suite)
    report_gen.end_time = time.time()
    
    # Generate reports
    if generate_report:
        print(report_gen.generate_summary_report(result))
        print(report_gen.generate_failure_report(result))
        
        # Save JSON report
        report_gen.save_json_report(result)
    
    # Print final verdict
    print_final_verdict(result)
    
    return result, report_gen


def print_final_verdict(result):
    """Print final test verdict"""
    print("\n" + "="*70)
    
    if result.wasSuccessful():
        print("""
  ╔═══════════════════════════════════════════════════════════════╗
  ║                                                               ║
  ║              ✓ ALL TESTS PASSED SUCCESSFULLY                 ║
  ║                                                               ║
  ║              SYSTEM READY FOR DEPLOYMENT                     ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
  ╔═══════════════════════════════════════════════════════════════╗
  ║                                                               ║
  ║              ✗ SOME TESTS FAILED                             ║
  ║                                                               ║
  ║              REVIEW REQUIRED BEFORE DEPLOYMENT               ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝
""")
        print("  Please review failure details above and fix issues.\n")
    
    print("="*70 + "\n")


def main():
    """Main entry point"""
    # Parse command line arguments
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    generate_report = '--report' in sys.argv or '--no-report' not in sys.argv
    
    # Run tests
    result, report_gen = run_all_tests(
        verbose=verbose,
        generate_report=generate_report
    )
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
