"""
Siren AI v2 - SARSA Engine Unit Tests
═════════════════════════════════════════
Complete unit test suite for SARSA(λ) agent functionality

Run: python -m unittest test_sarsa_engine -v
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sarsa_engine import SARSALambdaAgent, RewardEngine
from config import SARSA_CONFIG, MODES, NUM_MODES


class TestSARSAAgentInitialization(unittest.TestCase):
    """Test suite for SARSA agent initialization and basic properties"""
    
    def setUp(self):
        """Setup test environment before each test"""
        self.agent = SARSALambdaAgent(
            num_actions=NUM_MODES,
            config=SARSA_CONFIG,
            seed=42
        )
    
    def test_TC001_agent_initializes_correctly(self):
        """
        TC-001: Verify agent initializes with correct parameters
        Priority: HIGH
        Expected: Agent has correct number of actions and initialized structures
        """
        self.assertEqual(self.agent.num_actions, NUM_MODES)
        self.assertIsNotNone(self.agent.q_table)
        self.assertEqual(len(self.agent.action_counts), NUM_MODES)
        self.assertGreater(self.agent.epsilon, 0)
        print(f"✓ TC-001 PASSED: Agent initialized with {NUM_MODES} actions")
    
    def test_TC002_agent_has_valid_initial_epsilon(self):
        """
        TC-002: Verify initial epsilon is within valid range
        Priority: MEDIUM
        Expected: 0 < epsilon <= 1
        """
        self.assertGreater(self.agent.epsilon, 0)
        self.assertLessEqual(self.agent.epsilon, 1.0)
        print(f"✓ TC-002 PASSED: Initial epsilon = {self.agent.epsilon:.4f}")
    
    def test_TC003_q_table_initializes_empty(self):
        """
        TC-003: Verify Q-table starts empty
        Priority: LOW
        Expected: Q-table is initialized but empty
        """
        self.assertIsNotNone(self.agent.q_table)
        initial_size = len(self.agent.q_table)
        self.assertEqual(initial_size, 0)
        print(f"✓ TC-003 PASSED: Q-table initialized empty")


class TestEpsilonDecay(unittest.TestCase):
    """Test suite for epsilon-greedy exploration decay"""
    
    def setUp(self):
        """Setup agent for epsilon decay tests"""
        self.agent = SARSALambdaAgent(
            num_actions=NUM_MODES,
            config=SARSA_CONFIG,
            seed=42
        )
    
    def test_TC004_epsilon_decays_over_episodes(self):
        """
        TC-004: Verify epsilon decreases over time
        Priority: HIGH
        Expected: Epsilon after 100 episodes < initial epsilon
        """
        initial_epsilon = self.agent.epsilon
        
        for _ in range(100):
            self.agent._update_epsilon()
        
        final_epsilon = self.agent.epsilon
        self.assertLess(final_epsilon, initial_epsilon)
        print(f"✓ TC-004 PASSED: Epsilon decayed from {initial_epsilon:.4f} to {final_epsilon:.4f}")
    
    def test_TC005_epsilon_respects_floor(self):
        """
        TC-005: Verify epsilon never goes below floor
        Priority: CRITICAL
        Expected: Epsilon >= epsilon_floor always
        """
        # Decay epsilon many times
        for _ in range(10000):
            self.agent._update_epsilon()
        
        self.assertGreaterEqual(
            self.agent.epsilon, 
            self.agent.config.epsilon_floor,
            f"Epsilon {self.agent.epsilon} below floor {self.agent.config.epsilon_floor}"
        )
        print(f"✓ TC-005 PASSED: Epsilon floor {self.agent.config.epsilon_floor} respected")


class TestQValueManagement(unittest.TestCase):
    """Test suite for Q-value operations and clipping"""
    
    def setUp(self):
        """Setup agent for Q-value tests"""
        self.agent = SARSALambdaAgent(
            num_actions=NUM_MODES,
            config=SARSA_CONFIG,
            seed=42
        )
    
    def test_TC006_q_value_clipping_upper_bound(self):
        """
        TC-006: Verify Q-values are clipped at upper bound
        Priority: CRITICAL
        Expected: Q-values <= q_value_clip
        """
        state_key = "test_state_upper"
        # Set extreme high values
        self.agent.q_table[state_key] = np.array([100.0, 200.0, 150.0, 175.0])
        
        clipped = self.agent._clip_q_values(self.agent.q_table[state_key])
        
        max_clipped = np.max(clipped)
        self.assertLessEqual(
            max_clipped,
            self.agent.config.q_value_clip,
            f"Q-value {max_clipped} exceeds clip value {self.agent.config.q_value_clip}"
        )
        print(f"✓ TC-006 PASSED: Q-values clipped at upper bound {self.agent.config.q_value_clip}")
    
    def test_TC007_q_value_clipping_lower_bound(self):
        """
        TC-007: Verify Q-values are clipped at lower bound
        Priority: CRITICAL
        Expected: Q-values >= -q_value_clip
        """
        state_key = "test_state_lower"
        # Set extreme low values
        self.agent.q_table[state_key] = np.array([-100.0, -200.0, -150.0, -175.0])
        
        clipped = self.agent._clip_q_values(self.agent.q_table[state_key])
        
        min_clipped = np.min(clipped)
        self.assertGreaterEqual(
            min_clipped,
            -self.agent.config.q_value_clip,
            f"Q-value {min_clipped} below clip value {-self.agent.config.q_value_clip}"
        )
        print(f"✓ TC-007 PASSED: Q-values clipped at lower bound {-self.agent.config.q_value_clip}")
    
    def test_TC008_q_values_within_range(self):
        """
        TC-008: Verify normal Q-values stay unchanged
        Priority: MEDIUM
        Expected: Q-values within bounds are not modified
        """
        state_key = "test_state_normal"
        original_values = np.array([1.0, 2.0, 3.0, 4.0])
        self.agent.q_table[state_key] = original_values.copy()
        
        clipped = self.agent._clip_q_values(self.agent.q_table[state_key])
        
        np.testing.assert_array_almost_equal(
            clipped,
            original_values,
            decimal=6,
            err_msg="Normal Q-values should not be modified"
        )
        print(f"✓ TC-008 PASSED: Normal Q-values unchanged after clipping")


class TestActionSelection(unittest.TestCase):
    """Test suite for action selection mechanisms"""
    
    def setUp(self):
        """Setup agent for action selection tests"""
        self.agent = SARSALambdaAgent(
            num_actions=NUM_MODES,
            config=SARSA_CONFIG,
            seed=42
        )
    
    def test_TC009_exploration_with_high_epsilon(self):
        """
        TC-009: Verify agent explores when epsilon is high
        Priority: HIGH
        Expected: Multiple different actions chosen
        """
        self.agent.epsilon = 0.9  # High exploration
        state_key = "explore_state"
        self.agent.q_table[state_key] = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Select action 100 times
        actions = [self.agent.choose_action(state_key) for _ in range(100)]
        unique_actions = len(set(actions))
        
        self.assertGreater(
            unique_actions, 
            1, 
            "Agent should explore multiple actions with high epsilon"
        )
        print(f"✓ TC-009 PASSED: Agent explored {unique_actions} different actions")
    
    def test_TC010_exploitation_with_zero_epsilon(self):
        """
        TC-010: Verify agent exploits when epsilon is zero
        Priority: HIGH
        Expected: Always chooses action with highest Q-value
        """
        self.agent.epsilon = 0.0  # Pure exploitation
        state_key = "exploit_state"
        self.agent.q_table[state_key] = np.array([1.0, 5.0, 2.0, 3.0])
        
        # Action 1 has highest Q-value (5.0)
        actions = [self.agent.choose_action(state_key) for _ in range(20)]
        
        self.assertTrue(
            all(a == 1 for a in actions),
            "Agent should always choose action with highest Q-value"
        )
        print(f"✓ TC-010 PASSED: Agent consistently exploited best action (action 1)")
    
    def test_TC011_action_counts_increment(self):
        """
        TC-011: Verify action counts are tracked correctly
        Priority: MEDIUM
        Expected: Action counts increase after selection
        """
        state_key = "count_test_state"
        self.agent.q_table[state_key] = np.array([0.0, 0.0, 0.0, 0.0])
        initial_counts = self.agent.action_counts.copy()
        
        # Choose action multiple times
        for _ in range(10):
            self.agent.choose_action(state_key)
        
        total_new_actions = sum(self.agent.action_counts) - sum(initial_counts)
        self.assertEqual(
            total_new_actions,
            10,
            "Total action count should increase by number of selections"
        )
        print(f"✓ TC-011 PASSED: Action counts tracked correctly")


class TestRewardFunction(unittest.TestCase):
    """Test suite for reward calculation"""
    
    def setUp(self):
        """Setup reward engine for testing"""
        self.reward_engine = RewardEngine(SARSA_CONFIG)
    
    def test_TC012_correct_action_positive_reward(self):
        """
        TC-012: Verify correct action receives positive reward
        Priority: CRITICAL
        Expected: Reward > 0 when action matches optimal
        """
        state = {"risk_level": "MED"}
        
        reward = self.reward_engine.compute_reward(
            state=state,
            action=1,  # M1
            optimal_action=1,  # Correct choice
        )
        
        self.assertGreater(
            reward,
            0,
            "Correct action should give positive reward"
        )
        print(f"✓ TC-012 PASSED: Correct action reward = {reward:.4f}")
    
    def test_TC013_wrong_action_negative_reward(self):
        """
        TC-013: Verify wrong action receives negative reward
        Priority: CRITICAL
        Expected: Reward < 0 when action doesn't match optimal
        """
        state = {"risk_level": "HIGH"}
        
        reward = self.reward_engine.compute_reward(
            state=state,
            action=0,  # M0 (too weak)
            optimal_action=3,  # Should be M3
        )
        
        self.assertLess(
            reward,
            0,
            "Wrong action should give negative reward"
        )
        print(f"✓ TC-013 PASSED: Wrong action penalty = {reward:.4f}")
    
    def test_TC014_adjacent_mode_moderate_penalty(self):
        """
        TC-014: Verify adjacent mode gets moderate penalty
        Priority: MEDIUM
        Expected: Adjacent action penalty less severe than distant
        """
        state = {"risk_level": "MED"}
        
        # Adjacent action (distance = 1)
        reward_adjacent = self.reward_engine.compute_reward(
            state=state,
            action=1,  # M1
            optimal_action=2,  # M2 (adjacent)
        )
        
        # Distant action (distance = 2)
        reward_distant = self.reward_engine.compute_reward(
            state=state,
            action=0,  # M0
            optimal_action=2,  # M2 (distant)
        )
        
        self.assertGreater(
            reward_adjacent,
            reward_distant,
            "Adjacent action penalty should be less than distant"
        )
        print(f"✓ TC-014 PASSED: Adjacent penalty ({reward_adjacent:.4f}) < Distant penalty ({reward_distant:.4f})")


class TestStateManagement(unittest.TestCase):
    """Test suite for state tracking and visitation"""
    
    def setUp(self):
        """Setup agent for state management tests"""
        self.agent = SARSALambdaAgent(
            num_actions=NUM_MODES,
            config=SARSA_CONFIG,
            seed=42
        )
    
    def test_TC015_state_visitation_tracking(self):
        """
        TC-015: Verify state visits are tracked
        Priority: MEDIUM
        Expected: State visit count increases
        """
        state_key = "visited_state"
        initial_visits = self.agent.state_visits.get(state_key, 0)
        
        # Visit state multiple times
        for _ in range(5):
            self.agent.choose_action(state_key)
        
        final_visits = self.agent.state_visits.get(state_key, 0)
        
        self.assertGreater(
            final_visits,
            initial_visits,
            "State visit count should increase"
        )
        print(f"✓ TC-015 PASSED: State visited {final_visits} times")


def run_all_tests():
    """
    Run all test suites and generate summary report
    """
    print("\n" + "="*70)
    print("  SIREN AI v2 - SARSA ENGINE TEST SUITE")
    print("  Running comprehensive unit tests...")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSARSAAgentInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestEpsilonDecay))
    suite.addTests(loader.loadTestsFromTestCase(TestQValueManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestActionSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestStateManagement))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("  TEST EXECUTION SUMMARY")
    print("="*70)
    print(f"  Total Tests Run:    {result.testsRun}")
    print(f"  Tests Passed:       {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Tests Failed:       {len(result.failures)}")
    print(f"  Tests Errored:      {len(result.errors)}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"  Success Rate:       {success_rate:.1f}%")
    print("="*70)
    
    if result.wasSuccessful():
        print("\n  ✓ ALL TESTS PASSED - SARSA ENGINE VALIDATED\n")
    else:
        print("\n  ✗ SOME TESTS FAILED - REVIEW REQUIRED\n")
    
    return result


if __name__ == '__main__':
    # Run all tests
    result = run_all_tests()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
