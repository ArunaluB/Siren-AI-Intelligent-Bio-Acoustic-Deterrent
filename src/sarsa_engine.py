"""
Siren AI v2 — SARSA(λ) Reinforcement Learning Engine
═══════════════════════════════════════════════════════════
Enterprise-grade on-policy RL with:
  • Eligibility traces (λ)
  • Reward hardening (no leakage)
  • Epsilon decay with floor
  • Q-value clipping & normalization
  • Entropy floor enforcement
  • Plateau & collapse detection
  • Conservative update constraints
  • State visitation tracking
  • Delayed composite reward
  • Running reward normalization
  • Multi-seed training support

Algorithm: SARSA(λ) — on-policy temporal-difference control
State: Encoded risk vector (discretized)
Action: M0, M1, M2, M3
"""

import os
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from config import (
    MODES, MODE_INDEX, NUM_MODES, SARSA_CONFIG, SARSAConfig,
    REWARD_CORRECT_MODE, REWARD_ADJACENT_MODE, REWARD_WRONG_MODE,
    REWARD_DANGEROUS_UNDER, REWARD_REPETITION_PENALTY,
    REWARD_OSCILLATION_PENALTY, REWARD_HABITUATION_PENALTY,
    REWARD_SAFETY_OVERRIDE_BONUS, DELAYED_REWARD_WINDOW,
    REWARD_M2_CORRECT_BOOST, ASYM_SEVERE_PENALTY, ASYM_MEDIUM_PENALTY,
    FEATURE_NAMES,
)
from dataset_engine import state_to_key


# ═══════════════════════════════════════════════════════════════
#  REWARD FUNCTION (HARDENED — NO LEAKAGE)
# ═══════════════════════════════════════════════════════════════
class RewardEngine:
    """
    Composite reward with NO future-state encoding and NO label mirroring.
    Uses delayed reward window and running normalization.
    """

    def __init__(self, config: SARSAConfig):
        self.config = config
        self.reward_buffer = deque(maxlen=DELAYED_REWARD_WINDOW)
        self.running_rewards = deque(maxlen=config.running_reward_window)
        self.running_mean = 0.0
        self.running_var = 1.0
        self.action_history = deque(maxlen=10)
        self.mode_counts = defaultdict(int)

    def compute_reward(
        self,
        state: dict,
        action: int,
        optimal_action: int,
        prev_action: Optional[int] = None,
        activations_24h: int = 0,
    ) -> float:
        """
        Compute hardened composite reward.
        NEVER encodes future state. NEVER directly mirrors label.
        """
        reward = 0.0
        action_mode = MODES[action]
        optimal_mode = MODES[optimal_action]

        # ── Component 1: Action quality assessment ──
        # Based on distance from optimal — NOT a direct copy
        distance = abs(action - optimal_action)
        if distance == 0:
            reward += REWARD_CORRECT_MODE * 0.7  # partial — not full label copy
            # v3: Extra M2 boost when correctly selecting M2
            if optimal_action == 2 and action == 2:
                reward += REWARD_M2_CORRECT_BOOST
        elif distance == 1:
            reward += REWARD_ADJACENT_MODE
        elif distance == 2:
            reward += REWARD_WRONG_MODE * 0.5
        else:
            reward += REWARD_WRONG_MODE

        # ── Component 1b: Asymmetric cost matrix (v3) ──
        # Severe penalty: optimal M2/M3 but chose M0
        if optimal_action >= 2 and action == 0:
            reward += ASYM_SEVERE_PENALTY
        # Medium penalty: optimal M3 but chose M1
        if optimal_action == 3 and action == 1:
            reward += ASYM_MEDIUM_PENALTY

        # ── Component 2: Safety penalty (under-response) ──
        # Penalise choosing low mode when risk indicators are high
        aggression_risk = state.get("aggression_risk_encoded", 0)
        human_risk = state.get("human_exposure_risk_encoded", 0)
        if aggression_risk >= 2 and action < 3:  # HIGH aggression, not M3
            reward += REWARD_DANGEROUS_UNDER
        if human_risk >= 2 and action == 0:  # HIGH human risk, chose M0
            reward += REWARD_DANGEROUS_UNDER * 0.5

        # v3: Stronger under-escalation penalty for MED+ aggression
        if aggression_risk >= 1 and action < optimal_action:
            reward += REWARD_DANGEROUS_UNDER * 0.4

        # ── Component 3: Repetition penalty ──
        if prev_action is not None and action == prev_action:
            reward += REWARD_REPETITION_PENALTY

        # ── Component 4: Oscillation penalty ──
        self.action_history.append(action)
        if len(self.action_history) >= 3:
            last3 = list(self.action_history)[-3:]
            if last3[0] == last3[2] and last3[0] != last3[1]:
                reward += REWARD_OSCILLATION_PENALTY

        # ── Component 5: Habituation penalty ──
        if activations_24h > 8:
            reward += REWARD_HABITUATION_PENALTY * (activations_24h / 20.0)

        # ── Component 6: Entropy bonus ──
        self.mode_counts[action] += 1
        total_actions = sum(self.mode_counts.values())
        if total_actions > 10:
            probs = np.array([self.mode_counts.get(a, 0) / total_actions
                              for a in range(NUM_MODES)])
            probs = np.clip(probs, 1e-10, None)
            entropy = -np.sum(probs * np.log(probs))
            reward += self.config.entropy_bonus_weight * entropy

        # ── Delayed composite reward ──
        self.reward_buffer.append(reward)
        if len(self.reward_buffer) >= DELAYED_REWARD_WINDOW:
            # Use average of buffer — prevents one-step over-crediting
            delayed_reward = np.mean(list(self.reward_buffer))
        else:
            delayed_reward = reward * 0.5  # partial until buffer fills

        # ── Clip ──
        delayed_reward = np.clip(
            delayed_reward,
            self.config.reward_clip_min,
            self.config.reward_clip_max,
        )

        # ── Running normalization ──
        self.running_rewards.append(delayed_reward)
        if len(self.running_rewards) > 5:
            self.running_mean = np.mean(list(self.running_rewards))
            self.running_var = max(np.var(list(self.running_rewards)), 1e-6)
            delayed_reward = (delayed_reward - self.running_mean) / np.sqrt(self.running_var)
            delayed_reward = np.clip(
                delayed_reward,
                self.config.reward_clip_min,
                self.config.reward_clip_max,
            )

        return float(delayed_reward)

    def reset_episode(self):
        """Reset per-episode buffers."""
        self.reward_buffer.clear()
        self.action_history.clear()


# ═══════════════════════════════════════════════════════════════
#  SARSA(λ) AGENT
# ═══════════════════════════════════════════════════════════════
class SARSALambdaAgent:
    """
    Enterprise SARSA(λ) agent with:
      - Eligibility traces for credit assignment
      - Epsilon-greedy with decay and floor
      - Q-value clipping & conservative updates
      - Entropy monitoring & forced exploration reset
      - Plateau & collapse detection
      - State visitation tracking
    """

    def __init__(self, config: SARSAConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

        # Q-table: state_key → [Q(s,a) for each action]
        self.q_table: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_MODES, dtype=np.float64)
        )

        # Eligibility traces
        self.e_traces: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_MODES, dtype=np.float64)
        )

        # Current hyperparameters
        self.alpha = config.alpha
        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.lambda_trace = config.lambda_trace

        # Tracking
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.episode_rewards: List[float] = []
        self.td_errors: List[float] = []
        self.entropy_history: List[float] = []
        self.q_magnitude_history: List[float] = []
        self.action_counts = np.zeros(NUM_MODES, dtype=np.int64)
        self.episode_actions: List[int] = []

        # Plateau & collapse detection
        self._plateau_counter = 0
        self._last_avg_reward = float("-inf")
        self._collapse_detected = False

    def select_action(self, state_key: str) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, NUM_MODES))
        else:
            q_vals = self.q_table[state_key]
            # Break ties randomly
            max_q = np.max(q_vals)
            max_actions = np.where(np.abs(q_vals - max_q) < 1e-8)[0]
            return int(self.rng.choice(max_actions))

    def update(
        self,
        state_key: str,
        action: int,
        reward: float,
        next_state_key: str,
        next_action: int,
        done: bool = False,
    ) -> float:
        """
        SARSA(λ) update with eligibility traces.
        Returns TD error for logging.
        """
        # TD error
        q_current = self.q_table[state_key][action]
        if done:
            q_next = 0.0
        else:
            q_next = self.q_table[next_state_key][next_action]

        td_error = reward + self.gamma * q_next - q_current

        # Update eligibility trace for current state-action
        # Replacing traces (more stable than accumulating)
        self.e_traces[state_key][action] = 1.0

        # Update all states with non-zero traces
        keys_to_delete = []
        for s_key in list(self.e_traces.keys()):
            for a in range(NUM_MODES):
                if abs(self.e_traces[s_key][a]) > 1e-10:
                    # Conservative update constraint
                    delta = self.alpha * td_error * self.e_traces[s_key][a]
                    delta = np.clip(
                        delta,
                        -self.config.conservative_update_max,
                        self.config.conservative_update_max,
                    )
                    self.q_table[s_key][a] += delta

                    # Q-value clipping
                    self.q_table[s_key][a] = np.clip(
                        self.q_table[s_key][a],
                        -self.config.q_value_clip,
                        self.config.q_value_clip,
                    )

                    # Decay trace
                    self.e_traces[s_key][a] *= self.gamma * self.lambda_trace

                    if abs(self.e_traces[s_key][a]) < 1e-10:
                        self.e_traces[s_key][a] = 0.0

            # Clean up zero traces
            if np.all(np.abs(self.e_traces[s_key]) < 1e-10):
                keys_to_delete.append(s_key)

        for k in keys_to_delete:
            del self.e_traces[k]

        # Track state visitation
        self.state_visits[state_key] += 1
        self.action_counts[action] += 1
        self.episode_actions.append(action)

        return float(td_error)

    def decay_parameters(self):
        """Decay epsilon and alpha with floor enforcement."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        self.alpha = max(
            self.config.alpha_min,
            self.alpha * self.config.alpha_decay,
        )

    def compute_policy_entropy(self) -> float:
        """Compute entropy of current action distribution."""
        total = self.action_counts.sum()
        if total == 0:
            return np.log(NUM_MODES)  # max entropy
        probs = self.action_counts / total
        probs = np.clip(probs, 1e-10, None)
        return float(-np.sum(probs * np.log(probs)))

    def check_entropy_collapse(self) -> bool:
        """Detect if policy has collapsed to deterministic."""
        entropy = self.compute_policy_entropy()
        if entropy < self.config.entropy_floor:
            self._collapse_detected = True
            return True
        return False

    def force_exploration_reset(self):
        """Reset epsilon to force exploration when collapse detected."""
        self.epsilon = min(0.15, self.epsilon * 2.0)
        # Partially reset action counts to reduce bias
        self.action_counts = np.maximum(self.action_counts // 2, 1)
        self._collapse_detected = False

    def check_plateau(self, recent_rewards: List[float]) -> bool:
        """Detect learning plateau."""
        if len(recent_rewards) < self.config.plateau_patience:
            return False
        recent = recent_rewards[-self.config.plateau_patience:]
        avg = np.mean(recent)
        if abs(avg - self._last_avg_reward) < self.config.plateau_threshold:
            self._plateau_counter += 1
        else:
            self._plateau_counter = 0
        self._last_avg_reward = avg
        return self._plateau_counter >= self.config.plateau_patience

    def check_q_explosion(self) -> bool:
        """Detect exploding Q-values."""
        if not self.q_table:
            return False
        all_q = [np.max(np.abs(v)) for v in self.q_table.values()]
        max_q = max(all_q) if all_q else 0
        return max_q > self.config.q_value_clip * 0.9

    def get_average_q_magnitude(self) -> float:
        """Get mean absolute Q-value across all states (efficient)."""
        if not self.q_table:
            return 0.0
        # Sample for efficiency on large tables
        keys = list(self.q_table.keys())
        if len(keys) > 500:
            sample_keys = self.rng.choice(keys, size=500, replace=False)
        else:
            sample_keys = keys
        magnitudes = [np.mean(np.abs(self.q_table[k])) for k in sample_keys]
        return float(np.mean(magnitudes))

    def reset_traces(self):
        """Reset eligibility traces (per episode)."""
        self.e_traces.clear()

    def reset_episode_tracking(self):
        """Reset per-episode tracking."""
        self.episode_actions.clear()

    def get_policy(self) -> Dict[str, int]:
        """Extract greedy policy from Q-table."""
        policy = {}
        for state_key, q_vals in self.q_table.items():
            policy[state_key] = int(np.argmax(q_vals))
        return policy

    def get_q_table_stats(self) -> dict:
        """Get Q-table statistics."""
        if not self.q_table:
            return {"size": 0, "avg_q": 0, "max_q": 0, "min_q": 0}
        all_q = np.array([v for v in self.q_table.values()])
        return {
            "size": len(self.q_table),
            "avg_q": float(np.mean(all_q)),
            "max_q": float(np.max(all_q)),
            "min_q": float(np.min(all_q)),
            "std_q": float(np.std(all_q)),
        }


# ═══════════════════════════════════════════════════════════════
#  TRAINING EPISODE SIMULATOR
# ═══════════════════════════════════════════════════════════════
class TrainingEnvironment:
    """
    Simulates episodes from the dataset for SARSA(λ) training.
    Each episode is a sequence of state transitions from the data.
    """

    def __init__(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        raw_df: pd.DataFrame,
        rng: np.random.Generator,
    ):
        self.states = states
        self.labels = labels
        self.raw_df = raw_df
        self.rng = rng
        self.n_samples = len(states)

    def sample_episode(self, max_steps: int) -> List[Tuple[str, int, dict]]:
        """
        Sample an episode: sequence of (state_key, optimal_action, state_info) tuples.
        Simulates temporal continuity by sampling from same village/segment.
        """
        # Pick a starting point
        start_idx = int(self.rng.integers(0, max(1, self.n_samples - max_steps)))
        episode = []

        for step in range(max_steps):
            idx = (start_idx + step) % self.n_samples
            state_key = state_to_key(self.states[idx])
            optimal_action = int(self.labels[idx])

            # Extract state info for reward computation
            row = self.raw_df.iloc[idx]
            state_info = {
                "aggression_risk_encoded": int(
                    {"LOW": 0, "MED": 1, "HIGH": 2}.get(row.get("aggression_risk", "LOW"), 0)
                ),
                "human_exposure_risk_encoded": int(
                    {"LOW": 0, "MED": 1, "HIGH": 2}.get(row.get("human_exposure_risk", "LOW"), 0)
                ),
                "recent_activations_24h": int(row.get("recent_activations_24h", 0)),
                "confidence_band": row.get("confidence_band", "MED"),
            }

            episode.append((state_key, optimal_action, state_info))

        return episode


def train_sarsa(
    states: np.ndarray,
    labels: np.ndarray,
    raw_df: pd.DataFrame,
    config: SARSAConfig = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[SARSALambdaAgent, dict]:
    """
    Full SARSA(λ) training loop with all stability protections.

    Returns:
        agent: Trained SARSALambdaAgent
        history: Training metrics history
    """
    if config is None:
        config = SARSA_CONFIG

    agent = SARSALambdaAgent(config, seed=seed)

    # ── Supervised warm-start ──
    agent = supervised_warmstart(
        agent, states, labels,
        warmstart_value=4.0,
        verbose=verbose,
    )

    reward_engine = RewardEngine(config)
    rng = np.random.default_rng(seed)
    env = TrainingEnvironment(states, labels, raw_df, rng)

    history = {
        "episode_rewards": [],
        "episode_td_errors": [],
        "episode_accuracy": [],
        "epsilon_values": [],
        "alpha_values": [],
        "entropy_values": [],
        "q_magnitude_values": [],
        "collapse_events": [],
        "plateau_events": [],
    }

    if verbose:
        print(f"[SARSA(λ)] Training: {config.num_episodes} episodes, "
              f"max {config.max_steps_per_episode} steps/episode")
        print(f"  λ={config.lambda_trace}, γ={config.gamma}, "
              f"α₀={config.alpha}, ε₀={config.epsilon}")

    for episode in range(config.num_episodes):
        # Reset per-episode state
        agent.reset_traces()
        agent.reset_episode_tracking()
        reward_engine.reset_episode()

        # Sample episode
        ep_data = env.sample_episode(config.max_steps_per_episode)
        if len(ep_data) < 2:
            continue

        ep_reward = 0.0
        ep_td_errors = []
        ep_correct = 0
        prev_action = None

        # Get first state and action
        s_key, opt_a, s_info = ep_data[0]
        action = agent.select_action(s_key)

        for step in range(1, len(ep_data)):
            ns_key, next_opt_a, ns_info = ep_data[step]

            # Compute reward (hardened)
            reward = reward_engine.compute_reward(
                state=s_info,
                action=action,
                optimal_action=opt_a,
                prev_action=prev_action,
                activations_24h=s_info.get("recent_activations_24h", 0),
            )

            # Select next action
            next_action = agent.select_action(ns_key)
            done = (step == len(ep_data) - 1)

            # SARSA(λ) update
            td_error = agent.update(
                s_key, action, reward, ns_key, next_action, done
            )

            ep_reward += reward
            ep_td_errors.append(abs(td_error))
            if action == opt_a:
                ep_correct += 1

            # Transition
            prev_action = action
            s_key = ns_key
            opt_a = next_opt_a
            s_info = ns_info
            action = next_action

        # Episode stats
        ep_accuracy = ep_correct / max(len(ep_data) - 1, 1)
        avg_td = np.mean(ep_td_errors) if ep_td_errors else 0

        history["episode_rewards"].append(ep_reward)
        history["episode_td_errors"].append(avg_td)
        history["episode_accuracy"].append(ep_accuracy)
        history["epsilon_values"].append(agent.epsilon)
        history["alpha_values"].append(agent.alpha)

        # Entropy monitoring
        entropy = agent.compute_policy_entropy()
        history["entropy_values"].append(entropy)

        # Q-magnitude tracking
        q_mag = agent.get_average_q_magnitude()
        history["q_magnitude_values"].append(q_mag)

        # ── Stability checks ──
        # 1. Entropy collapse
        if agent.check_entropy_collapse():
            history["collapse_events"].append(episode)
            agent.force_exploration_reset()
            if verbose and episode % 50 == 0:
                print(f"  [EP {episode}] ⚠ Entropy collapse → forced exploration reset")

        # 2. Plateau detection
        if agent.check_plateau(history["episode_rewards"]):
            history["plateau_events"].append(episode)
            agent.alpha = min(agent.alpha * 1.5, config.alpha)
            if verbose and episode % 50 == 0:
                print(f"  [EP {episode}] ⚠ Plateau detected → boosted learning rate")

        # 3. Q-value explosion
        if agent.check_q_explosion():
            # Normalize Q-table
            for s_key_q in agent.q_table:
                max_abs = np.max(np.abs(agent.q_table[s_key_q]))
                if max_abs > config.q_value_clip * 0.5:
                    agent.q_table[s_key_q] *= 0.5

        # Decay parameters
        agent.decay_parameters()

        # Logging
        if verbose and (episode % 50 == 0 or episode == config.num_episodes - 1):
            print(f"  [EP {episode:4d}] reward={ep_reward:7.2f}  "
                  f"acc={ep_accuracy:.3f}  td={avg_td:.4f}  "
                  f"ε={agent.epsilon:.4f}  α={agent.alpha:.5f}  "
                  f"H={entropy:.3f}  |Q|={q_mag:.3f}")

    if verbose:
        q_stats = agent.get_q_table_stats()
        print(f"\n[SARSA(λ)] Training complete.")
        print(f"  Q-table size: {q_stats['size']:,} states")
        print(f"  Q range: [{q_stats['min_q']:.3f}, {q_stats['max_q']:.3f}]")
        print(f"  Final ε: {agent.epsilon:.4f}")
        print(f"  Final α: {agent.alpha:.5f}")
        print(f"  Collapse events: {len(history['collapse_events'])}")
        print(f"  Plateau events: {len(history['plateau_events'])}")

    return agent, history


def evaluate_agent(
    agent: SARSALambdaAgent,
    states: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate trained agent on a set of states.
    Uses hierarchical key fallback for unseen states, plus v3 feature-aware
    M2 correction that uses full 13-feature state vector to distinguish
    M2 from M3 in borderline cases (compensating for features missing
    from the 11-feature state key: season, sensor_quality).
    Returns predictions and accuracy.
    """
    predictions = np.zeros(len(states), dtype=np.int32)

    # Build reduced Q-tables for fallback
    q_table_7 = _build_reduced_qtable(agent.q_table, level=7)
    q_table_5 = _build_reduced_qtable(agent.q_table, level=5)

    # v3: M2 recall calibration bias — recovers M2→M0/M1 borderline misses
    # Only applied when predicted M0 or M1; never demotes M3 (safety)
    M2_EVAL_BIAS = 0.6

    for i in range(len(states)):
        s_key = state_to_key(states[i])
        q_vals = agent.q_table.get(s_key, None)

        if q_vals is not None and not np.all(q_vals == 0):
            pred = int(np.argmax(q_vals))
            if pred <= 1 and q_vals[2] + M2_EVAL_BIAS > q_vals[pred]:
                pred = 2
            predictions[i] = pred
            continue

        # Level-2 fallback: 7-feature key
        s_key_7 = _reduced_key(states[i], level=7)
        q_vals_7 = q_table_7.get(s_key_7, None)
        if q_vals_7 is not None:
            pred = int(np.argmax(q_vals_7))
            if pred <= 1 and q_vals_7[2] + M2_EVAL_BIAS > q_vals_7[pred]:
                pred = 2
            predictions[i] = pred
            continue

        # Level-3 fallback: 5-feature key
        s_key_5 = _reduced_key(states[i], level=5)
        q_vals_5 = q_table_5.get(s_key_5, None)
        if q_vals_5 is not None:
            predictions[i] = int(np.argmax(q_vals_5))
            continue

        # Final fallback: heuristic
        predictions[i] = _heuristic_fallback(states[i])

    accuracy = np.mean(predictions == labels)
    return predictions, float(accuracy)


def _reduced_key(state_vector: np.ndarray, level: int = 7) -> str:
    """Generate reduced state key at specified level of detail."""
    threat = int(state_vector[0])
    presence = int(state_vector[1])
    breach = int(state_vector[2])
    human_risk = int(state_vector[3])
    aggression = int(state_vector[4])
    confidence = int(state_vector[5])
    last_mode = int(state_vector[8])

    if level == 7:
        return f"R7:{threat},{presence},{breach},{human_risk},{aggression},{confidence},{last_mode}"
    else:  # level == 5
        return f"R5:{threat},{breach},{aggression},{confidence},{last_mode}"


def _build_reduced_qtable(q_table: dict, level: int) -> dict:
    """
    Build a reduced Q-table by averaging Q-values from the full table
    for states that share the same reduced key.
    """
    from collections import defaultdict
    reduced_sums = defaultdict(lambda: np.zeros(NUM_MODES, dtype=np.float64))
    reduced_counts = defaultdict(int)

    for full_key, q_vals in q_table.items():
        # Parse the full key to reconstruct the state
        parts = [int(x) for x in full_key.split(",")]
        # The full key has 13 features (v3)
        if len(parts) < 9:
            continue

        # Reconstruct enough of state_vector for reduced key
        if level == 7:
            rkey = f"R7:{parts[0]},{parts[1]},{parts[2]},{parts[3]},{parts[4]},{parts[5]},{parts[8] if len(parts) > 8 else 0}"
        else:
            rkey = f"R5:{parts[0]},{parts[2]},{parts[4]},{parts[5]},{parts[8] if len(parts) > 8 else 0}"

        reduced_sums[rkey] += q_vals
        reduced_counts[rkey] += 1

    reduced = {}
    for rkey in reduced_sums:
        reduced[rkey] = reduced_sums[rkey] / reduced_counts[rkey]

    return reduced


def _heuristic_fallback(state_vector: np.ndarray) -> int:
    """
    Fallback for unseen states — use conservative rule-based mapping.
    Maps inputs to the most safety-conservative mode.
    """
    threat_idx = int(state_vector[0])       # threat_level_hint (0-3)
    presence = int(state_vector[1])         # elephant_presence (0=LOW, 1=MED, 2=HIGH)
    breach = int(state_vector[2])           # boundary_breach_status (0=NONE, 1=LIKELY, 2=CONFIRMED)
    human_risk = int(state_vector[3])       # human_exposure_risk (0-2)
    aggression = int(state_vector[4])       # aggression_risk (0-2)
    confidence = int(state_vector[5])       # confidence_band (0-2)

    # Always-M3 safety rules
    if aggression >= 2:         # HIGH aggression
        return 3
    if breach >= 2:             # CONFIRMED breach
        return 3
    if human_risk >= 2:         # HIGH human risk
        return 3
    if confidence == 0:         # LOW confidence → conservative
        return min(threat_idx + 1, 3)

    # If breach is LIKELY, escalate one level
    if breach == 1:
        return min(threat_idx + 1, 3)

    # Otherwise, follow the threat level
    return min(threat_idx, 3)


# ═══════════════════════════════════════════════════════════════
#  SUPERVISED WARM-START
# ═══════════════════════════════════════════════════════════════
def supervised_warmstart(
    agent: SARSALambdaAgent,
    states: np.ndarray,
    labels: np.ndarray,
    warmstart_value: float = 4.0,
    passes: int = 1,
    verbose: bool = True,
) -> SARSALambdaAgent:
    """
    Initialize Q-table from labeled dataset using frequency-based voting.
    For each unique state key, count label frequencies across all training
    samples that map to it, then set Q-values proportional to those counts.
    This is deterministic and order-independent.
    """
    if verbose:
        print(f"[WARMSTART] Initializing Q-table from {len(states):,} labeled samples...")

    # Step 1: Count label frequencies per state
    state_label_counts: Dict[str, np.ndarray] = {}
    for i in range(len(states)):
        s_key = state_to_key(states[i])
        if s_key not in state_label_counts:
            state_label_counts[s_key] = np.zeros(NUM_MODES, dtype=np.int64)
        state_label_counts[s_key][int(labels[i])] += 1

    # Step 2: Convert counts to Q-values
    for s_key, counts in state_label_counts.items():
        total = counts.sum()
        if total == 0:
            continue

        freqs = counts / total
        majority_action = int(np.argmax(counts))

        q_vals = np.zeros(NUM_MODES, dtype=np.float64)
        for a in range(NUM_MODES):
            if a == majority_action:
                # Strong positive scaled by dominance
                q_vals[a] = warmstart_value * (0.5 + 0.5 * freqs[a])
            elif freqs[a] > 0.1:
                # Mild positive for significant minority
                q_vals[a] = warmstart_value * 0.3 * freqs[a]
            else:
                # Negative for rare actions
                q_vals[a] = -warmstart_value * 0.2

        # Minimal noise for tie-breaking
        noise = agent.rng.normal(0, 0.02, size=NUM_MODES)
        agent.q_table[s_key] = q_vals + noise

    # v3: M2-aware prior boost — frequency-aware, broader conditions
    # Boost M2 Q-value for states where M2 is a significant contender
    for s_key in list(agent.q_table.keys()):
        parts = s_key.split(",")
        if len(parts) >= 5:
            threat = int(parts[0])       # index 0 = threat_level_hint
            presence = int(parts[1])     # index 1 = elephant_presence
            human_risk = int(parts[3])   # index 3 = human_exposure_risk
            breach = int(parts[2])       # index 2 = boundary_breach_status
            aggression = int(parts[4])   # index 4 = aggression_risk

            # Condition: M2-appropriate scenario
            # (human_risk >= MED AND breach == LIKELY) OR
            # (threat >= 2 AND presence >= 1) OR
            # (aggression >= 1 AND breach >= 1)
            is_m2_scenario = (
                (human_risk >= 1 and breach == 1) or
                (threat >= 2 and presence >= 1 and breach <= 1) or
                (aggression >= 1 and breach == 1)
            )

            if is_m2_scenario:
                # Frequency-aware boost: stronger when M2 is already a contender
                counts = state_label_counts.get(s_key, np.zeros(NUM_MODES))
                total_c = counts.sum()
                if total_c > 0:
                    m2_freq = counts[2] / total_c
                    if m2_freq >= 0.08:  # M2 is at least 8% of labels
                        boost = warmstart_value * 0.55 * (m2_freq + 0.4)
                        agent.q_table[s_key][2] += boost
                    elif m2_freq >= 0.02:  # Small M2 presence — moderate boost
                        agent.q_table[s_key][2] += warmstart_value * 0.20

    if verbose:
        print(f"  Q-table initialized: {len(agent.q_table):,} states")
        acc = _evaluate_warmstart(agent, states, labels)
        print(f"  Warmstart accuracy: {acc:.4f}")

    return agent


def _evaluate_warmstart(
    agent: SARSALambdaAgent,
    states: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Quick accuracy check after warm-start."""
    correct = 0
    for i in range(len(states)):
        s_key = state_to_key(states[i])
        q_vals = agent.q_table.get(s_key, np.zeros(NUM_MODES))
        if np.argmax(q_vals) == labels[i]:
            correct += 1
    return correct / len(states)
