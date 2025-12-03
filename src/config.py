"""
Siren AI v2 — Enterprise Configuration
Wildlife 360 Conservative Deterrent Orchestrator

All system constants, thresholds, and RL hyperparameters.
LoRa-only communication. No MQTT.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(BASE_DIR, "Sound")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models", "trained")
EXPORT_DIR = os.path.join(BASE_DIR, "export")
DATA_DIR = os.path.join(BASE_DIR, "data")

for _d in [RESULTS_DIR, MODELS_DIR, EXPORT_DIR, DATA_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SOUND LIBRARY
# ─────────────────────────────────────────────────────────────
SOUND_CATEGORIES = {
    "BEE": os.path.join(SOUND_DIR, "Bee"),
    "LEOPARD": os.path.join(SOUND_DIR, "Leopard"),
    "SIREN": os.path.join(SOUND_DIR, "Siren"),
}

SOUND_CUE_MAP = {
    "M0": None,                    # Observe — no sound
    "M1": ["BEE"],                 # Minimal deterrent — bee only
    "M2": ["BEE", "LEOPARD"],      # Stronger deterrent — bee + predator
    "M3": ["SIREN"],               # Human escalation — community alarm
}

# ─────────────────────────────────────────────────────────────
# OPERATIONAL MODES
# ─────────────────────────────────────────────────────────────
MODES = ["M0", "M1", "M2", "M3"]
MODE_INDEX = {m: i for i, m in enumerate(MODES)}
NUM_MODES = len(MODES)

MODE_DESCRIPTIONS = {
    "M0": "Observe — no action",
    "M1": "Minimal deterrent — 1 speaker, short burst",
    "M2": "Stronger deterrent — 2 speakers, alternating",
    "M3": "Human escalation — community alert",
}

# ─────────────────────────────────────────────────────────────
# INPUT FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────
THREAT_LEVELS = ["M0", "M1", "M2", "M3"]
PRESENCE_LEVELS = ["LOW", "MED", "HIGH"]
BREACH_STATUS = ["NONE", "LIKELY", "CONFIRMED"]
RISK_LEVELS = ["LOW", "MED", "HIGH"]
CONFIDENCE_BANDS = ["LOW", "MED", "HIGH"]
SEASONS = ["DRY", "WET", "HARVEST"]
TIME_OF_DAY = ["DAY", "NIGHT"]
SENSOR_QUALITY = ["GOOD", "DEGRADED"]

FEATURE_NAMES = [
    "threat_level_hint",
    "elephant_presence",
    "boundary_breach_status",
    "human_exposure_risk",
    "aggression_risk",
    "confidence_band",
    "eta_to_boundary_window_sec",
    "recent_activations_24h",
    "last_mode_used",
    "cooldown_remaining_sec",
    "season",
    "time_of_day",
    "sensor_quality",
]

# Encoding maps for categorical features
ENCODINGS = {
    "threat_level_hint": {v: i for i, v in enumerate(THREAT_LEVELS)},
    "elephant_presence": {v: i for i, v in enumerate(PRESENCE_LEVELS)},
    "boundary_breach_status": {v: i for i, v in enumerate(BREACH_STATUS)},
    "human_exposure_risk": {v: i for i, v in enumerate(RISK_LEVELS)},
    "aggression_risk": {v: i for i, v in enumerate(RISK_LEVELS)},
    "confidence_band": {v: i for i, v in enumerate(CONFIDENCE_BANDS)},
    "last_mode_used": {v: i for i, v in enumerate(MODES)},
    "season": {v: i for i, v in enumerate(SEASONS)},
    "time_of_day": {v: i for i, v in enumerate(TIME_OF_DAY)},
    "sensor_quality": {v: i for i, v in enumerate(SENSOR_QUALITY)},
}

# Number of bins for continuous features → discretisation
ETA_BINS = 6       # 0-30, 30-120, 120-300, 300-600, 600-1800, 1800+
ACTIVATION_BINS = 5 # 0, 1-2, 3-5, 6-10, 11+
COOLDOWN_BINS = 4   # 0, 1-300, 301-900, 901+

# ─────────────────────────────────────────────────────────────
# DATASET ENGINE PARAMETERS
# ─────────────────────────────────────────────────────────────
DATASET_SIZE = 200_000
NUM_VILLAGES = 5
NUM_BOUNDARY_SEGMENTS = 4   # per village

ELEPHANT_BEHAVIOR_TYPES = ["TIMID", "HABITUAL", "AGGRESSIVE", "ADAPTIVE"]
BEHAVIOR_WEIGHTS = [0.30, 0.30, 0.15, 0.25]  # sampling probability

# Season distribution over 1 year (roughly)
SEASON_WEIGHTS = {"DRY": 0.33, "WET": 0.42, "HARVEST": 0.25}
TIME_WEIGHTS = {"DAY": 0.55, "NIGHT": 0.45}

# Rare event thresholds
RARE_AGGRESSION_SPIKE_PROB = 0.03
SENSOR_DEGRADATION_PROB = 0.12

# Imbalance abort thresholds
MAX_CLASS_SKEW = 0.15          # abort if any mode > 40% or < 10%
MAX_STATE_ENTROPY_RATIO = 0.15
MIN_MODE_ENTROPY = 1.20        # bits — uniform = log2(4) ≈ 2.0

# Noise injection
NOISE_INJECTION_RATE = 0.01
DOMAIN_RANDOMIZATION_RATE = 0.03

# ─────────────────────────────────────────────────────────────
# SARSA RL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
@dataclass
class SARSAConfig:
    alpha: float = 0.002           # v3: conservative to preserve warmstart
    alpha_min: float = 0.0005
    alpha_decay: float = 0.996
    gamma: float = 0.85            # discount factor
    epsilon: float = 0.02          # v3: minimal exploration to avoid disruption
    epsilon_min: float = 0.005     # lower floor
    epsilon_decay: float = 0.97    # v3: rapid decay
    lambda_trace: float = 0.15     # v3: short traces to limit propagation
    num_episodes: int = 100        # v3: fewer episodes, preserve warmstart
    max_steps_per_episode: int = 300  # v3: standard step count
    reward_clip_min: float = -2.0
    reward_clip_max: float = 2.0
    q_value_clip: float = 10.0     # v3: standard clip
    entropy_floor: float = 0.06    # v3: reduced
    entropy_bonus_weight: float = 0.001  # v3: minimal entropy bonus
    plateau_patience: int = 30
    plateau_threshold: float = 0.001
    conservative_update_max: float = 0.006  # v3: tight clamp preserves warmstart
    running_reward_window: int = 50
    collapse_detection_window: int = 20
    collapse_variance_threshold: float = 0.01
    multi_seed_count: int = 3


SARSA_CONFIG = SARSAConfig()

# ─────────────────────────────────────────────────────────────
# REWARD FUNCTION PARAMETERS
# ─────────────────────────────────────────────────────────────
REWARD_CORRECT_MODE = 1.0
REWARD_ADJACENT_MODE = 0.3
REWARD_WRONG_MODE = -1.0
REWARD_DANGEROUS_UNDER = -2.5   # v3: +25% (was -2.0)
REWARD_REPETITION_PENALTY = -0.3
REWARD_OSCILLATION_PENALTY = -0.4
REWARD_HABITUATION_PENALTY = -0.2
REWARD_SAFETY_OVERRIDE_BONUS = 0.5
REWARD_M2_CORRECT_BOOST = 2.0   # v3: strong boost for correct M2

# Asymmetric cost matrix penalties (v3)
ASYM_SEVERE_PENALTY = -3.0      # optimal M2/M3 but action M0
ASYM_MEDIUM_PENALTY = -2.0      # optimal M3 but action M1

# Delayed reward window
DELAYED_REWARD_WINDOW = 5       # v3: +2 steps (was 3)

# ─────────────────────────────────────────────────────────────
# TRAINING SPLIT
# ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
K_FOLDS = 5

# ─────────────────────────────────────────────────────────────
# OVERFITTING / UNDERFITTING THRESHOLDS
# ─────────────────────────────────────────────────────────────
OVERFIT_TRAIN_THRESHOLD = 0.95
OVERFIT_VAL_THRESHOLD = 0.75
OVERFIT_GAP_THRESHOLD = 0.20
UNDERFIT_THRESHOLD = 0.65
TARGET_ACCURACY = 0.85            # v3: raised from 0.80

# ─────────────────────────────────────────────────────────────
# SAFETY WRAPPER PARAMETERS
# ─────────────────────────────────────────────────────────────
AGGRESSION_LOCKOUT_SECONDS = 600  # 10 minutes
MAX_ACTIVATIONS_PER_HOUR = 6
MAX_ACTIVATIONS_PER_NIGHT = 12
DEFAULT_COOLDOWN_SECONDS = 120
ESCALATION_COOLDOWN_SECONDS = 300

# ─────────────────────────────────────────────────────────────
# SECURITY PARAMETERS (LoRa)
# ─────────────────────────────────────────────────────────────
LORA_TTL_MAX_SECONDS = 60
LORA_SEQUENCE_WINDOW = 100      # reject if seq gap > this
LORA_VALID_BOUNDARY_IDS = [
    f"village_{v}_segment_{s}"
    for v in range(1, NUM_VILLAGES + 1)
    for s in range(1, NUM_BOUNDARY_SEGMENTS + 1)
]
LORA_CHECKSUM_ALGO = "sha256"
LORA_REPLAY_WINDOW_SIZE = 256

# ─────────────────────────────────────────────────────────────
# EDGE EXPORT
# ─────────────────────────────────────────────────────────────
EDGE_MAX_MEMORY_KB = 200
EDGE_QUANTIZATION_BITS = 8
EDGE_EXPORT_FORMAT = "json"

# ─────────────────────────────────────────────────────────────
# SPEAKER CONFIGURATION (per zone)
# ─────────────────────────────────────────────────────────────
DEFAULT_SPEAKERS_PER_ZONE = 3
SPEAKER_ORIENTATIONS = ["FOREST_FACING", "BOUNDARY_PARALLEL", "VILLAGE_FACING"]

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
GRAPH_DPI = 300
GRAPH_FORMAT = "png"
