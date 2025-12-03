"""
         --------Namo Buddhaya--------
Siren AI v2 — Enterprise Dataset Engine
═══════════════════════════════════════════
Generates 100,000 realistic samples simulating 1 year of field deployment.

Features:
  • Multiple villages & boundary segments
  • Seasonal / diurnal variation
  • Elephant behaviour archetypes (Timid, Habitual, Aggressive, Adaptive)
  • Sensor degradation states
  • Uncertainty bands, ETA windows
  • Random rare aggression spikes
  • Cooldown & habituation memory
  • Stratified sampling & rare-event oversampling
  • Domain randomization & controlled noise injection
  • Imbalance validation before training

LoRa-only context — no MQTT.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional

from config import (
    DATASET_SIZE, NUM_VILLAGES, NUM_BOUNDARY_SEGMENTS,
    ELEPHANT_BEHAVIOR_TYPES, BEHAVIOR_WEIGHTS,
    THREAT_LEVELS, PRESENCE_LEVELS, BREACH_STATUS, RISK_LEVELS,
    CONFIDENCE_BANDS, SEASONS, TIME_OF_DAY, SENSOR_QUALITY, MODES, NUM_MODES,
    SEASON_WEIGHTS, TIME_WEIGHTS, MODE_INDEX,
    RARE_AGGRESSION_SPIKE_PROB, SENSOR_DEGRADATION_PROB,
    NOISE_INJECTION_RATE, DOMAIN_RANDOMIZATION_RATE,
    MAX_CLASS_SKEW, MIN_MODE_ENTROPY,
    ENCODINGS, ETA_BINS, ACTIVATION_BINS, COOLDOWN_BINS,
    DATA_DIR, FEATURE_NAMES,
)


# ═══════════════════════════════════════════════════════════════
#  HELPER: LABEL ASSIGNMENT (non-trivial, non-linearly-separable)
# ═══════════════════════════════════════════════════════════════
def _assign_optimal_mode(row: dict, rng: np.random.Generator) -> str:
    """
    Assign optimal_mode using a complex, non-linear rule set.
    This prevents linear separability.
    """
    threat = row["threat_level_hint"]
    presence = row["elephant_presence"]
    breach = row["boundary_breach_status"]
    human_risk = row["human_exposure_risk"]
    aggression = row["aggression_risk"]
    confidence = row["confidence_band"]
    eta = row["eta_to_boundary_window_sec"]
    activations = row["recent_activations_24h"]
    cooldown = row["cooldown_remaining_sec"]
    season = row["season"]
    time = row["time_of_day"]
    sensor = row["sensor_quality"]
    behavior = row.get("elephant_behavior", "TIMID")

    # ── RULE 1: Aggression always → M3 ──
    if aggression == "HIGH":
        return "M3"

    # ── RULE 2: Low confidence + any risk → M3 (safety-first) ──
    if confidence == "LOW" and (human_risk != "LOW" or breach != "NONE"):
        return "M3"

    # ── RULE 3: Confirmed breach → M3 ──
    if breach == "CONFIRMED":
        return "M3"

    # ── RULE 4: High cooldown active → M0 override ──
    if cooldown > 600:
        return "M0"

    # ── RULE 5: Budget exceeded → M0 or escalate ──
    if activations > 10:
        if human_risk == "HIGH":
            return "M3"
        return "M0"

    # ── RULE 6: Night + degraded sensor + medium+ risk → M3 ──
    if time == "NIGHT" and sensor == "DEGRADED" and human_risk in ("MED", "HIGH"):
        return "M3"

    # ── RULE 7: Interaction-based rules (non-linear) ──
    # -- Habitual elephants + repeated deterrence → escalate
    if behavior == "HABITUAL" and activations > 5:
        if threat in ("M1", "M2"):
            return "M2"  # step up
        return "M3"

    # -- Adaptive elephants + harvest season → harder to deter
    if behavior == "ADAPTIVE" and season == "HARVEST":
        if presence == "HIGH" and breach == "LIKELY":
            return "M2" if rng.random() > 0.3 else "M3"

    # -- Timid elephants → lighter response
    if behavior == "TIMID" and aggression == "LOW":
        if presence == "MED" and breach == "LIKELY":
            return "M1"
        if presence == "LOW":
            return "M0"

    # ── RULE 8: ETA-based urgency ──
    if eta < 60 and breach == "LIKELY" and human_risk in ("MED", "HIGH"):
        return "M2"
    if eta < 30 and human_risk == "HIGH":
        return "M3"

    # ── RULE 9: Standard threat-level mapping with noise ──
    base_map = {"M0": "M0", "M1": "M1", "M2": "M2", "M3": "M3"}
    base_mode = base_map.get(threat, "M0")

    # Contextual adjustments
    if human_risk == "HIGH" and base_mode in ("M0", "M1"):
        base_mode = "M2"
    if presence == "HIGH" and base_mode == "M0":
        base_mode = "M1"
    if sensor == "DEGRADED" and base_mode in ("M0", "M1"):
        # conservative bump
        mode_idx = MODE_INDEX[base_mode]
        base_mode = MODES[min(mode_idx + 1, 3)]

    # ── Stochastic perturbation (prevent linear separability) ──
    if rng.random() < 0.01:
        mode_idx = MODE_INDEX[base_mode]
        delta = rng.choice([-1, 1])
        new_idx = max(0, min(3, mode_idx + delta))
        base_mode = MODES[new_idx]

    return base_mode


# ═══════════════════════════════════════════════════════════════
#  SAMPLE GENERATOR
# ═══════════════════════════════════════════════════════════════
def _generate_single_sample(
    rng: np.random.Generator,
    village_id: int,
    segment_id: int,
    behavior: str,
    season: str,
    time_of_day: str,
    habituation_score: float,
    last_mode: str,
) -> dict:
    """Generate one realistic sample with all required features."""

    # Threat level hint (from WCE)
    if behavior == "AGGRESSIVE":
        threat_weights = [0.05, 0.15, 0.35, 0.45]
    elif behavior == "HABITUAL":
        threat_weights = [0.10, 0.30, 0.35, 0.25]
    elif behavior == "ADAPTIVE":
        threat_weights = [0.15, 0.25, 0.35, 0.25]
    else:  # TIMID
        threat_weights = [0.35, 0.30, 0.25, 0.10]

    # Season modifiers
    if season == "HARVEST":
        threat_weights[2] += 0.05
        threat_weights[3] += 0.05
        threat_weights[0] -= 0.05
        threat_weights[1] -= 0.05
    elif season == "WET":
        threat_weights[0] += 0.05
        threat_weights[3] -= 0.05

    # Normalize
    tw = np.array(threat_weights, dtype=np.float64)
    tw = np.clip(tw, 0.01, None)
    tw /= tw.sum()

    threat = rng.choice(THREAT_LEVELS, p=tw)

    # Presence
    if threat in ("M2", "M3"):
        presence = rng.choice(PRESENCE_LEVELS, p=[0.1, 0.3, 0.6])
    elif threat == "M1":
        presence = rng.choice(PRESENCE_LEVELS, p=[0.25, 0.45, 0.30])
    else:
        presence = rng.choice(PRESENCE_LEVELS, p=[0.55, 0.30, 0.15])

    # Breach status
    if threat == "M3":
        breach = rng.choice(BREACH_STATUS, p=[0.05, 0.25, 0.70])
    elif threat == "M2":
        breach = rng.choice(BREACH_STATUS, p=[0.15, 0.50, 0.35])
    elif threat == "M1":
        breach = rng.choice(BREACH_STATUS, p=[0.35, 0.45, 0.20])
    else:
        breach = rng.choice(BREACH_STATUS, p=[0.70, 0.22, 0.08])

    # Human exposure risk
    if time_of_day == "NIGHT":
        human_weights = [0.45, 0.35, 0.20]
    else:
        human_weights = [0.30, 0.35, 0.35]
    if season == "HARVEST":
        human_weights[2] += 0.10
        human_weights[0] -= 0.10
    hw = np.clip(human_weights, 0.01, None)
    hw /= sum(hw)
    human_risk = rng.choice(RISK_LEVELS, p=hw)

    # Aggression risk
    if behavior == "AGGRESSIVE":
        agg_weights = [0.15, 0.30, 0.55]
    elif behavior == "ADAPTIVE":
        agg_weights = [0.30, 0.40, 0.30]
    else:
        agg_weights = [0.55, 0.30, 0.15]

    # Rare aggression spike
    if rng.random() < RARE_AGGRESSION_SPIKE_PROB:
        agg_weights = [0.05, 0.15, 0.80]

    aw = np.clip(agg_weights, 0.01, None)
    aw /= sum(aw)
    aggression = rng.choice(RISK_LEVELS, p=aw)

    # Confidence band
    if rng.random() < SENSOR_DEGRADATION_PROB:
        sensor = "DEGRADED"
        confidence = rng.choice(CONFIDENCE_BANDS, p=[0.50, 0.35, 0.15])
    else:
        sensor = "GOOD"
        confidence = rng.choice(CONFIDENCE_BANDS, p=[0.10, 0.30, 0.60])

    # ETA to boundary (seconds)
    if breach == "CONFIRMED":
        eta = max(0, rng.exponential(30))
    elif breach == "LIKELY":
        eta = max(0, rng.exponential(180))
    else:
        eta = max(0, rng.exponential(600))
    eta = min(eta, 3600)

    # Recent activations
    base_acts = rng.poisson(3)
    if habituation_score > 0.5:
        base_acts += rng.integers(2, 6)
    activations = min(base_acts, 20)

    # Cooldown remaining
    if activations > 0 and rng.random() < 0.4:
        cooldown = rng.integers(0, 900)
    else:
        cooldown = 0

    sample = {
        "village_id": f"village_{village_id}",
        "boundary_segment_id": f"village_{village_id}_segment_{segment_id}",
        "elephant_behavior": behavior,
        "threat_level_hint": threat,
        "elephant_presence": presence,
        "boundary_breach_status": breach,
        "human_exposure_risk": human_risk,
        "aggression_risk": aggression,
        "confidence_band": confidence,
        "eta_to_boundary_window_sec": round(eta, 1),
        "recent_activations_24h": activations,
        "last_mode_used": last_mode,
        "cooldown_remaining_sec": cooldown,
        "season": season,
        "time_of_day": time_of_day,
        "sensor_quality": sensor,
    }

    # Assign label
    sample["optimal_mode"] = _assign_optimal_mode(sample, rng)
    return sample


# ═══════════════════════════════════════════════════════════════
#  MAIN DATASET GENERATION
# ═══════════════════════════════════════════════════════════════
def generate_dataset(
    n_samples: int = DATASET_SIZE,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate enterprise-grade 100K sample dataset simulating 1 year.
    Includes stratified sampling, rare-event oversampling,
    domain randomization, and noise injection.
    """
    rng = np.random.default_rng(seed)
    samples = []

    # Pre-compute distribution targets
    samples_per_village = n_samples // NUM_VILLAGES
    remaining = n_samples - (samples_per_village * NUM_VILLAGES)

    if verbose:
        print(f"[DATASET] Generating {n_samples:,} samples across "
              f"{NUM_VILLAGES} villages...")

    for v in range(1, NUM_VILLAGES + 1):
        village_samples = samples_per_village + (1 if v <= remaining else 0)
        habituation_scores = {}

        for i in range(village_samples):
            # Stratified sampling: season & time
            season = rng.choice(SEASONS, p=list(SEASON_WEIGHTS.values()))
            tod = rng.choice(TIME_OF_DAY, p=list(TIME_WEIGHTS.values()))
            segment = rng.integers(1, NUM_BOUNDARY_SEGMENTS + 1)
            behavior = rng.choice(ELEPHANT_BEHAVIOR_TYPES, p=BEHAVIOR_WEIGHTS)

            # Habituation memory per segment
            seg_key = f"v{v}_s{segment}"
            if seg_key not in habituation_scores:
                habituation_scores[seg_key] = rng.uniform(0, 0.3)

            # Update habituation over time
            if rng.random() < 0.1:
                habituation_scores[seg_key] = min(
                    1.0, habituation_scores[seg_key] + rng.uniform(0, 0.1)
                )

            last_mode = rng.choice(MODES, p=[0.40, 0.25, 0.20, 0.15])

            sample = _generate_single_sample(
                rng, v, segment, behavior, season, tod,
                habituation_scores[seg_key], last_mode,
            )
            samples.append(sample)

    df = pd.DataFrame(samples)

    # ── Domain randomization ──
    n_domain = int(len(df) * DOMAIN_RANDOMIZATION_RATE)
    if n_domain > 0:
        idx = rng.choice(len(df), size=n_domain, replace=False)
        for i in idx:
            feat = rng.choice(["eta_to_boundary_window_sec",
                               "recent_activations_24h",
                               "cooldown_remaining_sec"])
            if feat == "eta_to_boundary_window_sec":
                df.at[i, feat] = round(rng.uniform(0, 3600), 1)
            elif feat == "recent_activations_24h":
                df.at[i, feat] = rng.integers(0, 20)
            else:
                df.at[i, feat] = rng.integers(0, 900)

    # ── Controlled noise injection ──
    n_noise = int(len(df) * NOISE_INJECTION_RATE)
    if n_noise > 0:
        idx = rng.choice(len(df), size=n_noise, replace=False)
        for i in idx:
            # Flip label to adjacent mode
            current = df.at[i, "optimal_mode"]
            curr_idx = MODE_INDEX[current]
            delta = rng.choice([-1, 1])
            new_idx = max(0, min(3, curr_idx + delta))
            df.at[i, "optimal_mode"] = MODES[new_idx]

    # ── Balance enforcement via oversampling ──
    df = _balance_dataset(df, rng, verbose)

    if verbose:
        print(f"[DATASET] Final size: {len(df):,} samples")

    return df


def _balance_dataset(
    df: pd.DataFrame,
    rng: np.random.Generator,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ensure balanced class distribution via stratified oversampling.
    Target: each mode within 15% of uniform share.
    """
    mode_counts = df["optimal_mode"].value_counts()
    total = len(df)
    target_per_mode = total // NUM_MODES  # uniform target

    # v3: M2 gets +5% oversampling target
    m2_target = int(target_per_mode * 1.30)

    if verbose:
        print(f"[BALANCE] Pre-balance distribution:")
        for mode in MODES:
            cnt = mode_counts.get(mode, 0)
            pct = cnt / total * 100
            print(f"  {mode}: {cnt:,} ({pct:.1f}%)")

    # Oversample underrepresented modes
    frames = [df]
    for mode in MODES:
        cnt = mode_counts.get(mode, 0)
        # v3: Use higher target for M2
        mode_target = m2_target if mode == "M2" else target_per_mode
        if cnt < mode_target * 0.85:
            deficit = mode_target - cnt
            mode_df = df[df["optimal_mode"] == mode]
            if len(mode_df) > 0:
                oversampled = mode_df.sample(
                    n=deficit, replace=True, random_state=int(rng.integers(0, 2**31))
                )
                # Add small noise to oversampled continuous features
                for col in ["eta_to_boundary_window_sec", "cooldown_remaining_sec"]:
                    noise = rng.normal(0, 5, size=len(oversampled))
                    oversampled[col] = np.clip(
                        oversampled[col].values + noise, 0, 3600
                    )
                frames.append(oversampled)

    if len(frames) > 1:
        df = pd.concat(frames, ignore_index=True)

    # Downsample overrepresented modes
    mode_counts = df["optimal_mode"].value_counts()
    max_allowed = int(target_per_mode * 1.15)
    trimmed_frames = []
    for mode in MODES:
        mode_df = df[df["optimal_mode"] == mode]
        if len(mode_df) > max_allowed:
            mode_df = mode_df.sample(
                n=max_allowed, replace=False,
                random_state=int(rng.integers(0, 2**31))
            )
        trimmed_frames.append(mode_df)
    df = pd.concat(trimmed_frames, ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    if verbose:
        mode_counts = df["optimal_mode"].value_counts()
        total = len(df)
        print(f"[BALANCE] Post-balance distribution:")
        for mode in MODES:
            cnt = mode_counts.get(mode, 0)
            pct = cnt / total * 100
            print(f"  {mode}: {cnt:,} ({pct:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════
#  IMBALANCE VALIDATION
# ═══════════════════════════════════════════════════════════════
def validate_imbalance(df: pd.DataFrame) -> dict:
    """
    Compute imbalance metrics and abort if skew > 15%.
    Returns dict with all metrics and pass/fail status.
    """
    total = len(df)
    results = {"passed": True, "warnings": [], "metrics": {}}

    # 1. Class (mode) distribution
    mode_counts = df["optimal_mode"].value_counts(normalize=True)
    for mode in MODES:
        pct = mode_counts.get(mode, 0)
        if pct > 0.40:
            results["warnings"].append(f"Mode {mode} over-represented: {pct:.2%}")
            results["passed"] = False
        if pct < 0.10:
            results["warnings"].append(f"Mode {mode} under-represented: {pct:.2%}")
            results["passed"] = False
    results["metrics"]["mode_distribution"] = mode_counts.to_dict()

    # 2. Mode entropy
    probs = np.array([mode_counts.get(m, 1e-10) for m in MODES])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    results["metrics"]["mode_entropy"] = float(entropy)
    if entropy < MIN_MODE_ENTROPY:
        results["warnings"].append(f"Mode entropy too low: {entropy:.3f} < {MIN_MODE_ENTROPY}")
        results["passed"] = False

    # 3. Aggression ratio
    if "aggression_risk" in df.columns:
        agg_ratio = (df["aggression_risk"] == "HIGH").mean()
        results["metrics"]["aggression_ratio"] = float(agg_ratio)
        if agg_ratio < 0.05:
            results["warnings"].append(f"Aggression ratio too low: {agg_ratio:.3%}")
        if agg_ratio > 0.50:
            results["warnings"].append(f"Aggression ratio too high: {agg_ratio:.3%}")

    # 4. State frequency (check if any single state dominates)
    cat_cols = ["threat_level_hint", "elephant_presence",
                "boundary_breach_status", "aggression_risk"]
    for col in cat_cols:
        if col in df.columns:
            freqs = df[col].value_counts(normalize=True)
            max_freq = freqs.max()
            if max_freq > 0.65:
                results["warnings"].append(
                    f"State '{col}' dominated by '{freqs.idxmax()}': {max_freq:.2%}"
                )

    # 5. Season & time balance
    for col in ["season", "time_of_day"]:
        if col in df.columns:
            freqs = df[col].value_counts(normalize=True)
            results["metrics"][f"{col}_distribution"] = freqs.to_dict()

    return results


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENCODING (STATE VECTOR)
# ═══════════════════════════════════════════════════════════════
def encode_features(df: pd.DataFrame) -> np.ndarray:
    """
    Encode all features into a numeric state matrix.
    Categorical → integer codes, continuous → binned.
    """
    encoded = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)

    for i, feat in enumerate(FEATURE_NAMES):
        if feat in ENCODINGS:
            encoded[:, i] = df[feat].map(ENCODINGS[feat]).fillna(0).values
        elif feat == "eta_to_boundary_window_sec":
            bins = [0, 30, 120, 300, 600, 1800, np.inf]
            encoded[:, i] = np.digitize(df[feat].values, bins) - 1
        elif feat == "recent_activations_24h":
            bins = [0, 1, 3, 6, 11, np.inf]
            encoded[:, i] = np.digitize(df[feat].values, bins) - 1
        elif feat == "cooldown_remaining_sec":
            bins = [0, 1, 300, 900, np.inf]
            encoded[:, i] = np.digitize(df[feat].values, bins) - 1
        else:
            encoded[:, i] = df[feat].values

    return encoded


def state_to_key(state_vector: np.ndarray) -> str:
    """
    Convert state vector to a hashable string key for Q-table.
    v3: 13 discriminative features — added sensor_quality (Rule 6) and
    binary season (HARVEST vs non-HARVEST, Rule 7) for proper M2/M3
    discrimination. Compressed eta/activations (3→2 bins) to offset.
    """
    threat = int(state_vector[0])             # 4 vals
    presence = int(state_vector[1])           # 3 vals
    breach = int(state_vector[2])             # 3 vals
    human_risk = int(state_vector[3])         # 3 vals
    aggression = int(state_vector[4])         # 3 vals
    confidence = int(state_vector[5])         # 3 vals
    eta = min(int(state_vector[6]) // 3, 1)   # v3: 6 → 2 bins (near/far)
    activations = min(int(state_vector[7]) // 3, 1)  # v3: 5 → 2 bins (few/many)
    last_mode = int(state_vector[8])          # 4 vals
    cooldown = min(int(state_vector[9]) // 2, 1)     # 4 → 2 bins
    season_harvest = 1 if int(state_vector[10]) == 2 else 0  # v3: binary (HARVEST=1)
    time_of_day = int(state_vector[11])       # 2 vals
    sensor = int(state_vector[12])            # v3: 2 vals (GOOD/DEGRADED)

    return f"{threat},{presence},{breach},{human_risk},{aggression},{confidence},{eta},{activations},{last_mode},{cooldown},{season_harvest},{time_of_day},{sensor}"


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    """Encode optimal_mode labels to integer indices."""
    return df["optimal_mode"].map(MODE_INDEX).values.astype(np.int32)


# ═══════════════════════════════════════════════════════════════
#  SOUND FILE FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════
def extract_sound_features(sound_dir: str) -> dict:
    """
    Extract basic audio features from available sound files
    for integration into the training pipeline.
    Returns a dict of {category: [file_metadata, ...]}.
    """
    import warnings
    sound_features = {}

    for category, cat_dir in [("BEE", os.path.join(sound_dir, "Bee")),
                               ("LEOPARD", os.path.join(sound_dir, "Leopard")),
                               ("SIREN", os.path.join(sound_dir, "Siren"))]:
        sound_features[category] = []
        if not os.path.exists(cat_dir):
            continue

        for fname in os.listdir(cat_dir):
            fpath = os.path.join(cat_dir, fname)
            if not os.path.isfile(fpath):
                continue

            meta = {
                "filename": fname,
                "filepath": fpath,
                "category": category,
                "size_bytes": os.path.getsize(fpath),
            }

            # Try to extract audio features
            try:
                import librosa
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y, sr = librosa.load(fpath, sr=22050, duration=10.0)
                    meta["duration_sec"] = float(librosa.get_duration(y=y, sr=sr))
                    meta["sample_rate"] = sr
                    meta["rms_energy"] = float(np.sqrt(np.mean(y**2)))

                    # Spectral features
                    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    meta["spectral_centroid_mean"] = float(np.mean(spec_centroid))

                    # Zero crossing rate
                    zcr = librosa.feature.zero_crossing_rate(y)
                    meta["zcr_mean"] = float(np.mean(zcr))

            except Exception:
                meta["duration_sec"] = None
                meta["sample_rate"] = None
                meta["rms_energy"] = None

            sound_features[category].append(meta)

    return sound_features


# ═══════════════════════════════════════════════════════════════
#  CLI ENTRY
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  SIREN AI v2 — ENTERPRISE DATASET ENGINE")
    print("=" * 60)

    df = generate_dataset(n_samples=DATASET_SIZE, seed=42, verbose=True)

    # Validate
    val_result = validate_imbalance(df)
    print(f"\n[VALIDATION] Passed: {val_result['passed']}")
    for w in val_result["warnings"]:
        print(f"  ⚠ {w}")
    print(f"  Mode entropy: {val_result['metrics']['mode_entropy']:.4f}")

    # Save
    out_path = os.path.join(DATA_DIR, "siren_dataset_100k.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Sound features
    from config import SOUND_DIR
    sf = extract_sound_features(SOUND_DIR)
    for cat, files in sf.items():
        print(f"\n[SOUND] {cat}: {len(files)} files")
        for f in files:
            dur = f.get("duration_sec")
            dur_str = f"{dur:.1f}s" if dur else "N/A"
            print(f"  - {f['filename']} ({dur_str})")
