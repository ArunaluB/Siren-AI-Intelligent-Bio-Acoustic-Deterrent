"""
Siren AI v2 — Real-Time Safety Wrapper & Security Hardening
════════════════════════════════════════════════════════════════
MANDATORY safety layer that wraps all RL decisions.

Safety Wrapper:
  • Aggression override → M3 immediately
  • Low confidence → M3
  • Missing state fields → prior-based fallback
  • Cooldown active → override RL
  • Entropy collapse → conservative fallback
  • Budget enforcement
  • Lockout timer

Security Hardening (LoRa):
  • Strict input validation
  • Schema enforcement
  • Sequence number validation
  • TTL expiry check
  • Boundary ID validation
  • Replay attack prevention
  • Payload checksum (SHA-256)
  • Signature hash verification

ALL communication via LoRa ONLY. No MQTT.
"""

import os
import time
import json
import hashlib
import struct
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from config import (
    MODES, MODE_INDEX, NUM_MODES,
    AGGRESSION_LOCKOUT_SECONDS, MAX_ACTIVATIONS_PER_HOUR,
    MAX_ACTIVATIONS_PER_NIGHT, DEFAULT_COOLDOWN_SECONDS,
    ESCALATION_COOLDOWN_SECONDS,
    LORA_TTL_MAX_SECONDS, LORA_SEQUENCE_WINDOW,
    LORA_VALID_BOUNDARY_IDS, LORA_CHECKSUM_ALGO,
    LORA_REPLAY_WINDOW_SIZE,
    THREAT_LEVELS, PRESENCE_LEVELS, BREACH_STATUS,
    RISK_LEVELS, CONFIDENCE_BANDS, SEASONS, TIME_OF_DAY, SENSOR_QUALITY,
    DEFAULT_SPEAKERS_PER_ZONE, SPEAKER_ORIENTATIONS,
    SOUND_CUE_MAP,
)


# ═══════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════
@dataclass
class RiskUpdate:
    """Structured RISK_UPDATE message from WCE via LoRa."""
    zone_id: str = ""
    boundary_segment_id: str = ""
    breach_status: str = "NONE"
    risk_level: str = "LOW"
    aggression_flag: bool = False
    distance_band: str = "FAR"
    data_quality: str = "GOOD"
    ttl_seconds: int = 30
    sequence_number: int = 0
    timestamp_utc: float = 0.0
    checksum: str = ""
    signature: str = ""

    # Extended fields for RL state
    threat_level_hint: str = "M0"
    elephant_presence: str = "LOW"
    human_exposure_risk: str = "LOW"
    aggression_risk: str = "LOW"
    confidence_band: str = "MED"
    eta_to_boundary_window_sec: float = 600.0
    recent_activations_24h: int = 0
    last_mode_used: str = "M0"
    cooldown_remaining_sec: int = 0
    season: str = "DRY"
    time_of_day: str = "DAY"
    sensor_quality: str = "GOOD"


@dataclass
class SirenAction:
    """Action output from Siren AI."""
    mode: str = "M0"
    speakers: List[str] = field(default_factory=list)
    sound_cue: Optional[str] = None
    pattern: str = "BURST"
    duration_sec: int = 0
    reason_codes: List[str] = field(default_factory=list)
    safety_override: bool = False
    timestamp: float = 0.0


@dataclass
class ActionLog:
    """Structured action log for auditability."""
    action: SirenAction = field(default_factory=SirenAction)
    rl_suggestion: str = "M0"
    final_mode: str = "M0"
    override_reason: str = ""
    validation_passed: bool = True
    security_flags: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  SECURITY VALIDATOR (LoRa Messages)
# ═══════════════════════════════════════════════════════════════
class LoRaSecurityValidator:
    """
    Validates incoming LoRa RISK_UPDATE messages.
    Prevents replay attacks, expired messages, corrupted payloads.
    """

    def __init__(self, shared_key: str = "wildlife360_siren_v2"):
        self.shared_key = shared_key
        self._last_sequence: Dict[str, int] = {}
        self._seen_checksums: deque = deque(maxlen=LORA_REPLAY_WINDOW_SIZE)
        self._validation_log: List[dict] = []

    def validate_message(self, msg: dict) -> Tuple[bool, List[str]]:
        """
        Full validation pipeline for incoming LoRa message.
        Returns (is_valid, list_of_failure_reasons).
        """
        failures = []

        # 1. Schema enforcement — all required fields
        required_fields = [
            "zone_id", "boundary_segment_id", "breach_status",
            "risk_level", "ttl_seconds", "sequence_number",
            "timestamp_utc", "checksum",
        ]
        for f in required_fields:
            if f not in msg or msg[f] is None:
                failures.append(f"MISSING_FIELD:{f}")

        if failures:
            self._log_validation(msg, False, failures)
            return False, failures

        # 2. Input type & value validation
        type_checks = {
            "zone_id": str,
            "boundary_segment_id": str,
            "breach_status": str,
            "risk_level": str,
            "ttl_seconds": (int, float),
            "sequence_number": (int, float),
            "timestamp_utc": (int, float),
            "checksum": str,
        }
        for f, expected_type in type_checks.items():
            if not isinstance(msg.get(f), expected_type):
                failures.append(f"INVALID_TYPE:{f}")

        # 3. Enumeration validation
        if msg.get("breach_status") not in BREACH_STATUS:
            failures.append(f"INVALID_BREACH_STATUS:{msg.get('breach_status')}")
        if msg.get("risk_level") not in RISK_LEVELS:
            failures.append(f"INVALID_RISK_LEVEL:{msg.get('risk_level')}")

        # 4. Boundary ID validation
        bid = msg.get("boundary_segment_id", "")
        if bid and bid not in LORA_VALID_BOUNDARY_IDS:
            failures.append(f"INVALID_BOUNDARY_ID:{bid}")

        # 5. TTL expiry check
        ttl = msg.get("ttl_seconds", 0)
        ts = msg.get("timestamp_utc", 0)
        now = time.time()
        if ttl > LORA_TTL_MAX_SECONDS:
            failures.append(f"TTL_TOO_LARGE:{ttl}")
        if ts > 0 and (now - ts) > ttl:
            failures.append("MESSAGE_EXPIRED")

        # 6. Sequence number validation
        bid = msg.get("boundary_segment_id", "")
        seq = msg.get("sequence_number", 0)
        last_seq = self._last_sequence.get(bid, -1)
        if seq <= last_seq:
            failures.append(f"STALE_SEQUENCE:{seq}<={last_seq}")
        elif seq > last_seq + LORA_SEQUENCE_WINDOW:
            failures.append(f"SEQUENCE_GAP_TOO_LARGE:{seq}-{last_seq}")

        # 7. Replay attack prevention (checksum dedup)
        checksum = msg.get("checksum", "")
        if checksum in self._seen_checksums:
            failures.append("REPLAY_DETECTED")

        # 8. Payload checksum verification
        expected_checksum = self._compute_checksum(msg)
        if checksum and checksum != expected_checksum:
            failures.append("CHECKSUM_MISMATCH")

        # 9. Signature hash verification
        if "signature" in msg:
            expected_sig = self._compute_signature(msg)
            if msg["signature"] != expected_sig:
                failures.append("SIGNATURE_MISMATCH")

        # Update state if valid
        if not failures:
            self._last_sequence[bid] = seq
            if checksum:
                self._seen_checksums.append(checksum)

        self._log_validation(msg, len(failures) == 0, failures)
        return len(failures) == 0, failures

    def _compute_checksum(self, msg: dict) -> str:
        """Compute SHA-256 checksum of message payload."""
        payload = json.dumps(
            {k: v for k, v in sorted(msg.items()) if k != "checksum"},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _compute_signature(self, msg: dict) -> str:
        """Compute HMAC-style signature with shared key."""
        payload = json.dumps(
            {k: v for k, v in sorted(msg.items())
             if k not in ("checksum", "signature")},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(
            (payload + self.shared_key).encode()
        ).hexdigest()[:16]

    def _log_validation(self, msg: dict, passed: bool, reasons: List[str]):
        """Log validation result."""
        self._validation_log.append({
            "timestamp": time.time(),
            "boundary_id": msg.get("boundary_segment_id", "unknown"),
            "sequence": msg.get("sequence_number", -1),
            "passed": passed,
            "reasons": reasons,
        })

    def get_stats(self) -> dict:
        """Get validation statistics."""
        total = len(self._validation_log)
        passed = sum(1 for v in self._validation_log if v["passed"])
        return {
            "total_validated": total,
            "passed": passed,
            "rejected": total - passed,
            "rejection_rate": (total - passed) / max(total, 1),
        }


# ═══════════════════════════════════════════════════════════════
#  SAFETY WRAPPER (MANDATORY)
# ═══════════════════════════════════════════════════════════════
class SafetyWrapper:
    """
    MANDATORY safety wrapper that overrides ALL RL decisions.
    RL must NEVER bypass safety rules.

    Priority (highest first):
      1. Aggression override → M3
      2. Low confidence → M3
      3. Missing fields → fallback
      4. Cooldown active → override
      5. Budget exceeded → suppress/escalate
      6. Entropy collapse → conservative fallback
    """

    def __init__(self):
        # Lockout state per zone
        self._lockout_timers: Dict[str, float] = {}

        # Activation tracking per zone
        self._activation_counts: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Habituation tracking
        self._habituation_scores: Dict[str, float] = defaultdict(float)
        self._last_mode: Dict[str, str] = {}
        self._last_sound: Dict[str, str] = {}
        self._last_speaker: Dict[str, str] = {}

        # Speaker configuration per zone
        self._speaker_configs: Dict[str, List[dict]] = {}

        # Stats
        self.override_count = 0
        self.aggression_override_count = 0
        self.confidence_override_count = 0
        self.cooldown_override_count = 0
        self.budget_override_count = 0
        self.entropy_override_count = 0
        self.total_decisions = 0

    def safe_decision_wrapper(
        self,
        rl_action: int,
        risk_update: RiskUpdate,
        policy_entropy: float = 1.0,
        entropy_threshold: float = 0.30,
    ) -> SirenAction:
        """
        MAIN SAFETY WRAPPER — wraps every RL decision.
        Returns safe SirenAction that may override RL.
        """
        self.total_decisions += 1
        action = SirenAction(timestamp=time.time())
        reasons = []
        zone = risk_update.boundary_segment_id

        # ── RULE 1: Aggression override → M3 ──
        if risk_update.aggression_flag or risk_update.aggression_risk == "HIGH":
            action.mode = "M3"
            action.safety_override = True
            reasons.append("AGGRESSION_OVERRIDE")
            self.aggression_override_count += 1
            self.override_count += 1

            # Set lockout
            self._lockout_timers[zone] = (
                time.time() + AGGRESSION_LOCKOUT_SECONDS
            )

            action.reason_codes = reasons
            action.sound_cue = "SIREN"
            action.pattern = "ALERT"
            action.duration_sec = 0  # No deterrent — human alert only
            return action

        # ── RULE 2: Low confidence → M3 ──
        if risk_update.confidence_band == "LOW":
            if risk_update.risk_level != "LOW" or risk_update.breach_status != "NONE":
                action.mode = "M3"
                action.safety_override = True
                reasons.append("LOW_CONFIDENCE_ESCALATION")
                self.confidence_override_count += 1
                self.override_count += 1
                action.reason_codes = reasons
                action.sound_cue = "SIREN"
                action.pattern = "ALERT"
                action.duration_sec = 0
                return action

        # ── RULE 3: Missing state fields → prior-based fallback ──
        if self._has_missing_fields(risk_update):
            action.mode = "M3"  # conservative fallback
            action.safety_override = True
            reasons.append("MISSING_FIELDS_FALLBACK")
            self.override_count += 1
            action.reason_codes = reasons
            return action

        # ── RULE 4: Lockout active → suppress deterrents ──
        if zone in self._lockout_timers:
            if time.time() < self._lockout_timers[zone]:
                remaining = self._lockout_timers[zone] - time.time()
                if risk_update.risk_level == "HIGH":
                    action.mode = "M3"
                    reasons.append(f"LOCKOUT_ACTIVE_ESCALATE({remaining:.0f}s)")
                else:
                    action.mode = "M0"
                    reasons.append(f"LOCKOUT_ACTIVE_SUPPRESS({remaining:.0f}s)")
                action.safety_override = True
                self.override_count += 1
                action.reason_codes = reasons
                return action
            else:
                del self._lockout_timers[zone]

        # ── RULE 5: Cooldown active → override RL ──
        if risk_update.cooldown_remaining_sec > 0:
            action.mode = "M0"
            action.safety_override = True
            reasons.append(f"COOLDOWN_ACTIVE({risk_update.cooldown_remaining_sec}s)")
            self.cooldown_override_count += 1
            self.override_count += 1
            action.reason_codes = reasons
            return action

        # ── RULE 6: Budget exceeded → suppress or escalate ──
        hour_count = self._get_activations_last_hour(zone)
        if hour_count >= MAX_ACTIVATIONS_PER_HOUR:
            if risk_update.risk_level == "HIGH":
                action.mode = "M3"
                reasons.append("BUDGET_EXCEEDED_ESCALATE")
            else:
                action.mode = "M0"
                reasons.append("BUDGET_EXCEEDED_SUPPRESS")
            action.safety_override = True
            self.budget_override_count += 1
            self.override_count += 1
            action.reason_codes = reasons
            return action

        # ── RULE 7: Entropy collapse → conservative fallback ──
        if policy_entropy < entropy_threshold:
            # Don't trust RL — use rule-based
            rule_mode = self._rule_based_mode(risk_update)
            action.mode = rule_mode
            action.safety_override = True
            reasons.append("ENTROPY_COLLAPSE_FALLBACK")
            self.entropy_override_count += 1
            self.override_count += 1
            action.reason_codes = reasons
            self._finalize_action(action, risk_update, zone)
            return action

        # ── No override needed — use RL action ──
        rl_mode = MODES[rl_action]
        action.mode = rl_mode
        reasons.append("RL_DECISION")
        action.reason_codes = reasons

        # Finalize (speaker selection, sound cue)
        self._finalize_action(action, risk_update, zone)

        # Record activation
        if action.mode in ("M1", "M2"):
            self._activation_counts[zone].append(time.time())
            self._habituation_scores[zone] = min(
                1.0, self._habituation_scores[zone] + 0.05
            )

        self._last_mode[zone] = action.mode
        return action

    def _has_missing_fields(self, ru: RiskUpdate) -> bool:
        """Check for critical missing fields."""
        critical = [ru.boundary_segment_id, ru.breach_status, ru.risk_level]
        return any(f is None or f == "" for f in critical)

    def _get_activations_last_hour(self, zone: str) -> int:
        """Count activations in the last hour for a zone."""
        now = time.time()
        cutoff = now - 3600
        return sum(1 for t in self._activation_counts[zone] if t > cutoff)

    def _rule_based_mode(self, ru: RiskUpdate) -> str:
        """Conservative rule-based mode selection (Step 6 from spec)."""
        if ru.breach_status == "CONFIRMED":
            return "M3"
        if ru.risk_level == "HIGH" and ru.aggression_risk != "HIGH":
            return "M2"
        if ru.risk_level == "MED":
            return "M1"
        return "M0"

    def _finalize_action(
        self, action: SirenAction, ru: RiskUpdate, zone: str
    ):
        """Select speakers, sound cue, pattern, duration."""
        if action.mode == "M0":
            action.speakers = []
            action.sound_cue = None
            action.pattern = "NONE"
            action.duration_sec = 0
            return

        if action.mode == "M3":
            action.sound_cue = "SIREN"
            action.pattern = "ALERT"
            action.duration_sec = 0
            action.speakers = ["ALL"]  # community alert
            return

        # M1 or M2 — deterrent mode
        # Speaker selection (zone-aware, rotation)
        n_speakers = 1 if action.mode == "M1" else 2
        available = ["FOREST_FACING", "BOUNDARY_PARALLEL"]  # avoid village
        last = self._last_speaker.get(zone)
        if last and last in available and len(available) > 1:
            available.remove(last)  # rotation
        action.speakers = available[:n_speakers]
        self._last_speaker[zone] = action.speakers[0]

        # Sound cue selection (avoid repetition)
        cues = SOUND_CUE_MAP.get(action.mode, ["BEE"])
        last_sound = self._last_sound.get(zone)
        if last_sound and last_sound in cues and len(cues) > 1:
            cues = [c for c in cues if c != last_sound]
        action.sound_cue = cues[0] if cues else "BEE"
        self._last_sound[zone] = action.sound_cue

        # Pattern & duration
        if action.mode == "M1":
            action.pattern = "BURST"
            action.duration_sec = 5
        else:
            action.pattern = "ALT"  # alternating
            action.duration_sec = 10

    def get_stats(self) -> dict:
        """Get safety wrapper statistics."""
        return {
            "total_decisions": self.total_decisions,
            "total_overrides": self.override_count,
            "override_rate": self.override_count / max(self.total_decisions, 1),
            "aggression_overrides": self.aggression_override_count,
            "confidence_overrides": self.confidence_override_count,
            "cooldown_overrides": self.cooldown_override_count,
            "budget_overrides": self.budget_override_count,
            "entropy_overrides": self.entropy_override_count,
        }


# ═══════════════════════════════════════════════════════════════
#  INTEGRATED SAFETY EVALUATION (for test set)
# ═══════════════════════════════════════════════════════════════
def evaluate_safety_wrapper(
    predictions: "np.ndarray",
    labels: "np.ndarray",
    raw_df: "pd.DataFrame",
) -> dict:
    """
    Evaluate safety wrapper performance on test set.
    Returns aggression override precision, false negative rate, etc.
    """
    import numpy as np

    results = {
        "aggression_override_precision": 0.0,
        "false_negative_escalation_rate": 0.0,
        "safety_override_count": 0,
    }

    # Check aggression cases
    agg_mask = raw_df["aggression_risk"] == "HIGH"
    n_agg = agg_mask.sum()

    if n_agg > 0:
        # How many aggression cases did we correctly assign M3?
        agg_preds = predictions[agg_mask.values]
        agg_correct = (agg_preds == 3).sum()  # M3 = index 3
        results["aggression_override_precision"] = float(agg_correct / n_agg)

    # False negative escalation: HIGH risk cases where we chose M0 or M1
    high_risk_mask = (
        (raw_df["human_exposure_risk"] == "HIGH") |
        (raw_df["aggression_risk"] == "HIGH") |
        (raw_df["boundary_breach_status"] == "CONFIRMED")
    )
    n_high = high_risk_mask.sum()
    if n_high > 0:
        under_response = (predictions[high_risk_mask.values] <= 1).sum()
        results["false_negative_escalation_rate"] = float(under_response / n_high)

    # Safety override simulation
    conf_low_mask = raw_df["confidence_band"] == "LOW"
    safety_overrides = agg_mask.sum() + conf_low_mask.sum()
    results["safety_override_count"] = int(safety_overrides)

    return results
