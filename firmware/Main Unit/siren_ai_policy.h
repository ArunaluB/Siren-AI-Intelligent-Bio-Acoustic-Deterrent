/*
 * Siren AI v2 — ESP32-S3 Inference Header
 * AUTO-GENERATED — DO NOT EDIT
 * Wildlife 360 Conservative Deterrent Orchestrator
 * LoRa-only communication. No MQTT.
 */

#ifndef SIREN_AI_POLICY_H
#define SIREN_AI_POLICY_H

#include <stdint.h>
#include <stdbool.h>

/* ═══ Mode Definitions ═══ */
#define SIREN_MODE_M0  0  /* Observe */
#define SIREN_MODE_M1  1  /* Minimal deterrent */
#define SIREN_MODE_M2  2  /* Stronger deterrent */
#define SIREN_MODE_M3  3  /* Human escalation */
#define SIREN_NUM_MODES 4

/* ═══ Feature Encoding ═══ */
#define NUM_FEATURES 13

/* threat_level_hint */
#define FEAT_THREAT_LEVEL_HINT_M0  0
#define FEAT_THREAT_LEVEL_HINT_M1  1
#define FEAT_THREAT_LEVEL_HINT_M2  2
#define FEAT_THREAT_LEVEL_HINT_M3  3

/* elephant_presence */
#define FEAT_ELEPHANT_PRESENCE_LOW  0
#define FEAT_ELEPHANT_PRESENCE_MED  1
#define FEAT_ELEPHANT_PRESENCE_HIGH  2

/* boundary_breach_status */
#define FEAT_BOUNDARY_BREACH_STATUS_NONE  0
#define FEAT_BOUNDARY_BREACH_STATUS_LIKELY  1
#define FEAT_BOUNDARY_BREACH_STATUS_CONFIRMED  2

/* human_exposure_risk */
#define FEAT_HUMAN_EXPOSURE_RISK_LOW  0
#define FEAT_HUMAN_EXPOSURE_RISK_MED  1
#define FEAT_HUMAN_EXPOSURE_RISK_HIGH  2

/* aggression_risk */
#define FEAT_AGGRESSION_RISK_LOW  0
#define FEAT_AGGRESSION_RISK_MED  1
#define FEAT_AGGRESSION_RISK_HIGH  2

/* confidence_band */
#define FEAT_CONFIDENCE_BAND_LOW  0
#define FEAT_CONFIDENCE_BAND_MED  1
#define FEAT_CONFIDENCE_BAND_HIGH  2

/* last_mode_used */
#define FEAT_LAST_MODE_USED_M0  0
#define FEAT_LAST_MODE_USED_M1  1
#define FEAT_LAST_MODE_USED_M2  2
#define FEAT_LAST_MODE_USED_M3  3

/* season */
#define FEAT_SEASON_DRY  0
#define FEAT_SEASON_WET  1
#define FEAT_SEASON_HARVEST  2

/* time_of_day */
#define FEAT_TIME_OF_DAY_DAY  0
#define FEAT_TIME_OF_DAY_NIGHT  1

/* sensor_quality */
#define FEAT_SENSOR_QUALITY_GOOD  0
#define FEAT_SENSOR_QUALITY_DEGRADED  1

/* ═══ Quantization ═══ */
#define Q_SCALE  0.05634586f
#define Q_ZERO_POINT  0
#define Q_BITS  8

/* ═══ Safety Constants ═══ */
#define AGGRESSION_LOCKOUT_SEC  600
#define MAX_ACTIVATIONS_HOUR  6
#define DEFAULT_COOLDOWN_SEC  120

/* ═══ Safe Decision Wrapper ═══ */
static inline uint8_t siren_safe_decision(
    uint8_t rl_action,
    uint8_t aggression_risk,   /* 0=LOW, 1=MED, 2=HIGH */
    uint8_t confidence_band,   /* 0=LOW, 1=MED, 2=HIGH */
    uint8_t breach_status,     /* 0=NONE, 1=LIKELY, 2=CONFIRMED */
    uint8_t human_risk,        /* 0=LOW, 1=MED, 2=HIGH */
    uint16_t cooldown_sec,
    uint8_t activations_24h,
    bool aggression_flag
) {
    /* RULE 1: Aggression → M3 */
    if (aggression_flag || aggression_risk >= 2) return SIREN_MODE_M3;

    /* RULE 2: Low confidence + any risk → M3 */
    if (confidence_band == 0 && (human_risk > 0 || breach_status > 0))
        return SIREN_MODE_M3;

    /* RULE 3: Confirmed breach → M3 */
    if (breach_status >= 2) return SIREN_MODE_M3;

    /* RULE 4: Cooldown active → M0 */
    if (cooldown_sec > 0) return SIREN_MODE_M0;

    /* RULE 5: Budget exceeded */
    if (activations_24h >= 6) {
        if (human_risk >= 2) return SIREN_MODE_M3;
        return SIREN_MODE_M0;
    }

    /* No override — use RL action */
    return (rl_action < SIREN_NUM_MODES) ? rl_action : SIREN_MODE_M0;
}

/* ═══ Rule-Based Fallback ═══ */
static inline uint8_t siren_fallback_mode(
    uint8_t threat_level,
    uint8_t human_risk,
    uint8_t breach_status
) {
    if (breach_status >= 2) return SIREN_MODE_M3;
    if (human_risk >= 2) return SIREN_MODE_M2;
    if (threat_level >= 3) return SIREN_MODE_M3;
    if (threat_level >= 2) return SIREN_MODE_M2;
    if (threat_level >= 1) return SIREN_MODE_M1;
    return SIREN_MODE_M0;
}

/* ═══ LoRa Message Validation ═══ */
typedef struct {
    char zone_id[32];
    char boundary_id[64];
    uint8_t breach_status;
    uint8_t risk_level;
    bool aggression_flag;
    uint8_t distance_band;
    uint8_t data_quality;
    uint16_t ttl_seconds;
    uint32_t sequence_number;
    uint32_t timestamp;
    uint16_t checksum;
} siren_risk_update_t;

static inline bool siren_validate_message(
    const siren_risk_update_t* msg,
    uint32_t last_seq,
    uint32_t current_time
) {
    /* TTL check */
    if ((current_time - msg->timestamp) > msg->ttl_seconds) return false;

    /* Sequence check */
    if (msg->sequence_number <= last_seq) return false;

    /* Valid enum ranges */
    if (msg->breach_status > 2) return false;
    if (msg->risk_level > 2) return false;
    if (msg->data_quality > 1) return false;

    return true;
}

#endif /* SIREN_AI_POLICY_H */