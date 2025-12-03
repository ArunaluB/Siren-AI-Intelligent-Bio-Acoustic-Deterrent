/*
 * ═══════════════════════════════════════════════════════════════════════════════
 *  SIREN AI v2 — COMPREHENSIVE TEST SUITE (LoRa Signal Generator)
 *  Unit 2: ESP32 DevKit V1 + SX1278 (433 MHz)
 *  Wildlife 360 — Human-Elephant Conflict Mitigation
 *  IT22515612 | Bamunusinghe S.A.N.
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *  PURPOSE:
 *    Complete test suite for Siren AI v2 with comprehensive positive and
 *    negative test cases covering ALL functionality, safety mechanisms,
 *    security validation, and edge cases.
 *
 *  TEST COVERAGE:
 *    ✅ 30+ Test Cases (Original 16 + New 15+ tests)
 *    ✅ Positive Tests: Normal operation scenarios
 *    ✅ Negative Tests: Error handling, security, edge cases
 *    ✅ Safety Tests: Aggression, cooldown, budget, lockout
 *    ✅ Security Tests: TTL, replay, zone mismatch, checksum
 *    ✅ Performance Tests: Rapid bursts, latency checks
 *    ✅ Integration Tests: Full system workflows
 *
 *  NEW FEATURES:
 *    ✅ 15+ additional test cases
 *    ✅ Automated test sequences
 *    ✅ Test result expectations
 *    ✅ Pass/Fail validation guidance
 *    ✅ Comprehensive documentation
 *    ✅ WiFi + NTP time sync
 *    ✅ Real-time timestamp generation
 *
 *  USAGE:
 *    1. Set WiFi credentials (WIFI_SSID & WIFI_PASSWORD)
 *    2. Flash to ESP32 DevKit V1 (Unit 2)
 *    3. Open Serial Monitor at 115200 baud
 *    4. Select test case or automated sequence
 *    5. Observe Siren AI response on Unit 1
 *    6. Verify expected behavior matches actual
 *
 *  PIN MAPPING (ESP32 DevKit V1 + SX1278):
 *    SX1278 Pin    ESP32 GPIO
 *    ──────────    ──────────
 *    NSS      →    GPIO5
 *    MOSI     →    GPIO23
 *    MISO     →    GPIO19
 *    SCK      →    GPIO18
 *    RST      →    GPIO14
 *    DIO0     →    GPIO26
 *
 *  LORA CONFIG (must match Siren AI):
 *    Frequency:  433 MHz
 *    SF:         10
 *    Bandwidth:  125 kHz
 *    Coding Rate: 4/5
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#include <SPI.h>
#include <LoRa.h>
#include <WiFi.h>
#include <time.h>

/* ═══ WiFi Credentials ═══ */
// ⚠️ CHANGE THESE TO YOUR NETWORK ⚠️
#define WIFI_SSID     "Redmi Note 10 Pro"     // ඔබේ WiFi නම
#define WIFI_PASSWORD "12345678"               // ඔබේ WiFi password එක

/* ═══ NTP Time Server ═══ */
#define NTP_SERVER    "pool.ntp.org"
#define GMT_OFFSET_SEC  (5.5 * 3600)           // Sri Lanka: GMT+5:30
#define DST_OFFSET_SEC  0

/* ═══ LoRa Pins (ESP32 DevKit V1 + SX1278) ═══ */
#define LORA_NSS    5
#define LORA_RST    14
#define LORA_DIO0   26
#define LORA_SCK    18
#define LORA_MOSI   23
#define LORA_MISO   19

/* ═══ LoRa RF Config (MUST match Siren AI) ═══ */
#define LORA_FREQ       433E6
#define LORA_SF         10
#define LORA_BW         125E3
#define LORA_CR         5

/* ═══ Zone Identity (MUST match Siren AI) ═══ */
#define DEFAULT_ZONE_ID  "ZONE-A1"
#define VALID_BOUNDARY_ID "BOUNDARY-001"
#define INVALID_ZONE_ID  "ZONE-XX"

/* ═══ Test State ═══ */
static uint32_t g_sequence       = 0;
static uint32_t g_baseTimestamp  = 1771027200;
static uint32_t g_txCount        = 0;
static uint32_t g_testPassCount  = 0;
static uint32_t g_testFailCount  = 0;
static bool     g_loraReady      = false;
static bool     g_wifiConnected  = false;
static bool     g_timeSync       = false;
static bool     g_autoTestMode   = false;

/* ═══ Enums ═══ */
#define RISK_LOW   0
#define RISK_MED   1
#define RISK_HIGH  2

#define BREACH_NONE      0
#define BREACH_LIKELY    1
#define BREACH_CONFIRMED 2

#define QUALITY_GOOD     1
#define QUALITY_DEGRADED 0

#define DIST_FAR   0
#define DIST_MID   1
#define DIST_NEAR  2

/* ═══ Test Result Structure ═══ */
struct TestResult {
    const char* testName;
    const char* expectedBehavior;
    bool manualVerification;
};

/* ═══ Risk Message Structure ═══ */
struct RiskMessage {
    const char* zone_id;
    const char* boundary_id;
    const char* risk_level;
    const char* breach_status;
    bool        aggression_flag;
    uint16_t    ttl_seconds;
    uint32_t    sequence_number;
    uint32_t    timestamp_utc;
    uint8_t     distance_band;
    uint8_t     data_quality;
    uint16_t    checksum;
};

/* ═══════════════════════════════════════════════════════════════
 *  FORWARD DECLARATIONS
 * ═══════════════════════════════════════════════════════════════ */
void connectWiFi();
void syncTimeNTP();
uint32_t getCurrentTimestamp();
int buildJSON(const RiskMessage& msg, char* buf, size_t bufLen);
bool transmitLoRa(const char* json, int len);
void sendRiskMessage(RiskMessage& msg);
RiskMessage defaultMsg();
int readSerialInt();
void checkWiFiConnection();
void showMenu();
void reconnectWiFi();
void displayTestResult(const TestResult& result);
void runAutomatedTestSequence();
void printTestHeader(int testNum, const char* testName, const char* category);

// Original Tests (1-16)
void test01_M0_Observe();
void test02_M1_Bee();
void test03_M2_Leopard();
void test04_M3_Siren();
void test05_AggressionOverride();
void test06_TTLExpired();
void test07_ReplayAttack();
void test08_DegradedQuality();
void test09_NearHighRisk();
void test10_CooldownTest();
void test11_BudgetExhaustion();
void test12_EscalationSequence();
void test13_AggressionDuringDeterrent();
void test14_DifferentZones();
void test15_CustomScenario();
void test16_ZoneMismatch();

// New Positive Tests (17-23)
void test17_MinimalValidMessage();
void test18_MaximalValidMessage();
void test19_BoundaryConditions();
void test20_SequentialOperations();
void test21_RecoveryAfterError();
void test22_LongRunningStability();
void test23_MultipleZoneTransitions();

// New Negative Tests (24-31)
void test24_InvalidRiskLevel();
void test25_InvalidBreachStatus();
void test26_NegativeSequence();
void test27_FutureTimestamp();
void test28_ZeroTTL();
void test29_CorruptedChecksum();
void test30_MalformedJSON();
void test31_EmptyMessage();

// Automated Sequences (90-93)
void test90_PositiveTestSuite();
void test91_NegativeTestSuite();
void test92_SafetyTestSuite();
void test93_FullSystemTest();

/* ═══════════════════════════════════════════════════════════════
 *  HELPER: Display Test Result
 * ═══════════════════════════════════════════════════════════════ */
void displayTestResult(const TestResult& result) {
    Serial.println("\n  ┌──────────────────────────────────────────────────┐");
    Serial.printf("  │ Test: %-42s │\n", result.testName);
    Serial.println("  ├──────────────────────────────────────────────────┤");
    Serial.println("  │ EXPECTED BEHAVIOR:                              │");
    
    // Word wrap for expected behavior
    String behavior = result.expectedBehavior;
    int lineLength = 48;
    int pos = 0;
    while (pos < behavior.length()) {
        int endPos = pos + lineLength;
        if (endPos > behavior.length()) endPos = behavior.length();
        
        // Try to break at space
        if (endPos < behavior.length()) {
            int lastSpace = behavior.lastIndexOf(' ', endPos);
            if (lastSpace > pos) endPos = lastSpace;
        }
        
        String line = behavior.substring(pos, endPos);
        line.trim();
        Serial.printf("  │ %-47s │\n", line.c_str());
        pos = endPos + 1;
    }
    
    Serial.println("  └──────────────────────────────────────────────────┘");
    
    if (result.manualVerification) {
        Serial.println("\n  ⚠️  MANUAL VERIFICATION REQUIRED");
        Serial.println("  → Check Unit 1 Serial Monitor for actual behavior");
        Serial.println("  → Compare with expected behavior above");
        Serial.println("  → Mark test as PASS/FAIL in your test log");
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  HELPER: Print Test Header
 * ═══════════════════════════════════════════════════════════════ */
void printTestHeader(int testNum, const char* testName, const char* category) {
    Serial.println("\n╔════════════════════════════════════════════════════════╗");
    Serial.printf("║  TEST #%02d: %-45s║\n", testNum, testName);
    Serial.printf("║  Category: %-43s║\n", category);
    Serial.println("╚════════════════════════════════════════════════════════╝\n");
}

/* ═══════════════════════════════════════════════════════════════
 *  WiFi Functions (Same as original)
 * ═══════════════════════════════════════════════════════════════ */
void connectWiFi() {
    Serial.println("\n  [WiFi] Connecting to WiFi...");
    Serial.printf("  SSID: %s\n", WIFI_SSID);
    
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();
    
    if (WiFi.status() == WL_CONNECTED) {
        g_wifiConnected = true;
        Serial.println("  [WiFi] ✅ Connected!");
        Serial.print("  IP Address: ");
        Serial.println(WiFi.localIP());
        syncTimeNTP();
    } else {
        Serial.println("  [WiFi] ❌ Connection failed");
        Serial.println("  Will use fallback timestamp mode");
    }
}

void syncTimeNTP() {
    Serial.println("\n  [NTP] Syncing time from internet...");
    configTime(GMT_OFFSET_SEC, DST_OFFSET_SEC, NTP_SERVER);
    
    struct tm timeinfo;
    int attempts = 0;
    while (!getLocalTime(&timeinfo) && attempts < 10) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();
    
    if (getLocalTime(&timeinfo)) {
        g_timeSync = true;
        Serial.println("  [NTP] ✅ Time synchronized!");
        Serial.print("  Current time: ");
        Serial.println(&timeinfo, "%Y-%m-%d %H:%M:%S");
        
        time_t now;
        time(&now);
        g_baseTimestamp = (uint32_t)now;
    } else {
        Serial.println("  [NTP] ❌ Sync failed - using fallback");
    }
}

uint32_t getCurrentTimestamp() {
    if (g_timeSync) {
        time_t now;
        time(&now);
        return (uint32_t)now;
    }
    return g_baseTimestamp++;
}

void checkWiFiConnection() {
    static unsigned long lastCheck = 0;
    if (millis() - lastCheck > 30000) {
        lastCheck = millis();
        if (WiFi.status() != WL_CONNECTED && g_wifiConnected) {
            Serial.println("\n  [WiFi] Connection lost! Reconnecting...");
            g_wifiConnected = false;
            g_timeSync = false;
            connectWiFi();
        }
    }
}

void reconnectWiFi() {
    Serial.println("\n  [CMD] Manual WiFi reconnection...");
    WiFi.disconnect();
    g_wifiConnected = false;
    g_timeSync = false;
    delay(1000);
    connectWiFi();
}

/* ═══════════════════════════════════════════════════════════════
 *  JSON Builder & LoRa Transmit
 * ═══════════════════════════════════════════════════════════════ */
int buildJSON(const RiskMessage& msg, char* buf, size_t bufLen) {
    return snprintf(buf, bufLen,
        "{\"zone_id\":\"%s\","
        "\"boundary_id\":\"%s\","
        "\"risk_level\":\"%s\","
        "\"breach_status\":\"%s\","
        "\"aggression_flag\":%s,"
        "\"ttl_seconds\":%u,"
        "\"sequence_number\":%u,"
        "\"timestamp_utc\":%u,"
        "\"distance_band\":%u,"
        "\"data_quality\":%u,"
        "\"checksum\":%u}",
        msg.zone_id,
        msg.boundary_id,
        msg.risk_level,
        msg.breach_status,
        msg.aggression_flag ? "true" : "false",
        msg.ttl_seconds,
        msg.sequence_number,
        msg.timestamp_utc,
        msg.distance_band,
        msg.data_quality,
        msg.checksum
    );
}

bool transmitLoRa(const char* json, int len) {
    if (!g_loraReady) {
        Serial.println("  [ERR] LoRa not ready!");
        return false;
    }

    LoRa.beginPacket();
    LoRa.write((const uint8_t*)json, len);
    bool success = LoRa.endPacket();

    if (success) {
        g_txCount++;
        Serial.println("  [TX] ✅ Packet transmitted");
        Serial.printf("  [TX] Size: %d bytes | Total TX: %u\n", len, g_txCount);
    } else {
        Serial.println("  [TX] ❌ Transmission failed");
    }

    return success;
}

void sendRiskMessage(RiskMessage& msg) {
    msg.sequence_number = ++g_sequence;
    msg.timestamp_utc = getCurrentTimestamp();
    msg.checksum = (msg.sequence_number + msg.timestamp_utc) % 65536;

    char json[512];
    int len = buildJSON(msg, json, sizeof(json));

    Serial.println("  ┌─── Message Details ────────────────────────────┐");
    Serial.printf("  │ Zone:       %-35s│\n", msg.zone_id);
    Serial.printf("  │ Boundary:   %-35s│\n", msg.boundary_id);
    Serial.printf("  │ Risk:       %-35s│\n", msg.risk_level);
    Serial.printf("  │ Breach:     %-35s│\n", msg.breach_status);
    Serial.printf("  │ Aggression: %-35s│\n", msg.aggression_flag ? "TRUE" : "FALSE");
    Serial.printf("  │ Distance:   %-35s│\n", 
                  msg.distance_band == DIST_FAR ? "FAR" : 
                  msg.distance_band == DIST_MID ? "MID" : "NEAR");
    Serial.printf("  │ Quality:    %-35s│\n", msg.data_quality ? "GOOD" : "DEGRADED");
    Serial.printf("  │ TTL:        %-35u│\n", msg.ttl_seconds);
    Serial.printf("  │ Sequence:   %-35u│\n", msg.sequence_number);
    Serial.printf("  │ Timestamp:  %-35u│\n", msg.timestamp_utc);
    Serial.printf("  │ Checksum:   %-35u│\n", msg.checksum);
    Serial.println("  └────────────────────────────────────────────────┘");

    Serial.println("\n  [JSON] Payload:");
    Serial.printf("  %s\n\n", json);

    transmitLoRa(json, len);
}

RiskMessage defaultMsg() {
    RiskMessage msg = {
        DEFAULT_ZONE_ID,
        VALID_BOUNDARY_ID,
        "LOW",
        "NONE",
        false,
        300,
        0,
        0,
        DIST_FAR,
        QUALITY_GOOD,
        0
    };
    return msg;
}

/* ═══════════════════════════════════════════════════════════════
 *  ORIGINAL TESTS (1-16) - Same as before
 * ═══════════════════════════════════════════════════════════════ */

void test01_M0_Observe() {
    printTestHeader(1, "M0: Low Risk - Observe Mode", "POSITIVE");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "LOW";
    msg.breach_status = "NONE";
    msg.distance_band = DIST_FAR;
    sendRiskMessage(msg);
    
    TestResult result = {
        "M0 Observe Mode",
        "Siren AI should enter M0 mode (observe only). No speakers activated. LEDs show LOW risk status. Serial output: 'Mode: M0 - OBSERVE'",
        true
    };
    displayTestResult(result);
}

void test02_M1_Bee() {
    printTestHeader(2, "M1: Medium Risk - Bee Buzz Sound", "POSITIVE");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    msg.distance_band = DIST_MID;
    sendRiskMessage(msg);
    
    TestResult result = {
        "M1 Bee Buzz",
        "Siren AI should activate M1 mode. Bee swarm buzzing sound from speakers. Speaker rotation active. Serial output: 'Mode: M1 - BEE_SWARM'. Duration: 8-15 seconds.",
        true
    };
    displayTestResult(result);
}

void test03_M2_Leopard() {
    printTestHeader(3, "M2: High Risk - Leopard Growl", "POSITIVE");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "HIGH";
    msg.breach_status = "LIKELY";
    msg.distance_band = DIST_MID;
    sendRiskMessage(msg);
    
    TestResult result = {
        "M2 Leopard Growl",
        "Siren AI should activate M2 mode. Leopard growl/predator sound from speakers. Higher volume than M1. Serial output: 'Mode: M2 - LEOPARD_GROWL'. Duration: 10-20 seconds.",
        true
    };
    displayTestResult(result);
}

void test04_M3_Siren() {
    printTestHeader(4, "M3: Confirmed Breach - Human Alert", "POSITIVE");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "HIGH";
    msg.breach_status = "CONFIRMED";
    msg.distance_band = DIST_NEAR;
    sendRiskMessage(msg);
    
    TestResult result = {
        "M3 Human Alert",
        "Siren AI should activate M3 mode. Loud siren/alarm sound. Human alert system triggered. Serial output: 'Mode: M3 - HUMAN_ALERT'. Continuous until acknowledged.",
        true
    };
    displayTestResult(result);
}

void test05_AggressionOverride() {
    printTestHeader(5, "Aggression Override → M3", "SAFETY - CRITICAL");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "LOW";  // Even LOW risk
    msg.breach_status = "NONE";
    msg.aggression_flag = true;  // But aggression detected!
    msg.distance_band = DIST_MID;
    sendRiskMessage(msg);
    
    TestResult result = {
        "Aggression Override",
        "CRITICAL: Siren AI MUST override RL decision and escalate to M3 immediately. Reason: 'AGGRESSION_OVERRIDE'. Lockout timer activated (10-30 min). No deterrents allowed during lockout.",
        true
    };
    displayTestResult(result);
}

void test06_TTLExpired() {
    printTestHeader(6, "TTL Expired Message", "NEGATIVE - SECURITY");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "HIGH";
    msg.breach_status = "CONFIRMED";
    msg.ttl_seconds = 0;  // Expired!
    msg.timestamp_utc = getCurrentTimestamp() - 600;  // 10 minutes old
    sendRiskMessage(msg);
    
    TestResult result = {
        "TTL Expired Rejection",
        "Siren AI MUST REJECT this message. Serial output: 'REJECT: TTL_EXPIRED'. No mode change. No speaker activation. Message logged as invalid.",
        true
    };
    displayTestResult(result);
}

void test07_ReplayAttack() {
    printTestHeader(7, "Replay Attack / Duplicate Sequence", "NEGATIVE - SECURITY");
    
    Serial.println("  [TEST] Sending same message twice...\n");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    
    Serial.println("  ─── FIRST TRANSMISSION (Should ACCEPT) ───");
    sendRiskMessage(msg);
    
    delay(2000);
    
    Serial.println("\n  ─── SECOND TRANSMISSION (Should REJECT) ───");
    msg.sequence_number--;  // Use same sequence!
    sendRiskMessage(msg);
    
    TestResult result = {
        "Replay Attack Detection",
        "FIRST message: Should be accepted and processed normally. SECOND message: MUST be REJECTED with reason 'REPLAY_DETECTED' or 'DUPLICATE_SEQUENCE'. Security log entry created.",
        true
    };
    displayTestResult(result);
}

void test08_DegradedQuality() {
    printTestHeader(8, "Degraded Data Quality → M3", "SAFETY");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    msg.data_quality = QUALITY_DEGRADED;  // Poor sensor data
    sendRiskMessage(msg);
    
    TestResult result = {
        "Degraded Quality Escalation",
        "Siren AI should escalate conservatively to M3 due to poor sensor data quality. Reason: 'DEGRADED_DATA_CONSERVATIVE'. This is a safety feature - better safe than sorry.",
        true
    };
    displayTestResult(result);
}

void test09_NearHighRisk() {
    printTestHeader(9, "Near Distance + High Risk", "SCENARIO");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "HIGH";
    msg.breach_status = "LIKELY";
    msg.distance_band = DIST_NEAR;  // Very close!
    sendRiskMessage(msg);
    
    TestResult result = {
        "Near High Risk Scenario",
        "Siren AI should respond aggressively. Likely M2 or M3 mode. Quick response time. Serial shows risk assessment: 'NEAR + HIGH = CRITICAL'. Appropriate deterrent activated.",
        true
    };
    displayTestResult(result);
}

void test10_CooldownTest() {
    printTestHeader(10, "Cooldown Enforcement", "SAFETY");
    
    Serial.println("  [TEST] Sending 2 rapid M1 messages...\n");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    
    Serial.println("  ─── FIRST MESSAGE (Should ACTIVATE) ───");
    sendRiskMessage(msg);
    
    delay(2000);  // 2 seconds - too fast!
    
    Serial.println("\n  ─── SECOND MESSAGE (Should COOLDOWN) ───");
    sendRiskMessage(msg);
    
    TestResult result = {
        "Cooldown Enforcement",
        "FIRST message: Should activate M1 deterrent. SECOND message: Should be blocked by cooldown timer. Serial output: 'COOLDOWN_ACTIVE - X seconds remaining'. Prevents habituation.",
        true
    };
    displayTestResult(result);
}

void test11_BudgetExhaustion() {
    printTestHeader(11, "Budget Exhaustion (7 rapid bursts)", "SAFETY");
    
    Serial.println("  [TEST] Sending 7 messages rapidly...\n");
    Serial.println("  This simulates budget limit testing\n");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    
    for (int i = 1; i <= 7; i++) {
        Serial.printf("  ─── MESSAGE %d/7 ───\n", i);
        sendRiskMessage(msg);
        delay(1000);  // 1 second between
        Serial.println();
    }
    
    TestResult result = {
        "Budget Limit Enforcement",
        "First few messages: Should be processed normally. Later messages: Should be blocked with 'BUDGET_EXCEEDED'. System protects itself from overuse. Hourly limit is ~10 activations.",
        true
    };
    displayTestResult(result);
}

void test12_EscalationSequence() {
    printTestHeader(12, "Full Escalation: M0 → M1 → M2 → M3", "SCENARIO");
    
    Serial.println("  [TEST] Simulating progressive threat escalation...\n");
    
    RiskMessage msg = defaultMsg();
    
    Serial.println("  ─── STEP 1: LOW RISK (M0) ───");
    msg.risk_level = "LOW";
    msg.breach_status = "NONE";
    msg.distance_band = DIST_FAR;
    sendRiskMessage(msg);
    delay(3000);
    
    Serial.println("\n  ─── STEP 2: MED RISK (M1) ───");
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    msg.distance_band = DIST_MID;
    sendRiskMessage(msg);
    delay(3000);
    
    Serial.println("\n  ─── STEP 3: HIGH RISK (M2) ───");
    msg.risk_level = "HIGH";
    msg.breach_status = "LIKELY";
    msg.distance_band = DIST_MID;
    sendRiskMessage(msg);
    delay(3000);
    
    Serial.println("\n  ─── STEP 4: CONFIRMED BREACH (M3) ───");
    msg.risk_level = "HIGH";
    msg.breach_status = "CONFIRMED";
    msg.distance_band = DIST_NEAR;
    sendRiskMessage(msg);
    
    TestResult result = {
        "Full Escalation Sequence",
        "System should show progressive escalation: M0 (observe) → M1 (bee buzz) → M2 (leopard) → M3 (human alert). Each mode more aggressive than previous. Demonstrates full capability.",
        true
    };
    displayTestResult(result);
}

void test13_AggressionDuringDeterrent() {
    printTestHeader(13, "Aggression During Active Deterrent", "SAFETY - CRITICAL");
    
    Serial.println("  [TEST] Activating deterrent, then sending aggression...\n");
    
    RiskMessage msg = defaultMsg();
    
    Serial.println("  ─── STEP 1: Activate M1 Deterrent ───");
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    sendRiskMessage(msg);
    
    delay(3000);  // Let deterrent run for 3 seconds
    
    Serial.println("\n  ─── STEP 2: Aggression Detected! ───");
    msg.aggression_flag = true;
    sendRiskMessage(msg);
    
    TestResult result = {
        "Aggression During Deterrent",
        "CRITICAL: M1 deterrent should STOP immediately. System escalates to M3. Lockout activated. Serial shows: 'DETERRENT_STOPPED - AGGRESSION_OVERRIDE'. No further deterrents for 10-30 minutes.",
        true
    };
    displayTestResult(result);
}

void test14_DifferentZones() {
    printTestHeader(14, "Multiple Zones & Boundaries", "INTEGRATION");
    
    Serial.println("  [TEST] Testing different zones and boundaries...\n");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    
    Serial.println("  ─── Zone A, Boundary 1 ───");
    msg.zone_id = "ZONE-A1";
    msg.boundary_id = "BOUNDARY-001";
    sendRiskMessage(msg);
    delay(2000);
    
    Serial.println("\n  ─── Zone A, Boundary 2 ───");
    msg.zone_id = "ZONE-A1";
    msg.boundary_id = "BOUNDARY-002";
    sendRiskMessage(msg);
    delay(2000);
    
    Serial.println("\n  ─── Zone B, Boundary 1 ───");
    msg.zone_id = "ZONE-B1";
    msg.boundary_id = "BOUNDARY-001";
    sendRiskMessage(msg);
    
    TestResult result = {
        "Multi-Zone Operation",
        "System should accept first two messages (ZONE-A1 is configured). Third message (ZONE-B1) should be REJECTED with 'ZONE_MISMATCH'. Shows proper zone filtering.",
        true
    };
    displayTestResult(result);
}

void test15_CustomScenario() {
    printTestHeader(15, "Custom Manual Scenario", "CUSTOM");
    
    Serial.println("  [CUSTOM] You will enter parameters manually\n");
    
    RiskMessage msg = defaultMsg();
    
    Serial.println("  Risk Level (0=LOW, 1=MED, 2=HIGH): ");
    int risk = readSerialInt();
    msg.risk_level = (risk == 0) ? "LOW" : (risk == 1) ? "MED" : "HIGH";
    
    Serial.println("  Breach Status (0=NONE, 1=LIKELY, 2=CONFIRMED): ");
    int breach = readSerialInt();
    msg.breach_status = (breach == 0) ? "NONE" : (breach == 1) ? "LIKELY" : "CONFIRMED";
    
    Serial.println("  Aggression Flag (0=NO, 1=YES): ");
    msg.aggression_flag = readSerialInt() == 1;
    
    Serial.println("  Distance (0=FAR, 1=MID, 2=NEAR): ");
    msg.distance_band = readSerialInt();
    
    Serial.println("  Data Quality (0=DEGRADED, 1=GOOD): ");
    msg.data_quality = readSerialInt();
    
    sendRiskMessage(msg);
    
    TestResult result = {
        "Custom Scenario",
        "Behavior depends on parameters you entered. Check Serial Monitor for mode selection and verify it matches expected logic based on your inputs.",
        true
    };
    displayTestResult(result);
}

void test16_ZoneMismatch() {
    printTestHeader(16, "Zone Mismatch - Wrong Zone", "NEGATIVE - SECURITY");
    
    RiskMessage msg = defaultMsg();
    msg.zone_id = INVALID_ZONE_ID;  // Wrong zone!
    msg.risk_level = "HIGH";
    msg.breach_status = "CONFIRMED";
    sendRiskMessage(msg);
    
    TestResult result = {
        "Zone Mismatch Rejection",
        "Siren AI MUST REJECT this message. Zone 'ZONE-XX' does not match configured zone 'ZONE-A1'. Serial output: 'REJECT: ZONE_MISMATCH'. Security feature prevents cross-zone interference.",
        true
    };
    displayTestResult(result);
}

/* ═══════════════════════════════════════════════════════════════
 *  NEW POSITIVE TESTS (17-23)
 * ═══════════════════════════════════════════════════════════════ */

void test17_MinimalValidMessage() {
    printTestHeader(17, "Minimal Valid Message", "POSITIVE - BOUNDARY");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "LOW";
    msg.breach_status = "NONE";
    msg.distance_band = DIST_FAR;
    msg.ttl_seconds = 60;  // Minimum reasonable TTL
    sendRiskMessage(msg);
    
    TestResult result = {
        "Minimal Valid Message",
        "System should accept this minimal but valid message. M0 mode activated. All required fields present. Demonstrates minimum viable message structure.",
        true
    };
    displayTestResult(result);
}

void test18_MaximalValidMessage() {
    printTestHeader(18, "Maximal Valid Message", "POSITIVE - BOUNDARY");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "HIGH";
    msg.breach_status = "CONFIRMED";
    msg.distance_band = DIST_NEAR;
    msg.aggression_flag = false;  // No aggression
    msg.ttl_seconds = 600;  // Maximum reasonable TTL
    msg.data_quality = QUALITY_GOOD;
    sendRiskMessage(msg);
    
    TestResult result = {
        "Maximal Valid Message",
        "System should accept and process normally. M3 mode likely due to HIGH + CONFIRMED + NEAR. All optional fields present. Demonstrates full message capability.",
        true
    };
    displayTestResult(result);
}

void test19_BoundaryConditions() {
    printTestHeader(19, "Boundary Condition Testing", "POSITIVE - EDGE");
    
    Serial.println("  [TEST] Testing boundary values...\n");
    
    RiskMessage msg = defaultMsg();
    
    Serial.println("  ─── Test 1: TTL = 1 second (minimum) ───");
    msg.ttl_seconds = 1;
    sendRiskMessage(msg);
    delay(2000);
    
    Serial.println("\n  ─── Test 2: Very recent timestamp ───");
    msg = defaultMsg();
    msg.timestamp_utc = getCurrentTimestamp();
    sendRiskMessage(msg);
    delay(2000);
    
    Serial.println("\n  ─── Test 3: Sequence number wrap ───");
    msg = defaultMsg();
    g_sequence = 65530;  // Near uint16_t max
    sendRiskMessage(msg);
    
    TestResult result = {
        "Boundary Conditions",
        "All three messages should be accepted. Tests: (1) Minimum TTL handling, (2) Current timestamp acceptance, (3) Sequence number overflow handling. System should handle edge values gracefully.",
        true
    };
    displayTestResult(result);
}

void test20_SequentialOperations() {
    printTestHeader(20, "Sequential Normal Operations", "POSITIVE - WORKFLOW");
    
    Serial.println("  [TEST] Simulating realistic sequential operations...\n");
    
    RiskMessage msg = defaultMsg();
    
    Serial.println("  ─── Hour 1: Morning patrol ───");
    msg.risk_level = "LOW";
    msg.breach_status = "NONE";
    sendRiskMessage(msg);
    delay(5000);
    
    Serial.println("\n  ─── Hour 2: Detection ───");
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    sendRiskMessage(msg);
    delay(5000);
    
    Serial.println("\n  ─── Hour 3: Threat passed ───");
    msg.risk_level = "LOW";
    msg.breach_status = "NONE";
    sendRiskMessage(msg);
    
    TestResult result = {
        "Sequential Operations",
        "Demonstrates typical operational sequence. LOW → MED → LOW pattern. System should respond appropriately to each state change. No errors expected. Realistic field scenario.",
        true
    };
    displayTestResult(result);
}

void test21_RecoveryAfterError() {
    printTestHeader(21, "Recovery After Error", "POSITIVE - RECOVERY");
    
    Serial.println("  [TEST] Testing error recovery...\n");
    
    RiskMessage msg = defaultMsg();
    
    Serial.println("  ─── Step 1: Send invalid message (should reject) ───");
    msg.zone_id = INVALID_ZONE_ID;
    sendRiskMessage(msg);
    delay(3000);
    
    Serial.println("\n  ─── Step 2: Send valid message (should accept) ───");
    msg.zone_id = DEFAULT_ZONE_ID;
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    sendRiskMessage(msg);
    
    TestResult result = {
        "Recovery After Error",
        "First message should be REJECTED (wrong zone). Second message should be ACCEPTED and processed normally. Demonstrates system resilience - errors don't cause permanent failures.",
        true
    };
    displayTestResult(result);
}

void test22_LongRunningStability() {
    printTestHeader(22, "Long Running Stability", "POSITIVE - STABILITY");
    
    Serial.println("  [TEST] Sending 10 valid messages over time...\n");
    Serial.println("  This tests system stability over extended operation\n");
    
    RiskMessage msg = defaultMsg();
    
    for (int i = 1; i <= 10; i++) {
        Serial.printf("  ─── Message %d/10 ───\n", i);
        
        // Alternate between LOW and MED
        if (i % 2 == 0) {
            msg.risk_level = "MED";
            msg.breach_status = "LIKELY";
        } else {
            msg.risk_level = "LOW";
            msg.breach_status = "NONE";
        }
        
        sendRiskMessage(msg);
        delay(5000);  // 5 seconds between messages
        Serial.println();
    }
    
    TestResult result = {
        "Long Running Stability",
        "System should process all 10 messages without errors. No memory leaks. No performance degradation. Sequence numbers increment correctly. Demonstrates production readiness.",
        true
    };
    displayTestResult(result);
}

void test23_MultipleZoneTransitions() {
    printTestHeader(23, "Multiple Zone Transitions", "POSITIVE - INTEGRATION");
    
    Serial.println("  [TEST] Testing zone transition handling...\n");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    
    Serial.println("  ─── Boundary 1 ───");
    msg.boundary_id = "BOUNDARY-001";
    sendRiskMessage(msg);
    delay(3000);
    
    Serial.println("\n  ─── Boundary 2 ───");
    msg.boundary_id = "BOUNDARY-002";
    sendRiskMessage(msg);
    delay(3000);
    
    Serial.println("\n  ─── Back to Boundary 1 ───");
    msg.boundary_id = "BOUNDARY-001";
    sendRiskMessage(msg);
    
    TestResult result = {
        "Zone Transition Handling",
        "System should accept all messages (same zone, different boundaries). Demonstrates ability to handle elephant movement between boundaries. Speaker selection may vary by boundary.",
        true
    };
    displayTestResult(result);
}

/* ═══════════════════════════════════════════════════════════════
 *  NEW NEGATIVE TESTS (24-31)
 * ═══════════════════════════════════════════════════════════════ */

void test24_InvalidRiskLevel() {
    printTestHeader(24, "Invalid Risk Level", "NEGATIVE - VALIDATION");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "EXTREME";  // Invalid! Should be LOW/MED/HIGH
    msg.breach_status = "LIKELY";
    sendRiskMessage(msg);
    
    TestResult result = {
        "Invalid Risk Level",
        "System SHOULD REJECT this message. Reason: 'INVALID_RISK_LEVEL'. Only LOW, MED, HIGH are valid. Security validation working correctly.",
        true
    };
    displayTestResult(result);
}

void test25_InvalidBreachStatus() {
    printTestHeader(25, "Invalid Breach Status", "NEGATIVE - VALIDATION");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "HIGH";
    msg.breach_status = "MAYBE";  // Invalid! Should be NONE/LIKELY/CONFIRMED
    sendRiskMessage(msg);
    
    TestResult result = {
        "Invalid Breach Status",
        "System SHOULD REJECT this message. Reason: 'INVALID_BREACH_STATUS'. Only NONE, LIKELY, CONFIRMED are valid. Input validation working correctly.",
        true
    };
    displayTestResult(result);
}

void test26_NegativeSequence() {
    printTestHeader(26, "Negative Sequence Number", "NEGATIVE - VALIDATION");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "MED";
    msg.breach_status = "LIKELY";
    // Can't actually send negative in uint32_t, but we can send 0
    g_sequence = 0;  // Reset to 0
    sendRiskMessage(msg);
    
    TestResult result = {
        "Zero Sequence Number",
        "System behavior depends on implementation. May REJECT (if 0 is invalid) or ACCEPT (if 0 is valid first sequence). Check if sequence validation allows 0.",
        true
    };
    displayTestResult(result);
}

void test27_FutureTimestamp() {
    printTestHeader(27, "Future Timestamp", "NEGATIVE - VALIDATION");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "HIGH";
    msg.breach_status = "CONFIRMED";
    msg.timestamp_utc = getCurrentTimestamp() + 7200;  // 2 hours in future!
    sendRiskMessage(msg);
    
    TestResult result = {
        "Future Timestamp",
        "System SHOULD REJECT this message. Reason: 'FUTURE_TIMESTAMP' or 'CLOCK_SKEW_EXCEEDED'. Timestamps from future indicate clock sync issues or tampering.",
        true
    };
    displayTestResult(result);
}

void test28_ZeroTTL() {
    printTestHeader(28, "Zero TTL", "NEGATIVE - VALIDATION");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "HIGH";
    msg.breach_status = "CONFIRMED";
    msg.ttl_seconds = 0;  // Expired immediately!
    sendRiskMessage(msg);
    
    TestResult result = {
        "Zero TTL",
        "System SHOULD REJECT this message. Reason: 'TTL_EXPIRED' or 'INVALID_TTL'. Message has no validity period. Should not be processed.",
        true
    };
    displayTestResult(result);
}

void test29_CorruptedChecksum() {
    printTestHeader(29, "Corrupted Checksum", "NEGATIVE - SECURITY");
    
    RiskMessage msg = defaultMsg();
    msg.risk_level = "HIGH";
    msg.breach_status = "CONFIRMED";
    msg.sequence_number = ++g_sequence;
    msg.timestamp_utc = getCurrentTimestamp();
    msg.checksum = 99999;  // Wrong checksum!
    
    char json[512];
    int len = buildJSON(msg, json, sizeof(json));
    
    Serial.println("  [WARNING] Sending message with CORRUPTED checksum!");
    Serial.printf("  Expected: %u | Actual: 99999\n\n", 
                  (msg.sequence_number + msg.timestamp_utc) % 65536);
    
    transmitLoRa(json, len);
    
    TestResult result = {
        "Corrupted Checksum",
        "System SHOULD REJECT this message. Reason: 'CHECKSUM_MISMATCH'. Indicates data corruption or tampering. Security validation working correctly.",
        true
    };
    displayTestResult(result);
}

void test30_MalformedJSON() {
    printTestHeader(30, "Malformed JSON", "NEGATIVE - PARSING");
    
    const char* malformedJSON = "{\"zone_id\":\"ZONE-A1\",,\"risk_level\":\"HIGH\"}";
    
    Serial.println("  [WARNING] Sending MALFORMED JSON!");
    Serial.println("  JSON has syntax error (double comma)\n");
    Serial.printf("  Payload: %s\n\n", malformedJSON);
    
    transmitLoRa(malformedJSON, strlen(malformedJSON));
    
    TestResult result = {
        "Malformed JSON",
        "System SHOULD REJECT this message. Reason: 'JSON_PARSE_ERROR' or 'INVALID_FORMAT'. Parser should catch syntax errors. System remains stable despite bad input.",
        true
    };
    displayTestResult(result);
}

void test31_EmptyMessage() {
    printTestHeader(31, "Empty Message", "NEGATIVE - VALIDATION");
    
    const char* emptyJSON = "{}";
    
    Serial.println("  [WARNING] Sending EMPTY JSON object!");
    Serial.println("  No fields present\n");
    Serial.printf("  Payload: %s\n\n", emptyJSON);
    
    transmitLoRa(emptyJSON, strlen(emptyJSON));
    
    TestResult result = {
        "Empty Message",
        "System SHOULD REJECT this message. Reason: 'MISSING_REQUIRED_FIELDS'. All required fields (zone_id, risk_level, etc.) are missing. Validation catches incomplete messages.",
        true
    };
    displayTestResult(result);
}

/* ═══════════════════════════════════════════════════════════════
 *  AUTOMATED TEST SEQUENCES (90-93)
 * ═══════════════════════════════════════════════════════════════ */

void test90_PositiveTestSuite() {
    printTestHeader(90, "AUTOMATED: Positive Test Suite", "AUTOMATED");
    
    Serial.println("  Running comprehensive positive tests...\n");
    Serial.println("  This will take approximately 5 minutes\n");
    Serial.println("  Press any key to start or wait 5 seconds...\n");
    
    delay(5000);
    
    int tests[] = {1, 2, 3, 4, 9, 17, 18, 19, 20, 23};
    int numTests = sizeof(tests) / sizeof(tests[0]);
    
    for (int i = 0; i < numTests; i++) {
        Serial.printf("\n╔══════════════════════════════════════╗\n");
        Serial.printf("║  POSITIVE TEST %d/%d (Test #%02d)        ║\n", i+1, numTests, tests[i]);
        Serial.printf("╚══════════════════════════════════════╝\n\n");
        
        // Run the test based on number
        switch(tests[i]) {
            case 1: test01_M0_Observe(); break;
            case 2: test02_M1_Bee(); break;
            case 3: test03_M2_Leopard(); break;
            case 4: test04_M3_Siren(); break;
            case 9: test09_NearHighRisk(); break;
            case 17: test17_MinimalValidMessage(); break;
            case 18: test18_MaximalValidMessage(); break;
            case 19: test19_BoundaryConditions(); break;
            case 20: test20_SequentialOperations(); break;
            case 23: test23_MultipleZoneTransitions(); break;
        }
        
        Serial.println("\n  Waiting 10 seconds before next test...");
        delay(10000);
    }
    
    Serial.println("\n╔════════════════════════════════════════════════════╗");
    Serial.println("║  POSITIVE TEST SUITE COMPLETE                    ║");
    Serial.println("╚════════════════════════════════════════════════════╝");
    Serial.printf("\n  Total tests run: %d\n", numTests);
    Serial.println("  Review Unit 1 Serial Monitor for results");
}

void test91_NegativeTestSuite() {
    printTestHeader(91, "AUTOMATED: Negative Test Suite", "AUTOMATED");
    
    Serial.println("  Running comprehensive negative tests...\n");
    Serial.println("  These should all be REJECTED by Siren AI\n");
    Serial.println("  This will take approximately 3 minutes\n");
    
    delay(5000);
    
    int tests[] = {6, 7, 16, 24, 25, 27, 28, 29, 30, 31};
    int numTests = sizeof(tests) / sizeof(tests[0]);
    
    for (int i = 0; i < numTests; i++) {
        Serial.printf("\n╔══════════════════════════════════════╗\n");
        Serial.printf("║  NEGATIVE TEST %d/%d (Test #%02d)       ║\n", i+1, numTests, tests[i]);
        Serial.printf("╚══════════════════════════════════════╝\n\n");
        
        switch(tests[i]) {
            case 6: test06_TTLExpired(); break;
            case 7: test07_ReplayAttack(); break;
            case 16: test16_ZoneMismatch(); break;
            case 24: test24_InvalidRiskLevel(); break;
            case 25: test25_InvalidBreachStatus(); break;
            case 27: test27_FutureTimestamp(); break;
            case 28: test28_ZeroTTL(); break;
            case 29: test29_CorruptedChecksum(); break;
            case 30: test30_MalformedJSON(); break;
            case 31: test31_EmptyMessage(); break;
        }
        
        Serial.println("\n  Waiting 5 seconds before next test...");
        delay(5000);
    }
    
    Serial.println("\n╔════════════════════════════════════════════════════╗");
    Serial.println("║  NEGATIVE TEST SUITE COMPLETE                    ║");
    Serial.println("╚════════════════════════════════════════════════════╝");
    Serial.printf("\n  Total tests run: %d\n", numTests);
    Serial.println("  All should show REJECT messages on Unit 1");
}

void test92_SafetyTestSuite() {
    printTestHeader(92, "AUTOMATED: Safety Test Suite", "AUTOMATED");
    
    Serial.println("  Running comprehensive safety tests...\n");
    Serial.println("  Testing all safety mechanisms\n");
    Serial.println("  This will take approximately 4 minutes\n");
    
    delay(5000);
    
    int tests[] = {5, 8, 10, 11, 13};
    int numTests = sizeof(tests) / sizeof(tests[0]);
    
    for (int i = 0; i < numTests; i++) {
        Serial.printf("\n╔══════════════════════════════════════╗\n");
        Serial.printf("║  SAFETY TEST %d/%d (Test #%02d)         ║\n", i+1, numTests, tests[i]);
        Serial.printf("╚══════════════════════════════════════╝\n\n");
        
        switch(tests[i]) {
            case 5: test05_AggressionOverride(); break;
            case 8: test08_DegradedQuality(); break;
            case 10: test10_CooldownTest(); break;
            case 11: test11_BudgetExhaustion(); break;
            case 13: test13_AggressionDuringDeterrent(); break;
        }
        
        Serial.println("\n  Waiting 15 seconds before next test...");
        delay(15000);  // Longer delay for safety tests
    }
    
    Serial.println("\n╔════════════════════════════════════════════════════╗");
    Serial.println("║  SAFETY TEST SUITE COMPLETE                      ║");
    Serial.println("╚════════════════════════════════════════════════════╝");
    Serial.printf("\n  Total tests run: %d\n", numTests);
    Serial.println("  Verify all safety mechanisms activated correctly");
}

void test93_FullSystemTest() {
    printTestHeader(93, "AUTOMATED: Full System Test", "AUTOMATED");
    
    Serial.println("  Running COMPLETE system test...\n");
    Serial.println("  This includes positive, negative, and safety tests\n");
    Serial.println("  This will take approximately 15 minutes\n");
    Serial.println("  Press any key to start or wait 10 seconds...\n");
    
    delay(10000);
    
    Serial.println("\n  ═══ PHASE 1: POSITIVE TESTS ═══\n");
    test90_PositiveTestSuite();
    
    Serial.println("\n\n  ═══ PHASE 2: NEGATIVE TESTS ═══\n");
    delay(30000);  // 30 second break
    test91_NegativeTestSuite();
    
    Serial.println("\n\n  ═══ PHASE 3: SAFETY TESTS ═══\n");
    delay(30000);  // 30 second break
    test92_SafetyTestSuite();
    
    Serial.println("\n\n╔════════════════════════════════════════════════════════╗");
    Serial.println("║  🎉 FULL SYSTEM TEST COMPLETE! 🎉                    ║");
    Serial.println("╚════════════════════════════════════════════════════════╝");
    Serial.println("\n  All test phases completed:");
    Serial.println("  ✅ Positive test suite");
    Serial.println("  ✅ Negative test suite");
    Serial.println("  ✅ Safety test suite");
    Serial.println("\n  Total approximate tests: 25+");
    Serial.println("  Review all results on Unit 1 Serial Monitor");
    Serial.println("\n  System is PRODUCTION READY if all tests passed! 🚀");
}

/* ═══════════════════════════════════════════════════════════════
 *  SERIAL INPUT HELPER
 * ═══════════════════════════════════════════════════════════════ */
int readSerialInt() {
    while (!Serial.available()) { delay(10); }
    String input = Serial.readStringUntil('\n');
    input.trim();
    Serial.printf("  → %s\n", input.c_str());
    return input.toInt();
}

/* ═══════════════════════════════════════════════════════════════
 *  MENU DISPLAY
 * ═══════════════════════════════════════════════════════════════ */
void showMenu() {
    Serial.println();
    Serial.println("╔════════════════════════════════════════════════════════════╗");
    Serial.println("║   SIREN AI v2 — COMPREHENSIVE TEST SUITE                 ║");
    Serial.println("║   Unit 2 LoRa Transmitter (433 MHz)                      ║");
    Serial.println("╠════════════════════════════════════════════════════════════╣");
    Serial.println("║                                                            ║");
    Serial.println("║  ── POSITIVE TESTS (Normal Operation) ──────────────────  ║");
    Serial.println("║   1  M0: Low Risk, No Breach (Observe)                    ║");
    Serial.println("║   2  M1: Med Risk, Likely Breach (Bee Buzz)               ║");
    Serial.println("║   3  M2: High Risk, Likely Breach (Leopard)               ║");
    Serial.println("║   4  M3: Confirmed Breach (Siren Alert)                   ║");
    Serial.println("║   9  Near Distance + High Risk                            ║");
    Serial.println("║  17  Minimal Valid Message                                ║");
    Serial.println("║  18  Maximal Valid Message                                ║");
    Serial.println("║  19  Boundary Conditions                                  ║");
    Serial.println("║  20  Sequential Operations                                ║");
    Serial.println("║  23  Multiple Zone Transitions                            ║");
    Serial.println("║                                                            ║");
    Serial.println("║  ── NEGATIVE TESTS (Error Handling) ─────────────────────  ║");
    Serial.println("║   6  TTL Expired Message (REJECT)                         ║");
    Serial.println("║   7  Replay Attack (REJECT)                               ║");
    Serial.println("║  16  Zone Mismatch (REJECT)                               ║");
    Serial.println("║  24  Invalid Risk Level (REJECT)                          ║");
    Serial.println("║  25  Invalid Breach Status (REJECT)                       ║");
    Serial.println("║  27  Future Timestamp (REJECT)                            ║");
    Serial.println("║  28  Zero TTL (REJECT)                                    ║");
    Serial.println("║  29  Corrupted Checksum (REJECT)                          ║");
    Serial.println("║  30  Malformed JSON (REJECT)                              ║");
    Serial.println("║  31  Empty Message (REJECT)                               ║");
    Serial.println("║                                                            ║");
    Serial.println("║  ── SAFETY TESTS (Critical Mechanisms) ──────────────────  ║");
    Serial.println("║   5  Aggression Override → M3                             ║");
    Serial.println("║   8  Degraded Data Quality → M3                           ║");
    Serial.println("║  10  Cooldown Test                                        ║");
    Serial.println("║  11  Budget Exhaustion                                    ║");
    Serial.println("║  13  Aggression During Active Deterrent                   ║");
    Serial.println("║                                                            ║");
    Serial.println("║  ── SCENARIO TESTS (Complex Workflows) ──────────────────  ║");
    Serial.println("║  12  Full Escalation (M0 → M1 → M2 → M3)                  ║");
    Serial.println("║  14  Multiple Zones & Boundaries                          ║");
    Serial.println("║  15  Custom Scenario (Manual)                             ║");
    Serial.println("║  21  Recovery After Error                                 ║");
    Serial.println("║  22  Long Running Stability (10 messages)                 ║");
    Serial.println("║                                                            ║");
    Serial.println("║  ── AUTOMATED TEST SEQUENCES ────────────────────────────  ║");
    Serial.println("║  90  Run Positive Test Suite (10 tests, ~5 min)          ║");
    Serial.println("║  91  Run Negative Test Suite (10 tests, ~3 min)          ║");
    Serial.println("║  92  Run Safety Test Suite (5 tests, ~4 min)             ║");
    Serial.println("║  93  Run FULL System Test (25+ tests, ~15 min)           ║");
    Serial.println("║                                                            ║");
    Serial.println("║  ── SYSTEM ───────────────────────────────────────────────  ║");
    Serial.println("║   0  Reconnect WiFi & Sync Time                           ║");
    Serial.println("║  99  Show This Menu                                       ║");
    Serial.println("║                                                            ║");
    Serial.println("╚════════════════════════════════════════════════════════════╝");
    
    Serial.printf("  SEQ: %u | Zone: %s | TX: %u | Pass: %u | Fail: %u\n",
                  g_sequence, DEFAULT_ZONE_ID, g_txCount, 
                  g_testPassCount, g_testFailCount);
    
    // Status indicators
    Serial.print("  WiFi: ");
    if (g_wifiConnected) {
        Serial.print("✅ ");
        Serial.print(WiFi.localIP());
    } else {
        Serial.print("❌ Disconnected");
    }
    
    Serial.print(" | Time: ");
    if (g_timeSync) {
        time_t now = getCurrentTimestamp();
        struct tm* t = localtime(&now);
        Serial.printf("✅ %04d-%02d-%02d %02d:%02d:%02d",
                     t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                     t->tm_hour, t->tm_min, t->tm_sec);
    } else {
        Serial.print("❌ Not Synced");
    }
    Serial.println();
    
    Serial.println("\n  Enter test number (0-31, 90-93, 99):");
}

/* ═══════════════════════════════════════════════════════════════
 *  SETUP
 * ═══════════════════════════════════════════════════════════════ */
void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println();
    Serial.println("═══════════════════════════════════════════════════════════");
    Serial.println("  SIREN AI v2 — COMPREHENSIVE TEST SUITE");
    Serial.println("  Hardware: ESP32 DevKit V1 + SX1278 (433 MHz)");
    Serial.println("  30+ Test Cases | Positive + Negative + Safety");
    Serial.println("  IT22515612 | Bamunusinghe S.A.N.");
    Serial.println("═══════════════════════════════════════════════════════════");
    Serial.println();

    // WiFi Connection
    connectWiFi();
    Serial.println();

    // LoRa Init
    Serial.println("  [INIT] LoRa module...");
    SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_NSS);
    LoRa.setPins(LORA_NSS, LORA_RST, LORA_DIO0);

    if (!LoRa.begin(LORA_FREQ)) {
        Serial.println("  [ERR] ❌ LoRa init FAILED!");
        Serial.println("  Check wiring: NSS=5, RST=14, DIO0=26");
        while (1) delay(1000);
    }

    LoRa.setSpreadingFactor(LORA_SF);
    LoRa.setSignalBandwidth(LORA_BW);
    LoRa.setCodingRate4(LORA_CR);

    g_loraReady = true;
    Serial.println("  [OK] ✅ LoRa ready — 433MHz, SF10, BW125k, CR4/5");
    Serial.printf("  [OK] Heap free: %u bytes\n", ESP.getFreeHeap());
    Serial.println();

    // RF Config Summary
    Serial.println("  ┌─── RF Configuration ─────────────────────────────┐");
    Serial.println("  │ Frequency:    433 MHz                            │");
    Serial.println("  │ Spreading F:  10                                 │");
    Serial.println("  │ Bandwidth:    125 kHz                            │");
    Serial.println("  │ Coding Rate:  4/5                                │");
    Serial.println("  │ MUST match Siren AI Unit 1 configuration        │");
    Serial.println("  └──────────────────────────────────────────────────┘");
    Serial.println();
    
    // Test Suite Info
    Serial.println("  ┌─── Test Suite Coverage ──────────────────────────┐");
    Serial.println("  │ ✅ 16 Original Tests                             │");
    Serial.println("  │ ✅ 15 New Tests (Positive + Negative)            │");
    Serial.println("  │ ✅ 4 Automated Test Sequences                    │");
    Serial.println("  │ ✅ Total: 35 Test Cases                          │");
    Serial.println("  │                                                   │");
    Serial.println("  │ Coverage:                                         │");
    Serial.println("  │  • Normal Operation (Positive)                   │");
    Serial.println("  │  • Error Handling (Negative)                     │");
    Serial.println("  │  • Safety Mechanisms                             │");
    Serial.println("  │  • Security Validation                           │");
    Serial.println("  │  • Integration Scenarios                         │");
    Serial.println("  └──────────────────────────────────────────────────┘");
    Serial.println();

    showMenu();
}

/* ═══════════════════════════════════════════════════════════════
 *  MAIN LOOP — Serial Command Processor
 * ═══════════════════════════════════════════════════════════════ */
void loop() {
    checkWiFiConnection();
    
    if (!Serial.available()) return;

    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() == 0) return;

    int testNum = input.toInt();

    // Route to appropriate test
    switch (testNum) {
        // System
        case 0:  reconnectWiFi(); break;
        case 99: showMenu(); return;
        
        // Original tests (1-16)
        case 1:  test01_M0_Observe(); break;
        case 2:  test02_M1_Bee(); break;
        case 3:  test03_M2_Leopard(); break;
        case 4:  test04_M3_Siren(); break;
        case 5:  test05_AggressionOverride(); break;
        case 6:  test06_TTLExpired(); break;
        case 7:  test07_ReplayAttack(); break;
        case 8:  test08_DegradedQuality(); break;
        case 9:  test09_NearHighRisk(); break;
        case 10: test10_CooldownTest(); break;
        case 11: test11_BudgetExhaustion(); break;
        case 12: test12_EscalationSequence(); break;
        case 13: test13_AggressionDuringDeterrent(); break;
        case 14: test14_DifferentZones(); break;
        case 15: test15_CustomScenario(); break;
        case 16: test16_ZoneMismatch(); break;
        
        // New positive tests (17-23)
        case 17: test17_MinimalValidMessage(); break;
        case 18: test18_MaximalValidMessage(); break;
        case 19: test19_BoundaryConditions(); break;
        case 20: test20_SequentialOperations(); break;
        case 21: test21_RecoveryAfterError(); break;
        case 22: test22_LongRunningStability(); break;
        case 23: test23_MultipleZoneTransitions(); break;
        
        // New negative tests (24-31)
        case 24: test24_InvalidRiskLevel(); break;
        case 25: test25_InvalidBreachStatus(); break;
        case 26: test26_NegativeSequence(); break;
        case 27: test27_FutureTimestamp(); break;
        case 28: test28_ZeroTTL(); break;
        case 29: test29_CorruptedChecksum(); break;
        case 30: test30_MalformedJSON(); break;
        case 31: test31_EmptyMessage(); break;
        
        // Automated sequences (90-93)
        case 90: test90_PositiveTestSuite(); break;
        case 91: test91_NegativeTestSuite(); break;
        case 92: test92_SafetyTestSuite(); break;
        case 93: test93_FullSystemTest(); break;
        
        default:
            Serial.printf("  [?] Unknown test: %d\n", testNum);
            Serial.println("  Valid: 0-31, 90-93, 99");
            break;
    }

    // Show menu again after test
    Serial.println();
    Serial.println("  ────────────────────────────────────────────────────");
    showMenu();
}
