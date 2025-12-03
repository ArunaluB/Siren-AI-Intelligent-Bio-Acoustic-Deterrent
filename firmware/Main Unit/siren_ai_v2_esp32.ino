
/*
╔══════════════════════════════════════════════════════╗
║                  SIREN AI v3                         ║
║        Wildlife 360 Conservative Deterrent           ║
║                                                      ║
║  Author  : Arunalu Bamunusinghe                      ║
║  Message : Namo Buddhaya                             ║
╚══════════════════════════════════════════════════════╝
*/

// =====================================================
// Version       : v3.0
// Hardware      : ESP32-S3-DevKitC-1 N16R8
// Architecture  : Dual-Core FreeRTOS
// Communication : LoRa (SX1278), WiFi
// Audio Output  : MAX98357 (I2S)
// Storage       : SD Card (FSPI)
// =====================================================

// =====================================================
// SYSTEM OVERVIEW
// =====================================================
/*!
 * @file siren_ai_v2_esp32.ino
 * @brief Wildlife 360 Conservative Deterrent Firmware
 *
 * @details
 *  - Platform: ESP32-S3-DevKitC-1 N16R8 (Dual-Core, FreeRTOS)
 *  - Hardware: SX1278 LoRa (433MHz), MAX98357 I2S Amp, 16x2 I2C LCD, DS3231 RTC, MicroSD Card, 4 Ohm 5W Speaker
 *  - Communication: LoRa (HSPI), WiFi (NTP), I2C (LCD/RTC), I2S (Audio), SD Card (FSPI)
 *  - Audio: WAV playback via I2S, thread-safe SD access, fallback tone
 *  - Task Structure: Core 0 (LoRa, LCD, Decision), Core 1 (Audio, Logging)
 *  - FreeRTOS Queues: Audio and Log command queues for inter-core comms
 *  - Safety: Mutex-protected SD, strict bus separation, robust error handling
 *
 *  Communication Flow:
 *    [LoRa RX] → [Parse/Validate] → [Decision] → [Audio/Log Task] → [I2S/SD]
 *
 *  Audio Playback Logic:
 *    - WAV files selected per mode, parsed for correct chunk, played via I2S
 *    - Fallback tone if file missing or error
 *
 *  FreeRTOS Task Structure:
 *    - Core 0: Main loop, LoRa RX, LCD, decision logic
 *    - Core 1: Audio playback, SD logging (audioTask)
 *
 *  Stability & Safety:
 *    - SPI bus separation (FSPI/HSPI)
 *    - SD card mutex for thread safety
 *    - All hardware and timing logic preserved
 */

// =====================================================
// ASCII SYSTEM FLOW DIAGRAM
// =====================================================
/*
    +-------------------+      +-------------------+      +-------------------+
    |   LoRa RX (C0)    | ---> |  Parse/Validate   | ---> |   Decision Logic  |
    +-------------------+      +-------------------+      +-------------------+
                                                                                |
                                                                                v
    +-------------------+      +-------------------+      +-------------------+
    |   LCD Update (C0) |      |  Audio Cmd Q (C0) | ---> |  Audio Task (C1)  |
    +-------------------+      +-------------------+      +-------------------+
                                                                                |
                                                                                v
    +-------------------+      +-------------------+
    |   I2S Output (C1) |      |   SD Logging (C1) |
    +-------------------+      +-------------------+
*/

// =====================================================
// TODO
// =====================================================
// - Add more robust error reporting to LCD
// - Implement configuration via SD card
// - Add OTA update support

// =====================================================
// FUTURE IMPROVEMENTS
// =====================================================
// - Adaptive deterrent scheduling
// - Remote diagnostics via WiFi
// - Enhanced audio mixing and effects

// ─────────────────────────────────────────────
// GLOBAL CONFIGURATION
// ─────────────────────────────────────────────

#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <RTClib.h>
#include <SD.h>
#include <driver/i2s.h>
#include <cJSON.h>
#include <WiFi.h>
#include <time.h>
#include "siren_ai_policy.h"

#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <RTClib.h>
#include <SD.h>
#include <driver/i2s.h>
#include <cJSON.h>
#include <WiFi.h>
#include <time.h>
#include "siren_ai_policy.h"

// ─────────────────────────────────────────────
// HARDWARE PIN DEFINITIONS
// ─────────────────────────────────────────────
#define WIFI_SSID     "Redmi Note 10 Pro"
#define WIFI_PASSWORD "12345678"

/* ═══ NTP Time Server Config ═══ */
#define NTP_SERVER      "pool.ntp.org"
#define GMT_OFFSET_SEC  19800    // Sri Lanka: GMT+5:30
#define DST_OFFSET_SEC  0
#define TIME_ADJUST_SEC 0

/* ═══ ESP32-S3 SPI Bus Definitions ═══ */
#ifndef FSPI
  #define FSPI 1  // SPI2 on ESP32-S3
#endif
#ifndef HSPI
  #define HSPI 2  // SPI3 on ESP32-S3
#endif

/* ═══ Pin Definitions ═══ */
// LoRa SX1278 (HSPI)
#define LORA_SCK    12
#define LORA_MISO   13
#define LORA_MOSI   11
#define LORA_NSS    10
#define LORA_RST    9
#define LORA_DIO0   46

// I2S Audio (MAX98357)
#define I2S_BCLK    18
#define I2S_LRC     17
#define I2S_DOUT    16

// I2C (LCD + RTC)
#define I2C_SDA     8
#define I2C_SCL     3

// SD Card (FSPI)
#define SD_CS       4
#define SD_SCK      6
#define SD_MOSI     5
#define SD_MISO     7

// Status LED
#define LED_PIN     38

/* ═══ Zone Identity ═══ */
#define SIREN_ZONE_ID  "ZONE-A1"

/* ═══ Audio Constants ═══ */
#define AUDIO_SAMPLE_RATE    22050
#define AUDIO_BUF_SIZE       512
#define WAV_HEADER_SIZE      44
#define MAX_SOUND_FILES      10
#define SOUND_PATH_LEN       64

// ─────────────────────────────────────────────
// INITIALIZATION SECTION
// ─────────────────────────────────────────────
SPIClass* spiSD = nullptr;    // HSPI for SD card (Core 1)
SPIClass* spiLoRa = nullptr;  // FSPI for LoRa (Core 0)

/* ═══════════════════════════════════════════════════════
 *  THREAD SAFETY — SD Card Mutex
 *  Prevents Core 0 and Core 1 from accessing SD simultaneously
 * ═══════════════════════════════════════════════════════ */
SemaphoreHandle_t sdMutex = NULL;

// ...existing code...
typedef struct {
    uint8_t  mode;
    uint8_t  repeat;
    uint16_t gap_ms;
    bool     stop;
} AudioCommand_t;

typedef struct {
    uint8_t  mode;
    int      rssi;
    uint32_t sequence;
    char     boundary_id[64];
} LogCommand_t;

static QueueHandle_t audioQueue   = NULL;
static QueueHandle_t logQueue     = NULL;
static TaskHandle_t  audioTaskHandle = NULL;

// Audio playback state
static volatile bool     audio_playing = false;
static volatile uint8_t  audio_current_mode = 0;

// ...existing code...
class SDSafe {
public:
    static bool begin(uint8_t csPin, SPIClass &spi, uint32_t frequency = 4000000) {
        if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
            bool result = SD.begin(csPin, spi, frequency);
            xSemaphoreGive(sdMutex);
            return result;
        }
        return false;
    }
    
    static File open(const char* path, const char* mode = FILE_READ) {
        File f;
        if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
            f = SD.open(path, mode);
            xSemaphoreGive(sdMutex);
        }
        return f;
    }
    
    static bool exists(const char* path) {
        bool result = false;
        if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
            result = SD.exists(path);
            xSemaphoreGive(sdMutex);
        }
        return result;
    }
    
    static bool mkdir(const char* path) {
        bool result = false;
        if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
            result = SD.mkdir(path);
            xSemaphoreGive(sdMutex);
        }
        return result;
    }
};

// ...existing code...
bool syncRTCwithNTP(RTC_DS3231& rtc, LiquidCrystal_I2C* lcd) {
    Serial.println("\n[NTP] Starting WiFi connection for RTC sync...");
    if (lcd) {
        lcd->clear();
        lcd->setCursor(0, 0);
        lcd->print("WiFi Connecting");
        lcd->setCursor(0, 1);
        lcd->print("NTP Time Sync...");
    }
    
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();
    
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[NTP] WiFi connection failed!");
        if (lcd) {
            lcd->clear();
            lcd->setCursor(0, 0);
            lcd->print("WiFi Failed");
            lcd->setCursor(0, 1);
            lcd->print("Using RTC Time");
        }
        delay(2000);
        return false;
    }
    
    Serial.println("[NTP] WiFi connected!");
    Serial.print("[NTP] IP: ");
    Serial.println(WiFi.localIP());
    
    if (lcd) {
        lcd->clear();
        lcd->setCursor(0, 0);
        lcd->print("WiFi Connected");
        lcd->setCursor(0, 1);
        lcd->print("Syncing Time...");
    }
    
    configTime(GMT_OFFSET_SEC + TIME_ADJUST_SEC, DST_OFFSET_SEC, NTP_SERVER);
    
    struct tm timeinfo;
    attempts = 0;
    while (!getLocalTime(&timeinfo) && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();
    
    if (!getLocalTime(&timeinfo)) {
        Serial.println("[NTP] Time sync failed!");
        WiFi.disconnect(true);
        delay(2000);
        return false;
    }
    
    Serial.println("[NTP] Time synchronized from internet!");
    Serial.printf("[NTP] Local time: %04d-%02d-%02d %02d:%02d:%02d\n",
                  timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday,
                  timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
    
    time_t now;
    time(&now);
    DateTime ntpTime = DateTime((uint32_t)now);
    rtc.adjust(ntpTime);
    
    Serial.println("[NTP] DS3231 RTC updated!");
    
    if (lcd) {
        lcd->clear();
        lcd->setCursor(0, 0);
        lcd->print("RTC Sync OK");
        char timeBuf[17];
        snprintf(timeBuf, 17, "%02d/%02d %02d:%02d:%02d",
                 timeinfo.tm_mon + 1, timeinfo.tm_mday,
                 timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
        lcd->setCursor(0, 1);
        lcd->print(timeBuf);
    }
    
    delay(2000);
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    Serial.println("[NTP] WiFi disconnected — power saving mode");
    
    return true;
}

// ...existing code...
class SoundLibrary {
public:
    void init() {
        _countBee = 0;
        _countLeopard = 0;
        _countSiren = 0;
        
        // Scan directories with SD mutex protection
        _scanDirectory("/sounds/bee",     _beeFiles,     _countBee);
        _scanDirectory("/sounds/leopard", _leopardFiles, _countLeopard);
        _scanDirectory("/sounds/siren",   _sirenFiles,   _countSiren);
        
        Serial.printf("[SND] Library: Bee=%d, Leopard=%d, Siren=%d\n",
                      _countBee, _countLeopard, _countSiren);
        
        // Verify we can actually open the files
        Serial.println("[SND] Verifying file access...");
        if (_countBee > 0) {
            char testPath[64];
            snprintf(testPath, 64, "/sounds/bee/%s", _beeFiles[0]);
            _verifyFileAccess(testPath);
        }
        if (_countLeopard > 0) {
            char testPath[64];
            snprintf(testPath, 64, "/sounds/leopard/%s", _leopardFiles[0]);
            _verifyFileAccess(testPath);
        }
        if (_countSiren > 0) {
            char testPath[64];
            snprintf(testPath, 64, "/sounds/siren/%s", _sirenFiles[0]);
            _verifyFileAccess(testPath);
        }
    }

    bool getFileForMode(uint8_t mode, char* outPath, size_t maxLen) {
        switch (mode) {
            case SIREN_MODE_M1:
                return _pickRandom(_beeFiles, _countBee, "/sounds/bee", outPath, maxLen);
            case SIREN_MODE_M2:
                return _pickRandom(_leopardFiles, _countLeopard, "/sounds/leopard", outPath, maxLen);
            case SIREN_MODE_M3:
                return _pickRandom(_sirenFiles, _countSiren, "/sounds/siren", outPath, maxLen);
            default:
                return false;
        }
    }

    uint8_t getBeeCount()     { return _countBee; }
    uint8_t getLeopardCount() { return _countLeopard; }
    uint8_t getSirenCount()   { return _countSiren; }

private:
    char    _beeFiles[MAX_SOUND_FILES][48];
    char    _leopardFiles[MAX_SOUND_FILES][48];
    char    _sirenFiles[MAX_SOUND_FILES][48];
    uint8_t _countBee;
    uint8_t _countLeopard;
    uint8_t _countSiren;

    void _scanDirectory(const char* dirPath, char files[][48], uint8_t &count) {
        count = 0;
        
        Serial.printf("[SND] Scanning directory: %s\n", dirPath);
        
        // Thread-safe SD access - take mutex
        if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
            Serial.printf("[SND] ERR: Failed to acquire mutex for %s\n", dirPath);
            return;
        }
        
        File dir = SD.open(dirPath);
        if (!dir) {
            Serial.printf("[SND] WARN: Cannot open directory: %s\n", dirPath);
            xSemaphoreGive(sdMutex);
            return;
        }
        
        if (!dir.isDirectory()) {
            Serial.printf("[SND] WARN: %s is not a directory\n", dirPath);
            dir.close();
            xSemaphoreGive(sdMutex);
            return;
        }
        
        File entry;
        while ((entry = dir.openNextFile()) && count < MAX_SOUND_FILES) {
            if (!entry.isDirectory()) {
                const char* name = entry.name();
                
                // Get just the filename (remove path if present)
                const char* fileName = strrchr(name, '/');
                if (fileName) {
                    fileName++;  // Skip the '/'
                } else {
                    fileName = name;
                }
                
                size_t nlen = strlen(fileName);
                if (nlen > 4 && (strcasecmp(fileName + nlen - 4, ".wav") == 0)) {
                    strncpy(files[count], fileName, 47);
                    files[count][47] = '\0';
                    count++;
                    Serial.printf("[SND]   + %s/%s\n", dirPath, fileName);
                }
            }
            entry.close();
        }
        dir.close();
        xSemaphoreGive(sdMutex);
        
        Serial.printf("[SND] Found %d files in %s\n", count, dirPath);
    }

    bool _pickRandom(char files[][48], uint8_t count,
                     const char* dir, char* outPath, size_t maxLen) {
        if (count == 0) return false;
        uint8_t idx = (count == 1) ? 0 : (random(0, count));
        snprintf(outPath, maxLen, "%s/%s", dir, files[idx]);
        return true;
    }
    
    void _verifyFileAccess(const char* path) {
        if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(2000)) == pdTRUE) {
            File testFile = SD.open(path, FILE_READ);
            if (testFile) {
                Serial.printf("[SND] ✓ Can open: %s (size: %u bytes)\n", path, testFile.size());
                testFile.close();
            } else {
                Serial.printf("[SND] ✗ Cannot open: %s\n", path);
            }
            xSemaphoreGive(sdMutex);
        } else {
            Serial.printf("[SND] ✗ Mutex timeout for: %s\n", path);
        }
    }
};

// ...existing code...
class WAVPlayer {
public:
    bool play(const char* filePath) {
        Serial.printf("[WAV] Attempting to open: %s\n", filePath);
        
        // CRITICAL FIX: Take mutex for entire read operation
        if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
            Serial.println("[WAV] ERR: Failed to acquire SD mutex");
            return false;
        }
        
        // Open file with direct SD.open() while holding mutex
        File wavFile = SD.open(filePath, FILE_READ);
        if (!wavFile) {
            Serial.printf("[WAV] ERR: Cannot open %s\n", filePath);
            xSemaphoreGive(sdMutex);
            return false;
        }
        
        Serial.printf("[WAV] File opened successfully, size: %u bytes\n", wavFile.size());

        // ═══ NEW: Proper WAV Header Parsing ═══
        uint8_t header[WAV_HEADER_SIZE];
        if (wavFile.read(header, 12) != 12) {  // Read first 12 bytes (RIFF header)
            Serial.println("[WAV] ERR: Cannot read RIFF header");
            wavFile.close();
            xSemaphoreGive(sdMutex);
            return false;
        }

        // Verify RIFF/WAVE
        if (header[0] != 'R' || header[1] != 'I' ||
            header[2] != 'F' || header[3] != 'F' ||
            header[8] != 'W' || header[9] != 'A' ||
            header[10] != 'V' || header[11] != 'E') {
            Serial.println("[WAV] ERR: Not a valid WAV file");
            wavFile.close();
            xSemaphoreGive(sdMutex);
            return false;
        }

        // Search for "fmt " chunk
        uint16_t channels = 0;
        uint32_t sampleRate = 0;
        uint16_t bitsPerSamp = 0;
        bool fmtFound = false;

        while (!fmtFound && wavFile.available() >= 8) {
            uint8_t chunkHeader[8];
            if (wavFile.read(chunkHeader, 8) != 8) break;

            char chunkID[5] = {0};
            memcpy(chunkID, chunkHeader, 4);
            uint32_t chunkSize = chunkHeader[4] | (chunkHeader[5] << 8) |
                                (chunkHeader[6] << 16) | (chunkHeader[7] << 24);

            Serial.printf("[WAV] Found chunk: '%s', size: %u bytes\n", chunkID, chunkSize);

            if (strcmp(chunkID, "fmt ") == 0) {
                // Read fmt chunk
                if (chunkSize < 16) {
                    Serial.println("[WAV] ERR: Invalid fmt chunk size");
                    break;
                }

                uint8_t fmtData[16];
                if (wavFile.read(fmtData, 16) != 16) {
                    Serial.println("[WAV] ERR: Cannot read fmt chunk");
                    break;
                }

                uint16_t audioFormat = fmtData[0] | (fmtData[1] << 8);
                channels = fmtData[2] | (fmtData[3] << 8);
                sampleRate = fmtData[4] | (fmtData[5] << 8) |
                            (fmtData[6] << 16) | (fmtData[7] << 24);
                bitsPerSamp = fmtData[14] | (fmtData[15] << 8);

                Serial.printf("[WAV] Format: %u, Channels: %u, Rate: %u Hz, Bits: %u\n",
                             audioFormat, channels, sampleRate, bitsPerSamp);

                // Validate
                if (audioFormat != 1) {  // 1 = PCM
                    Serial.printf("[WAV] ERR: Unsupported audio format: %u (only PCM supported)\n", audioFormat);
                    wavFile.close();
                    xSemaphoreGive(sdMutex);
                    return false;
                }

                if (channels == 0 || channels > 2) {
                    Serial.printf("[WAV] ERR: Invalid channel count: %u\n", channels);
                    wavFile.close();
                    xSemaphoreGive(sdMutex);
                    return false;
                }

                if (bitsPerSamp != 8 && bitsPerSamp != 16) {
                    Serial.printf("[WAV] ERR: Unsupported bit depth: %u\n", bitsPerSamp);
                    wavFile.close();
                    xSemaphoreGive(sdMutex);
                    return false;
                }

                fmtFound = true;

                // Skip remaining fmt chunk data if any
                if (chunkSize > 16) {
                    wavFile.seek(wavFile.position() + (chunkSize - 16));
                }
            } else {
                // Skip this chunk
                wavFile.seek(wavFile.position() + chunkSize);
            }
        }

        if (!fmtFound) {
            Serial.println("[WAV] ERR: fmt chunk not found");
            wavFile.close();
            xSemaphoreGive(sdMutex);
            return false;
        }

        // Search for "data" chunk
        uint32_t dataSize = 0;
        bool dataFound = false;

        while (!dataFound && wavFile.available() >= 8) {
            uint8_t chunkHeader[8];
            if (wavFile.read(chunkHeader, 8) != 8) break;

            char chunkID[5] = {0};
            memcpy(chunkID, chunkHeader, 4);
            uint32_t chunkSize = chunkHeader[4] | (chunkHeader[5] << 8) |
                                (chunkHeader[6] << 16) | (chunkHeader[7] << 24);

            Serial.printf("[WAV] Found chunk: '%s', size: %u bytes\n", chunkID, chunkSize);

            if (strcmp(chunkID, "data") == 0) {
                dataSize = chunkSize;
                dataFound = true;
                Serial.printf("[WAV] ✓ Data chunk found at position %u, size: %u bytes\n",
                             wavFile.position(), dataSize);
            } else {
                // Skip this chunk
                wavFile.seek(wavFile.position() + chunkSize);
            }
        }

        if (!dataFound || dataSize == 0) {
            Serial.println("[WAV] ERR: data chunk not found or empty");
            wavFile.close();
            xSemaphoreGive(sdMutex);
            return false;
        }

        // Now we're at the start of actual audio data
        Serial.printf("[WAV] Playing: %s\n", filePath);
        Serial.printf("[WAV]   %uHz, %u-bit, %uch, %u bytes audio data\n",
                      sampleRate, bitsPerSamp, channels, dataSize);

        // Set sample rate if different
        if (sampleRate != AUDIO_SAMPLE_RATE) {
            i2s_set_sample_rates(I2S_NUM_0, sampleRate);
        }

        // Playback loop
        int16_t audioBuf[AUDIO_BUF_SIZE];
        size_t bytesRead, bytesWritten;
        uint32_t totalRead = 0;

        audio_playing = true;

        while (totalRead < dataSize) {
            // Check for stop command
            AudioCommand_t cmd;
            if (xQueuePeek(audioQueue, &cmd, 0) == pdTRUE && cmd.stop) {
                Serial.println("[WAV] Playback stopped by command");
                audio_playing = false;
                wavFile.close();
                xSemaphoreGive(sdMutex);
                return false;
            }

            // Calculate how much to read
            size_t toRead = sizeof(audioBuf);
            if (totalRead + toRead > dataSize) {
                toRead = dataSize - totalRead;
            }

            bytesRead = wavFile.read((uint8_t*)audioBuf, toRead);
            if (bytesRead == 0) {
                Serial.println("[WAV] Read error or EOF");
                break;
            }

            // ═══ Audio Processing ═══
            size_t samplesRead = bytesRead / (bitsPerSamp / 8);

            // Stereo to mono conversion (if needed)
            if (channels == 2 && bitsPerSamp == 16) {
                // 16-bit stereo → 16-bit mono
                size_t monoSamples = samplesRead / 2;
                for (size_t i = 0; i < monoSamples; i++) {
                    int32_t left = audioBuf[i * 2];
                    int32_t right = audioBuf[i * 2 + 1];
                    audioBuf[i] = (int16_t)((left + right) / 2);  // Average L+R
                }
                bytesRead = monoSamples * 2;  // Update byte count
            } else if (channels == 2 && bitsPerSamp == 8) {
                // 8-bit stereo → first convert to 16-bit, then mix
                uint8_t* src = (uint8_t*)audioBuf;
                for (int i = samplesRead - 1; i >= 0; i -= 2) {
                    if (i >= 1) {
                        int32_t left = ((int16_t)src[i - 1] - 128) << 8;
                        int32_t right = ((int16_t)src[i] - 128) << 8;
                        audioBuf[i / 2] = (int16_t)((left + right) / 2);
                    }
                }
                bytesRead = (samplesRead / 2) * 2;
            }

            // 8-bit to 16-bit conversion (if mono 8-bit)
            if (channels == 1 && bitsPerSamp == 8) {
                uint8_t* src = (uint8_t*)audioBuf;
                for (int i = bytesRead - 1; i >= 0; i--) {
                    audioBuf[i] = ((int16_t)src[i] - 128) << 8;
                }
                bytesRead *= 2;
            }

            // Write to I2S
            i2s_write(I2S_NUM_0, audioBuf, bytesRead, &bytesWritten, portMAX_DELAY);
            totalRead += toRead;

            vTaskDelay(1);  // Yield to prevent watchdog
        }

        // Restore default sample rate if changed
        if (sampleRate != AUDIO_SAMPLE_RATE) {
            i2s_set_sample_rates(I2S_NUM_0, AUDIO_SAMPLE_RATE);
        }

        // Flush I2S buffer
        int16_t silence[64] = {0};
        for (int i = 0; i < 4; i++) {
            i2s_write(I2S_NUM_0, silence, sizeof(silence), &bytesWritten, 100);
        }

        audio_playing = false;
        wavFile.close();
        
        // CRITICAL: Release mutex after all file operations complete
        xSemaphoreGive(sdMutex);
        
        Serial.printf("[WAV] ✓ Playback complete: %u bytes played\n", totalRead);
        return true;
    }

    void playTone(uint16_t freqHz, uint32_t durationMs) {
        const int samples = (AUDIO_SAMPLE_RATE * durationMs) / 1000;
        int16_t sample;
        size_t bytesWritten;

        audio_playing = true;
        Serial.printf("[WAV] Fallback tone: %uHz, %ums\n", freqHz, durationMs);

        for (int i = 0; i < samples; i++) {
            sample = (int16_t)(16000.0f * sinf(2.0f * PI * freqHz * i / AUDIO_SAMPLE_RATE));
            i2s_write(I2S_NUM_0, &sample, sizeof(sample), &bytesWritten, 100);
        }

        audio_playing = false;
    }
};

// ...existing code...
class LCDManager {
public:
    void init() {
        Wire.beginTransmission(0x27);
        if (Wire.endTransmission() == 0) {
            _addr = 0x27;
        } else {
            Wire.beginTransmission(0x3F);
            _addr = (Wire.endTransmission() == 0) ? 0x3F : 0x27;
        }
        _lcd = new LiquidCrystal_I2C(_addr, 16, 2);
        _lcd->init();
        _lcd->backlight();
        _clearBuf();

        byte charBar[8]  = {0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18};
        byte charFwd[8]  = {0x01,0x02,0x04,0x08,0x10,0x00,0x00,0x00};
        byte charDash[8] = {0x00,0x00,0x00,0x1F,0x1F,0x00,0x00,0x00};
        byte charBck[8]  = {0x10,0x08,0x04,0x02,0x01,0x00,0x00,0x00};
        byte charOK[8]   = {0x00,0x01,0x03,0x16,0x1C,0x08,0x00,0x00};
        byte charX[8]    = {0x00,0x11,0x0A,0x04,0x0A,0x11,0x00,0x00};
        byte charLk[8]   = {0x0E,0x11,0x11,0x1F,0x1B,0x1B,0x1F,0x00};
        byte charSp[8]   = {0x04,0x0E,0x15,0x04,0x04,0x04,0x0E,0x00};

        _lcd->createChar(0, charBar);
        _lcd->createChar(1, charFwd);
        _lcd->createChar(2, charDash);
        _lcd->createChar(3, charBck);
        _lcd->createChar(4, charOK);
        _lcd->createChar(5, charX);
        _lcd->createChar(6, charLk);
        _lcd->createChar(7, charSp);

        _spinIdx = 0;
        _lastSpinMs = 0;
        _lastMainMs = 0;
        _lastFooterMs = 0;
        _footerToggle = false;
        _eventExpireMs = 0;
        _eventActive = false;
    }

    LiquidCrystal_I2C* getLCD() { return _lcd; }

    void showBoot() {
        _writeLine(0, "  SIREN AI v3   ");
        _writeLine(1, "  FINAL FIX!    ");
        _flush();
        delay(1200);
    }

    void showSystemReady() {
        _writeLine(0, "  System Ready  ");
        _writeLine(1, " WAV Fix OK!   ");
        _flush();
        delay(1500);
    }

    void showLoRaOK() {
        _writeLine(0, "  LoRa Init OK  ");
        _writeLine(1, "  433MHz Ready  ");
        _flush();
        delay(1000);
    }

    void showLoRaError() {
        _writeLine(0, "  LoRa ERROR    ");
        _writeLine(1, "  Check Module  ");
        _flush();
    }

    void showSDInfo(uint8_t bee, uint8_t leopard, uint8_t siren) {
        char l1[17];
        snprintf(l1, 17, "B:%d L:%d S:%d     ", bee, leopard, siren);
        _writeLine(0, " SD Sound Files ");
        _writeLine(1, l1);
        _flush();
        delay(1200);
    }

    void showLoRaWaiting() {
        _writeLine(0, " Awaiting LoRa  ");
        _writeLine(1, "  Listening...  ");
        _flush();
    }

    void showLoRaAccepted(uint32_t seq) {
        char l0[17], l1[17];
        snprintf(l0, 17, " %c LoRa Valid   ", '\x04');
        snprintf(l1, 17, "SEQ:%010lu", (unsigned long)seq);
        _setEvent(l0, l1, 2000);
    }

    void showLoRaRejected(const char* reason) {
        char l0[17], l1[17];
        snprintf(l0, 17, " %c LoRa Reject ", '\x05');
        snprintf(l1, 17, "%-16s", reason);
        _setEvent(l0, l1, 2000);
    }

    void showTTLExpired() {
        char l0[17];
        snprintf(l0, 17, " %c TTL Expired ", '\x05');
        _setEvent(l0, "Msg Too Old     ", 2000);
    }

    void showReplayDetected() {
        char l0[17];
        snprintf(l0, 17, " %c Replay Det  ", '\x06');
        _setEvent(l0, "Seq# Duplicate  ", 2000);
    }

    void showDecision(uint8_t mode, uint8_t confidence) {
        const char* modeStr[] = {"M0:Observe", "M1:Deter  ", "M2:Strong ", "M3:ALERT  "};
        const char* confStr[] = {"LOW", "MED", "HI "};
        uint8_t m = (mode < 4) ? mode : 0;
        uint8_t c = (confidence < 3) ? confidence : 1;
        char l0[17], l1[17];
        snprintf(l0, 17, "RL> %-10s  ", modeStr[m]);
        snprintf(l1, 17, "Conf:%-3s        ", confStr[c]);
        _setEvent(l0, l1, 2500);
    }

    void showSafetyOverride(uint8_t forced_mode) {
        const char* mStr[] = {"M0", "M1", "M2", "M3"};
        uint8_t m = (forced_mode < 4) ? forced_mode : 3;
        char l0[17], l1[17];
        snprintf(l0, 17, " SAFETY OVERRIDE");
        snprintf(l1, 17, "Forced > %-2s %c   ", mStr[m], '\x06');
        _setEvent(l0, l1, 2500);
    }

    void showSoundPlaying(uint8_t mode) {
        const char* snd[] = {"---", "Bee Buzz", "Leopard", "Siren"};
        uint8_t m = (mode < 4) ? mode : 0;
        char l0[17], l1[17];
        snprintf(l0, 17, " %c Playing...  ", '\x07');
        snprintf(l1, 17, "%-16s", snd[m]);
        _setEvent(l0, l1, 3000);
    }

    void showCooldown(uint16_t seconds) {
        char l0[17], l1[17];
        snprintf(l0, 17, "COOLDOWN ACTIVE ");
        snprintf(l1, 17, "Remain: %4us   ", seconds);
        _setEvent(l0, l1, 1500);
    }

    void showLockout() {
        char l0[17];
        snprintf(l0, 17, " %c LOCKOUT ON  ", '\x06');
        _setEvent(l0, "Aggr. Override  ", 2500);
    }

    void showBudgetLimit() {
        _setEvent(" BUDGET LIMIT   ", "MaxAct Reached  ", 2000);
    }

    void showMainStatus(uint8_t mode, uint16_t cooldown, uint8_t activations) {
        _curMode = mode;
        _curCooldown = cooldown;
        _curActivations = activations;
    }

    void updateSpinner() {
        unsigned long now = millis();
        if (now - _lastSpinMs >= 500) {
            _lastSpinMs = now;
            _spinIdx = (_spinIdx + 1) % 4;
        }
    }

    void updateFooter() {
        unsigned long now = millis();
        if (now - _lastFooterMs >= 5000) {
            _lastFooterMs = now;
            _footerToggle = !_footerToggle;
        }
    }

    void nonBlockingUpdate() {
        unsigned long now = millis();
        updateSpinner();
        updateFooter();

        if (_eventActive) {
            if (now >= _eventExpireMs) {
                _eventActive = false;
                _forceRefresh = true;
            } else {
                return;
            }
        }

        if (now - _lastMainMs >= 1000 || _forceRefresh) {
            _lastMainMs = now;
            _forceRefresh = false;

            const char* mTag[] = {"M0", "M1", "M2", "M3"};
            const char* mLbl[] = {"SAFE ", "DETER", "STRNG", "ALERT"};
            uint8_t m = (_curMode < 4) ? _curMode : 0;

            char line0[17];
            if (audio_playing) {
                snprintf(line0, 17, "MODE:%s %-5s%c", mTag[m], mLbl[m], '\x07');
            } else {
                snprintf(line0, 17, "MODE:%s %-5s %c", mTag[m], mLbl[m], _spinChar());
            }

            char line1[17];
            if (_curCooldown > 0 || _curActivations > 0) {
                if (_footerToggle) {
                    _buildFooter(line1);
                } else {
                    snprintf(line1, 17, "CD:%03u ACT:%02u   ",
                            _curCooldown > 999 ? 999 : _curCooldown,
                            _curActivations > 99 ? 99 : _curActivations);
                }
            } else {
                _buildFooter(line1);
            }

            _writeLineSelective(0, line0);
            _writeLineSelective(1, line1);
        }
    }

private:
    LiquidCrystal_I2C* _lcd;
    uint8_t _addr;
    char _prev[2][17];
    char _buf[2][17];
    uint8_t  _curMode = 0;
    uint16_t _curCooldown = 0;
    uint8_t  _curActivations = 0;
    uint8_t  _spinIdx;
    unsigned long _lastSpinMs;
    bool _footerToggle;
    unsigned long _lastFooterMs;
    unsigned long _lastMainMs;
    bool _forceRefresh = true;
    bool _eventActive;
    unsigned long _eventExpireMs;

    void _clearBuf() {
        memset(_prev[0],' ',16); _prev[0][16]='\0';
        memset(_prev[1],' ',16); _prev[1][16]='\0';
        memset(_buf[0],' ',16);  _buf[0][16]='\0';
        memset(_buf[1],' ',16);  _buf[1][16]='\0';
    }

    char _spinChar() {
        const char sp[] = {'|','/','-','\\'};
        return sp[_spinIdx % 4];
    }

    void _writeLine(uint8_t row, const char* text) {
        char buf[17];
        snprintf(buf,17,"%-16s",text);
        memcpy(_buf[row], buf, 16);
        _buf[row][16]='\0';
    }

    void _flush() {
        for (uint8_t r=0; r<2; r++) {
            _lcd->setCursor(0, r);
            _lcd->print(_buf[r]);
            memcpy(_prev[r], _buf[r], 17);
        }
    }

    void _writeLineSelective(uint8_t row, const char* text) {
        char padded[17];
        snprintf(padded,17,"%-16s",text);
        for (uint8_t col=0; col<16; col++) {
            if (padded[col] != _prev[row][col]) {
                _lcd->setCursor(col, row);
                _lcd->print(padded[col]);
                _prev[row][col] = padded[col];
            }
        }
    }

    void _setEvent(const char* l0, const char* l1, uint16_t ms) {
        char p0[17], p1[17];
        snprintf(p0,17,"%-16s",l0);
        snprintf(p1,17,"%-16s",l1);
        _writeLineSelective(0, p0);
        _writeLineSelective(1, p1);
        _eventActive = true;
        _eventExpireMs = millis() + ms;
    }

    void _buildFooter(char* out) {
        if (_footerToggle) snprintf(out,17," Namo Buddhaya  ");
        else               snprintf(out,17," System Stable  ");
    }
};

// ─────────────────────────────────────────────
// MAIN LOOP / TASK HANDLERS
// ─────────────────────────────────────────────
LCDManager    lcdMgr;
RTC_DS3231    rtc;
SoundLibrary  soundLib;
WAVPlayer     wavPlayer;

/* ═══ Core 0 State Variables ═══ */
static uint32_t last_sequence      = 0;
static uint8_t  current_mode       = SIREN_MODE_M0;
static uint8_t  activations_count  = 0;
static uint32_t lockout_until      = 0;
static uint32_t cooldown_until     = 0;
static uint32_t last_activation_time = 0;

/* ═══════════════════════════════════════════════════════
 *  CORE 1 TASK — Audio Playback & Logging
 *  CRITICAL FIX: Proper delay and stability improvements
 * ═══════════════════════════════════════════════════════ */
/**
 * @brief FreeRTOS Core 1 audio playback and logging task
 * @details Handles audio commands and SD logging on Core 1
 */
void audioTask(void* pvParameters) {
    Serial.println("[CORE1] Audio task started");
    
    // CRITICAL: Give Core 1 time to stabilize
    vTaskDelay(pdMS_TO_TICKS(500));
    
    AudioCommand_t cmd;
    LogCommand_t   logCmd;
    char filePath[SOUND_PATH_LEN];

    for (;;) {
        // Process log commands (non-blocking)
        while (xQueueReceive(logQueue, &logCmd, 0) == pdTRUE) {
            _writeLogEntry(logCmd.mode, logCmd.boundary_id,
                          logCmd.rssi, logCmd.sequence);
        }

        // Wait for audio command
        if (xQueueReceive(audioQueue, &cmd, pdMS_TO_TICKS(50)) == pdTRUE) {
            if (cmd.stop) {
                audio_playing = false;
                Serial.println("[CORE1] Audio stopped");
                continue;
            }

            if (cmd.mode == SIREN_MODE_M0) continue;

            audio_current_mode = cmd.mode;

            for (uint8_t r = 0; r < cmd.repeat; r++) {
                if (soundLib.getFileForMode(cmd.mode, filePath, SOUND_PATH_LEN)) {
                    Serial.printf("[CORE1] Mode M%d → %s (rep %d/%d)\n",
                                  cmd.mode, filePath, r + 1, cmd.repeat);
                    
                    if (!wavPlayer.play(filePath)) {
                        Serial.println("[CORE1] Playback failed, using fallback tone");
                        uint16_t freq = (cmd.mode == SIREN_MODE_M1) ? 400 :
                                       (cmd.mode == SIREN_MODE_M2) ? 600 : 900;
                        uint32_t dur  = (cmd.mode == SIREN_MODE_M1) ? 2000 :
                                       (cmd.mode == SIREN_MODE_M2) ? 3000 : 5000;
                        wavPlayer.playTone(freq, dur);
                    }
                } else {
                    Serial.printf("[CORE1] No WAV for M%d — fallback tone\n", cmd.mode);
                    uint16_t freq = (cmd.mode == SIREN_MODE_M1) ? 400 :
                                   (cmd.mode == SIREN_MODE_M2) ? 600 : 900;
                    uint32_t dur  = (cmd.mode == SIREN_MODE_M1) ? 2000 :
                                   (cmd.mode == SIREN_MODE_M2) ? 3000 : 5000;
                    wavPlayer.playTone(freq, dur);
                }

                if (r < cmd.repeat - 1 && cmd.gap_ms > 0) {
                    vTaskDelay(pdMS_TO_TICKS(cmd.gap_ms));
                }

                AudioCommand_t peek;
                if (xQueuePeek(audioQueue, &peek, 0) == pdTRUE && peek.stop) break;
            }

            audio_current_mode = 0;
        }
        
        // CRITICAL: Prevent watchdog timeout
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

/* ═══ SD Log Writer (Core 1) ═══ */
/**
 * @brief Write log entry to SD card
 * @param mode Siren mode
 * @param boundary_id Boundary segment
 * @param rssi LoRa RSSI
 * @param sequence Sequence number
 */
void _writeLogEntry(uint8_t mode, const char* boundary_id,
                   int rssi, uint32_t sequence) {
    if (!SDSafe::exists("/logs")) {
        SDSafe::mkdir("/logs");
    }

    DateTime now = rtc.now();
    char filename[32];
    sprintf(filename, "/logs/%04d%02d%02d.log",
            now.year(), now.month(), now.day());

    File logFile = SDSafe::open(filename, FILE_APPEND);
    if (logFile) {
        logFile.printf("%04d-%02d-%02dT%02d:%02d:%02d,",
                      now.year(), now.month(), now.day(),
                      now.hour(), now.minute(), now.second());
        logFile.printf("M%d,%s,%d,%u\n", mode, boundary_id, rssi, sequence);
        logFile.close();
    }
}

/* ═══════════════════════════════════════════════════════
 *  SETUP — Core 0 Initialization
 * ═══════════════════════════════════════════════════════ */
/**
 * @brief Arduino setup function (Core 0)
 * @details Initializes all hardware, peripherals, and tasks
 */
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("=== Siren AI v3 — FINAL FIX ===");
    Serial.println("Wildlife 360 Deterrent System");
    Serial.printf("Zone ID: %s\n", SIREN_ZONE_ID);
    Serial.println("NEW: Proper WAV chunk parsing + divide-by-zero protection");

    // ═══ CRITICAL: Create SD mutex FIRST ═══
    sdMutex = xSemaphoreCreateMutex();
    if (sdMutex == NULL) {
        Serial.println("[ERR] Failed to create SD mutex!");
        while (1) delay(1000);
    }
    Serial.println("[OK] SD mutex created");

    // I2C init
    Wire.begin(I2C_SDA, I2C_SCL);

    // LCD init
    lcdMgr.init();
    lcdMgr.showBoot();

    // RTC init
    if (!rtc.begin()) {
        Serial.println("[ERR] RTC not found!");
        lcdMgr.getLCD()->clear();
        lcdMgr.getLCD()->setCursor(0, 0);
        lcdMgr.getLCD()->print("RTC ERROR!");
        delay(3000);
    } else {
        Serial.println("[OK] RTC ready");
        syncRTCwithNTP(rtc, lcdMgr.getLCD());
    }

    // ═══ Initialize SPI Buses ═══
    Serial.println("\n[INIT] Configuring SPI buses...");
    
    // HSPI for SD Card
    spiSD = new SPIClass(HSPI);
    spiSD->begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
    Serial.println("[OK] HSPI initialized for SD card");
    
    // FSPI for LoRa
    spiLoRa = new SPIClass(FSPI);
    spiLoRa->begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_NSS);
    Serial.println("[OK] FSPI initialized for LoRa");

    // ═══ LoRa Init ═══
    Serial.println("[INIT] Starting LoRa on FSPI...");
    LoRa.setSPI(*spiLoRa);
    LoRa.setPins(LORA_NSS, LORA_RST, LORA_DIO0);
    
    if (!LoRa.begin(433E6)) {
        Serial.println("[ERR] LoRa init failed!");
        lcdMgr.showLoRaError();
        while (1) delay(1000);
    }
    
    LoRa.setSpreadingFactor(10);
    LoRa.setSignalBandwidth(125E3);
    LoRa.setCodingRate4(5);
    Serial.println("[OK] LoRa 433MHz ready");

    // ═══ SD Card Init ═══
    Serial.println("[INIT] Starting SD card on HSPI...");
    delay(100);
    
    if (!SDSafe::begin(SD_CS, *spiSD, 4000000)) {
        Serial.println("[WARN] SD card not found — trying slower speed...");
        delay(500);
        if (!SDSafe::begin(SD_CS, *spiSD, 1000000)) {
            Serial.println("[ERR] SD card init failed! Audio fallback only.");
        } else {
            Serial.println("[OK] SD card ready (slow mode)");
        }
    } else {
        Serial.println("[OK] SD card ready");
    }
    
    delay(200);

    // ═══ Sound Library Init ═══
    soundLib.init();
    lcdMgr.showSDInfo(soundLib.getBeeCount(),
                     soundLib.getLeopardCount(),
                     soundLib.getSirenCount());

    // ═══ I2S Init ═══
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = AUDIO_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = AUDIO_BUF_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = true,
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK,
        .ws_io_num = I2S_LRC,
        .data_out_num = I2S_DOUT,
        .data_in_num = I2S_PIN_NO_CHANGE,
    };
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    Serial.println("[OK] I2S audio ready");

    // ═══ Create Queues ═══
    audioQueue = xQueueCreate(4, sizeof(AudioCommand_t));
    logQueue   = xQueueCreate(8, sizeof(LogCommand_t));
    Serial.println("[OK] Inter-core queues created");

    // ═══ Launch Core 1 Task ═══
    xTaskCreatePinnedToCore(
        audioTask,
        "AudioTask",
        8192,
        NULL,
        2,
        &audioTaskHandle,
        1  // Core 1
    );
    Serial.println("[OK] Core 1 audio task launched");

    delay(500);

    lcdMgr.showLoRaOK();
    lcdMgr.showSystemReady();
    lcdMgr.showMainStatus(SIREN_MODE_M0, 0, 0);

    Serial.println("[READY] Siren AI v3 — WAV PARSING FIXED!");
    Serial.println("[CORE0] LoRa + LCD + Decision");
    Serial.println("[CORE1] WAV playback + Logging");
}

/* ═══════════════════════════════════════════════════════
 *  MAIN LOOP — Core 0 Only
 * ═══════════════════════════════════════════════════════ */
/**
 * @brief Arduino main loop (Core 0)
 * @details Handles LoRa RX, state management, and LCD updates
 */
void loop() {
    // LoRa receive
    int packetSize = LoRa.parsePacket();
    if (packetSize > 0) {
        process_lora_message(packetSize);
    }

    // Time-based state management
    DateTime now = rtc.now();
    uint32_t current_time = now.unixtime();

    if (lockout_until > 0 && current_time >= lockout_until) {
        lockout_until = 0;
        Serial.println("[LOCK] Lockout expired");
    }

    uint16_t cd_remaining = 0;
    if (cooldown_until > 0 && current_time >= cooldown_until) {
        cooldown_until = 0;
    } else if (cooldown_until > current_time) {
        cd_remaining = (uint16_t)(cooldown_until - current_time);
    }

    if (current_time - last_activation_time > 3600) {
        activations_count = 0;
        last_activation_time = current_time;
    }

    lcdMgr.showMainStatus(current_mode, cd_remaining, activations_count);

    if (lockout_until > 0 && lockout_until > current_time) {
        static unsigned long lastLockoutShow = 0;
        if (millis() - lastLockoutShow > 10000) {
            lcdMgr.showLockout();
            lastLockoutShow = millis();
        }
    }

    lcdMgr.nonBlockingUpdate();
    yield();
}

/* ═══════════════════════════════════════════════════════
 *  LoRa Message Processing
 * ═══════════════════════════════════════════════════════ */
/**
 * @brief Process received LoRa message
 * @param packetSize Size of received packet
 */
void process_lora_message(int packetSize) {
    char buffer[256];
    int len = 0;
    while (LoRa.available() && len < 255) {
        buffer[len++] = (char)LoRa.read();
    }
    buffer[len] = '\0';

    int rssi = LoRa.packetRssi();
    Serial.printf("[LORA] Received %d bytes, RSSI: %d\n", len, rssi);

    siren_risk_update_t msg;
    if (!parse_risk_update(buffer, len, &msg)) {
        Serial.println("[SEC] Parse failed");
        lcdMgr.showLoRaRejected("Parse Failed");
        return;
    }

    if (strcmp(msg.zone_id, SIREN_ZONE_ID) != 0) {
        Serial.printf("[SEC] Zone mismatch: got '%s'\n", msg.zone_id);
        lcdMgr.showLoRaRejected("Zone Mismatch");
        return;
    }

    DateTime now = rtc.now();
    uint32_t curTime = now.unixtime();

    if (!siren_validate_message(&msg, last_sequence, curTime)) {
        Serial.println("[SEC] Validation failed");
        if ((curTime - msg.timestamp) > msg.ttl_seconds) {
            lcdMgr.showTTLExpired();
        } else if (msg.sequence_number <= last_sequence) {
            lcdMgr.showReplayDetected();
        } else {
            lcdMgr.showLoRaRejected("Validation Fail");
        }
        return;
    }

    lcdMgr.showLoRaAccepted(msg.sequence_number);
    last_sequence = msg.sequence_number;

    uint16_t cooldown_sec = 0;
    if (cooldown_until > curTime) {
        cooldown_sec = cooldown_until - curTime;
    }

    uint8_t rl_action = siren_fallback_mode(msg.risk_level, msg.risk_level, msg.breach_status);
    uint8_t aggression_risk_level = msg.aggression_flag ? 2 : 0;

    uint8_t mode = siren_safe_decision(
        rl_action,
        aggression_risk_level,
        msg.data_quality == 0 ? 0 : 2,
        msg.breach_status,
        msg.risk_level,
        cooldown_sec,
        activations_count,
        msg.aggression_flag
    );

    uint8_t conf_band = (msg.data_quality == 0) ? 0 : 2;
    lcdMgr.showDecision(rl_action, conf_band);

    if (mode != rl_action) {
        lcdMgr.showSafetyOverride(mode);
    }

    if (cooldown_sec > 0) {
        lcdMgr.showCooldown(cooldown_sec);
    }

    if (activations_count >= MAX_ACTIVATIONS_HOUR) {
        lcdMgr.showBudgetLimit();
    }

    execute_mode(mode, &msg);

    if (mode == SIREN_MODE_M1 || mode == SIREN_MODE_M2) {
        activations_count++;
        cooldown_until = now.unixtime() + DEFAULT_COOLDOWN_SEC;
    }

    LogCommand_t logCmd;
    logCmd.mode = mode;
    logCmd.rssi = rssi;
    logCmd.sequence = msg.sequence_number;
    strncpy(logCmd.boundary_id, msg.boundary_id, 63);
    logCmd.boundary_id[63] = '\0';
    xQueueSend(logQueue, &logCmd, 0);
}

/* ═══════════════════════════════════════════════════════
 *  Execute Mode
 * ═══════════════════════════════════════════════════════ */
/**
 * @brief Execute deterrent mode (audio, alert)
 * @param mode Siren mode
 * @param msg Risk update message
 */
void execute_mode(uint8_t mode, const siren_risk_update_t* msg) {
    current_mode = mode;
    lcdMgr.showMainStatus(mode, 0, activations_count);

    AudioCommand_t cmd;
    cmd.stop = false;

    switch (mode) {
        case SIREN_MODE_M0:
            Serial.println("[MODE] M0 — Observe");
            break;

        case SIREN_MODE_M1:
            Serial.println("[MODE] M1 — Bee deterrent");
            cmd.mode = SIREN_MODE_M1;
            cmd.repeat = 1;
            cmd.gap_ms = 500;
            xQueueSend(audioQueue, &cmd, pdMS_TO_TICKS(100));
            lcdMgr.showSoundPlaying(SIREN_MODE_M1);
            break;

        case SIREN_MODE_M2:
            Serial.println("[MODE] M2 — Leopard deterrent");
            cmd.mode = SIREN_MODE_M2;
            cmd.repeat = 2;
            cmd.gap_ms = 300;
            xQueueSend(audioQueue, &cmd, pdMS_TO_TICKS(100));
            lcdMgr.showSoundPlaying(SIREN_MODE_M2);
            break;

        case SIREN_MODE_M3:
            Serial.println("[MODE] M3 — SIREN ALERT");
            cmd.mode = SIREN_MODE_M3;
            cmd.repeat = 3;
            cmd.gap_ms = 200;
            xQueueSend(audioQueue, &cmd, pdMS_TO_TICKS(100));
            lcdMgr.showSoundPlaying(SIREN_MODE_M3);
            send_community_alert(msg);
            break;
    }
}

/* ═══ Community Alert ═══ */
/**
 * @brief Send community alert via LoRa
 * @param msg Risk update message
 */
void send_community_alert(const siren_risk_update_t* msg) {
    LoRa.beginPacket();
    LoRa.print("ALERT:");
    LoRa.print(msg->boundary_id);
    LoRa.endPacket();
    Serial.println("[ALERT] Community alert sent");
}

/* ═══ Parse Risk Update ═══ */
/**
 * @brief Parse risk update JSON message
 * @param data JSON string
 * @param len Length of data
 * @param msg Output: parsed message struct
 * @return true if parse successful, false otherwise
 */
bool parse_risk_update(const char* data, int len, siren_risk_update_t* msg) {
    if (len < 10) return false;
    memset(msg, 0, sizeof(siren_risk_update_t));

    cJSON* root = cJSON_Parse(data);
    if (!root) {
        Serial.println("[SEC] JSON parse failed");
        return false;
    }

    cJSON* zone     = cJSON_GetObjectItem(root, "zone_id");
    cJSON* boundary = cJSON_GetObjectItem(root, "boundary_segment_id");
    cJSON* risk     = cJSON_GetObjectItem(root, "risk_level");
    cJSON* breach   = cJSON_GetObjectItem(root, "breach_status");

    if (!cJSON_IsString(zone) || !cJSON_IsString(boundary) ||
        !cJSON_IsString(risk) || !cJSON_IsString(breach)) {
        Serial.println("[SEC] Missing required fields");
        cJSON_Delete(root);
        return false;
    }

    strncpy(msg->zone_id, zone->valuestring, sizeof(msg->zone_id) - 1);
    strncpy(msg->boundary_id, boundary->valuestring, sizeof(msg->boundary_id) - 1);

    const char* risk_str = risk->valuestring;
    if (strcmp(risk_str, "LOW") == 0)       msg->risk_level = 0;
    else if (strcmp(risk_str, "MED") == 0)  msg->risk_level = 1;
    else if (strcmp(risk_str, "HIGH") == 0) msg->risk_level = 2;
    else {
        cJSON_Delete(root);
        return false;
    }

    const char* breach_str = breach->valuestring;
    if (strcmp(breach_str, "NONE") == 0)           msg->breach_status = 0;
    else if (strcmp(breach_str, "LIKELY") == 0)    msg->breach_status = 1;
    else if (strcmp(breach_str, "CONFIRMED") == 0) msg->breach_status = 2;
    else {
        cJSON_Delete(root);
        return false;
    }

    cJSON* aggro = cJSON_GetObjectItem(root, "aggression_flag");
    msg->aggression_flag = (aggro && cJSON_IsTrue(aggro));

    cJSON* ttl = cJSON_GetObjectItem(root, "ttl_seconds");
    msg->ttl_seconds = (ttl && cJSON_IsNumber(ttl)) ? (uint16_t)ttl->valueint : 60;

    cJSON* seq = cJSON_GetObjectItem(root, "sequence_number");
    if (!seq || !cJSON_IsNumber(seq)) {
        cJSON_Delete(root);
        return false;
    }
    msg->sequence_number = (uint32_t)seq->valueint;

    cJSON* ts = cJSON_GetObjectItem(root, "timestamp_utc");
    if (!ts || !cJSON_IsNumber(ts)) {
        cJSON_Delete(root);
        return false;
    }
    msg->timestamp = (uint32_t)ts->valuedouble;

    cJSON* dist = cJSON_GetObjectItem(root, "distance_band");
    msg->distance_band = (dist && cJSON_IsNumber(dist)) ? (uint8_t)dist->valueint : 0;

    cJSON* qual = cJSON_GetObjectItem(root, "data_quality");
    msg->data_quality = (qual && cJSON_IsNumber(qual)) ? (uint8_t)qual->valueint : 1;

    cJSON* cksum = cJSON_GetObjectItem(root, "checksum");
    if (cksum && cJSON_IsNumber(cksum)) {
        msg->checksum = (uint16_t)cksum->valueint;
    }

    cJSON_Delete(root);
    Serial.printf("[PARSE] OK zone=%s risk=%d seq=%u\n",
                  msg->zone_id, msg->risk_level, msg->sequence_number);
    return true;
}
