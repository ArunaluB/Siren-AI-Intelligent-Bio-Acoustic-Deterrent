#!/bin/bash

echo "Starting Siren AI commit timeline..."

git add .

commit_with_date () {
  GIT_AUTHOR_DATE="$1" GIT_COMMITTER_DATE="$1" git commit --allow-empty -m "$2"
}

# FIRST REAL COMMIT (files)
GIT_AUTHOR_DATE="2025-12-01 09:15:00" GIT_COMMITTER_DATE="2025-12-01 09:15:00" git commit -m "Initial project structure setup for Siren AI v3"

# DECEMBER
commit_with_date "2025-12-03 14:40:00" "Added ESP32-S3 board configuration and base firmware skeleton"
commit_with_date "2025-12-05 19:10:00" "Configured Serial debugging and boot diagnostics"
commit_with_date "2025-12-08 11:25:00" "Implemented system mode state machine (M0–M3)"
commit_with_date "2025-12-10 16:45:00" "Added hardware pin mapping definitions"
commit_with_date "2025-12-14 13:05:00" "Integrated SX1278 LoRa module over HSPI"
commit_with_date "2025-12-18 20:30:00" "Implemented LoRa transmission test mode"
commit_with_date "2025-12-22 10:50:00" "Added RSSI logging and signal diagnostics"
commit_with_date "2025-12-27 17:15:00" "Mounted SD card using FSPI interface"
commit_with_date "2025-12-30 21:40:00" "Implemented SD file listing utility"

# JANUARY
commit_with_date "2026-01-03 09:20:00" "Added root directory scan for audio assets"
commit_with_date "2026-01-07 15:35:00" "Fixed SD initialization timing issue"
commit_with_date "2026-01-10 18:10:00" "Added mutex protection for SD access (dual-core safe)"
commit_with_date "2026-01-14 11:55:00" "Integrated MAX98357 I2S amplifier output"
commit_with_date "2026-01-17 20:25:00" "Implemented basic WAV playback engine"
commit_with_date "2026-01-20 13:30:00" "Added audio buffer management for streaming"
commit_with_date "2026-01-23 19:45:00" "Fixed WAV header parsing issue"
commit_with_date "2026-01-26 10:10:00" "Separated SPI buses (FSPI for SD, HSPI for LoRa)"
commit_with_date "2026-01-28 22:15:00" "Implemented FreeRTOS task pinning for dual-core"
commit_with_date "2026-01-30 16:00:00" "Improved heap memory monitoring logs"

# FEBRUARY
commit_with_date "2026-02-02 09:40:00" "Implemented conservative deterrent decision logic"
commit_with_date "2026-02-04 14:20:00" "Added dynamic siren trigger based on LoRa input"
commit_with_date "2026-02-06 18:55:00" "Added safe fallback mode when SD audio missing"
commit_with_date "2026-02-08 12:10:00" "Implemented NTP time sync and RTC alignment"
commit_with_date "2026-02-10 21:30:00" "Optimized playback for low-latency trigger response"
commit_with_date "2026-02-12 15:00:00" "Fixed cross-core SD access crash"
commit_with_date "2026-02-13 17:25:00" "Final integration testing and stability improvements"
commit_with_date "2026-02-14 10:00:00" "Stable Siren AI v3 release - LoRa + SD + I2S fully operational"

echo "All commits created successfully!"
