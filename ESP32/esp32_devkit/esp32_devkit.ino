/*
  ╔═══════════════════════════════════════════════════════════════════╗
  ║   ParkIN — DEV MODULE (NodeMCU ESP32)                v6.0 (STABLE)   ║
  ║   Architecture: PROXY-BASED (Non-blocking logic)                  ║
  ╚═══════════════════════════════════════════════════════════════════╝
*/

#include "soc/rtc_cntl_reg.h"
#include "soc/soc.h"
#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <HTTPClient.h>
#include <Preferences.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>

// ════════════════════════════════════════════════════════════════════
//  PIN DEFINITIONS & GATE CONFIG
// ════════════════════════════════════════════════════════════════════
#define SERVO_PIN 13
#define IR_SLOT_1 18
#define IR_SLOT_2 14
#define IR_SLOT_3 27
#define IR_SLOT_4 26
#define IR_ENTRY 32
#define IR_EXIT 33

// Servo configuration for secure dual-lane barrier gate:
// Idle/Open position (Upward / Straight Up): 90 degrees
// Block Entry position (Downward to right): 180 degrees
// Block Exit position (Downward to left): 0 degrees
#define GATE_OPEN_ANGLE 90
#define GATE_BLOCK_ENTRY_ANGLE 180
#define GATE_BLOCK_EXIT_ANGLE 0

hd44780_I2Cexp lcd;

// UART Pins to communicate with ESP32-CAM
#define CAM_SERIAL_BAUD 9600
#define CAM_RX_PIN 16
#define CAM_TX_PIN 17
#define PROTOCOL_START '['
#define PROTOCOL_END ']'
#define CAM_SERIAL Serial2
#define MAX_CAM_BUF 128

// ════════════════════════════════════════════════════════════════════
//  CREDENTIALS & CONFIG
// ════════════════════════════════════════════════════════════════════
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#define DEFAULT_SERVER_IP "YOUR_SERVER_IP"
#define SERVER_PORT 5000
#define TOTAL_SLOTS 10
#define SENSOR_ACTIVE_STATE LOW

// ═══════════════════════════════restart the server
// ═════════════════════════════════════
//  GLOBAL STATE
// ════════════════════════════════════════════════════════════════════
String serverIP = DEFAULT_SERVER_IP;
bool serverReady = false;
Preferences preferences;

WiFiUDP udp;
const unsigned int udpPort = 51234;
bool udpStarted = false;

int slotsLeft = TOTAL_SLOTS;
int vehiclesParked = 0;

bool entryTriggered = false;
bool exitTriggered = false;
unsigned long triggerCooldown = 0;
int expectedSlot = 0;
unsigned long entryDetectedAt = 0;
Servo gateServo;

// Non-blocking Timers & Flags
unsigned long lastDisplayChange = 0;
unsigned long tempMessageExpiry = 0;
unsigned long gateCloseAt = 0;
unsigned long lastNetworkCheck = 0;
unsigned long lastSlotFetch = 0;
bool overrideDisplay = false;
int displayState = 0;
bool gateOpen = false;

struct BookingResult {
  bool found;
  bool paid;
  bool active;
  int slot;
  String name;
  unsigned long entryTime;
};

// ════════════════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════════════════
void setup() {
  // STABILITY: Disable brownout detector to prevent resets when the servo motor
  // actuates (draws high starting current)
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n[BOOT] ParkIN Master DEV Board - v6.0 (STABLE)");

  // Load saved server IP from flash memory (EEPROM)
  preferences.begin("server-cfg", true);
  serverIP = preferences.getString("server_ip", DEFAULT_SERVER_IP);
  preferences.end();
  Serial.println("[BOOT] Loaded server IP: " + serverIP);

  CAM_SERIAL.begin(CAM_SERIAL_BAUD, SERIAL_8N1, CAM_RX_PIN, CAM_TX_PIN);
  Wire.begin();
  lcd.begin(16, 2);
  lcd.backlight();
  lcd.print("ParkIN System");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");

  gateServo.attach(SERVO_PIN);
  gateServo.write(GATE_OPEN_ANGLE); // Default to upward/open state
  delay(500);
  gateServo.detach();

  pinMode(IR_SLOT_1, INPUT);
  pinMode(IR_SLOT_2, INPUT);
  pinMode(IR_SLOT_3, INPUT);
  pinMode(IR_SLOT_4, INPUT);
  pinMode(IR_ENTRY, INPUT);
  pinMode(IR_EXIT, INPUT);

  connectWiFi();

  if (WiFi.status() == WL_CONNECTED) {
    testServer();
    if (serverReady) {
      initializeSlotsIfMissing();
    }
  }

  lcd.clear();
  Serial.println("[SYS] Setup Complete.");
}

void connectWiFi() {
  preferences.begin("wifi-cfg", true); // Read-only
  String ssid = preferences.getString("ssid", WIFI_SSID);
  String pass = preferences.getString("pass", WIFI_PASSWORD);
  preferences.end();

  Serial.println("[WiFi] Connecting to: " + ssid);
  WiFi.begin(ssid.c_str(), pass.c_str());
  WiFi.setSleep(false); // STABILITY: Keep WiFi awake
}

void updateServerIP(String ip) {
  serverIP = ip;
  serverReady = false; // Trigger re-test
  Serial.println("[SYS] Server IP dynamically updated to: " + serverIP);

  preferences.begin("server-cfg", false);
  preferences.putString("server_ip", serverIP);
  preferences.end();
}

void listenForServerIP() {
  if (WiFi.status() != WL_CONNECTED)
    return;

  if (!udpStarted) {
    udp.begin(udpPort);
    udpStarted = true;
    Serial.println("[UDP] Listening for server IP beacon on port 51234...");
  }

  int packetSize = udp.parsePacket();
  if (packetSize) {
    char packetBuffer[64];
    int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
    if (len > 0) {
      packetBuffer[len] = '\0';
      String msg = String(packetBuffer);
      if (msg.startsWith("PARKIN_SERVER_IP:")) {
        String newIP = msg.substring(17);
        newIP.trim();
        if (newIP != serverIP && !newIP.isEmpty()) {
          Serial.println("[UDP] Server IP Auto-Discovered: " + newIP);
          updateServerIP(newIP);
        }
      }
    }
  }
}

void checkNetwork() {
  if (millis() - lastNetworkCheck < 10000)
    return;
  lastNetworkCheck = millis();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Lost connection. Reconnecting...");
    WiFi.disconnect();
    connectWiFi();
    serverReady = false;
  } else if (!serverReady) {
    testServer();
  }
}

void testServer() {
  HTTPClient http;
  String url = "http://" + serverIP + ":" + String(SERVER_PORT) + "/";
  http.begin(url);
  http.setTimeout(2000);
  int code = http.GET();
  if (code == 200) {
    serverReady = true;
    Serial.println("[SYS] Server OK");
  } else {
    serverReady = false;
    Serial.printf("[SYS] Server Check Failed: %d\n", code);
  }
  http.end();
}

// ════════════════════════════════════════════════════════════════════
//  WIFI FALLBACK BYPASS (Fail-Safe for Broken Serial Wires)
// ════════════════════════════════════════════════════════════════════
unsigned long lastProcessedTimestamp = 0;
unsigned long lastProcessedQrTimestamp = 0;

void pollLatestPlateAndQrFromServer() {
  if (WiFi.status() != WL_CONNECTED || !serverReady)
    return;

  static unsigned long lastPollTime = 0;
  if (millis() - lastPollTime < 800)
    return; // Poll every 800ms
  lastPollTime = millis();

  // 1. Fallback for Entry Gate (License Plates)
  if (digitalRead(IR_ENTRY) == LOW) {
    HTTPClient http;
    http.begin("http://" + serverIP + ":5000/v1/proxy/get-latest-plate");
    http.setTimeout(800);
    int code = http.GET();
    if (code == 200) {
      DynamicJsonDocument doc(256);
      DeserializationError err = deserializeJson(doc, http.getString());
      if (!err) {
        String plate = doc["plate"] | "";
        float conf = doc["confidence"] | 0.0;
        unsigned long ts = doc["timestamp"] | 0;

        if (!plate.isEmpty() && ts > lastProcessedTimestamp) {
          Serial.println(
              "[WIFI FALLBACK] New plate detected via Server HTTP polling: " +
              plate);
          lastProcessedTimestamp = ts;
          processPlateResult(plate + "," + String((int)(conf * 100)));
        }
      }
    }
    http.end();
  }

  // 2. Fallback for Exit Gate (QR Codes)
  if (digitalRead(IR_EXIT) == LOW) {
    HTTPClient http;
    http.begin("http://" + serverIP + ":5000/v1/proxy/get-latest-qr");
    http.setTimeout(800);
    int code = http.GET();
    if (code == 200) {
      DynamicJsonDocument doc(256);
      DeserializationError err = deserializeJson(doc, http.getString());
      if (!err) {
        String qrData = doc["data"] | "";
        unsigned long ts = doc["timestamp"] | 0;

        if (!qrData.isEmpty() && ts > lastProcessedQrTimestamp) {
          Serial.println(
              "[WIFI FALLBACK] New QR code decoded via Server HTTP polling: " +
              qrData);
          lastProcessedQrTimestamp = ts;
          processQrResult(qrData);
        }
      }
    }
    http.end();
  }
}

// ════════════════════════════════════════════════════════════════════
//  MAIN LOOP
// ════════════════════════════════════════════════════════════════════
void loop() {
  listenForServerIP();
  checkNetwork();
  updateParkingSlots();
  checkEntryExitGates();
  handleCamCommands();
  pollLatestPlateAndQrFromServer();
  monitorWrongSlot();
  handleTimers();
  updateDisplay();
}

void handleTimers() {
  // Clear temp message if expired
  if (overrideDisplay && millis() > tempMessageExpiry) {
    clearTempMessage();
  }

  // Close gate if timer expired
  if (gateOpen && millis() > gateCloseAt) {
    closeGate();
  }
}

// ════════════════════════════════════════════════════════════════════
//  CORE FUNCTIONS
// ════════════════════════════════════════════════════════════════════

void updateParkingSlots() {
  int s1 = (digitalRead(IR_SLOT_1) == SENSOR_ACTIVE_STATE) ? 1 : 0;
  int s2 = (digitalRead(IR_SLOT_2) == SENSOR_ACTIVE_STATE) ? 1 : 0;
  int s3 = (digitalRead(IR_SLOT_3) == SENSOR_ACTIVE_STATE) ? 1 : 0;
  int s4 = (digitalRead(IR_SLOT_4) == SENSOR_ACTIVE_STATE) ? 1 : 0;

  int localVehiclesParked = s1 + s2 + s3 + s4;
  int localSlotsLeft = TOTAL_SLOTS - localVehiclesParked;

  if (WiFi.status() == WL_CONNECTED && serverReady &&
      (millis() - lastSlotFetch >= 4000 || lastSlotFetch == 0)) {
    lastSlotFetch = millis();
    HTTPClient http;
    http.begin("http://" + serverIP + ":5000/v1/proxy/get-free-slots");
    http.setTimeout(2000);
    int code = http.GET();
    if (code == 200) {
      DynamicJsonDocument doc(512);
      DeserializationError err = deserializeJson(doc, http.getString());
      if (!err) {
        slotsLeft = doc["free_slots"] | localSlotsLeft;
        int total = doc["total_slots"] | TOTAL_SLOTS;
        vehiclesParked = total - slotsLeft;
        Serial.printf("[DB_SYNC] Free Slots: %d/%d (Sync SUCCESS)\n", slotsLeft,
                      total);
        http.end();
        return;
      }
    }
    http.end();
  }

  vehiclesParked = localVehiclesParked;
  slotsLeft = localSlotsLeft;
}

void checkEntryExitGates() {
  if (millis() - triggerCooldown < 8000)
    return;

  if (digitalRead(IR_ENTRY) == LOW && !entryTriggered) {
    Serial.println("[GATE] IR_ENTRY Triggered! Blocking Entry Gate...");

    // Move servo to block entry lane immediately (downward)
    gateServo.attach(SERVO_PIN);
    gateServo.write(GATE_BLOCK_ENTRY_ANGLE);
    delay(500); // Allow physical block to finish
    gateServo.detach();

    showTempMessage("Verifying Plate", "Please wait...", 30000); // 30s timeout
    CAM_SERIAL.print(PROTOCOL_START);
    CAM_SERIAL.print("ENTRY");
    CAM_SERIAL.println(PROTOCOL_END);
    triggerCooldown = millis();
    entryTriggered = true;
  } else if (digitalRead(IR_ENTRY) == HIGH) {
    entryTriggered = false;
  }

  if (digitalRead(IR_EXIT) == LOW && !exitTriggered) {
    Serial.println("[GATE] IR_EXIT Triggered! Blocking Exit Gate...");

    // Move servo to block exit lane immediately (downward)
    gateServo.attach(SERVO_PIN);
    gateServo.write(GATE_BLOCK_EXIT_ANGLE);
    delay(500); // Allow physical block to finish
    gateServo.detach();

    showTempMessage("Reading QR", "Please wait...", 30000);
    CAM_SERIAL.print(PROTOCOL_START);
    CAM_SERIAL.print("EXIT");
    CAM_SERIAL.println(PROTOCOL_END);
    triggerCooldown = millis();
    exitTriggered = true;
  } else if (digitalRead(IR_EXIT) == HIGH) {
    exitTriggered = false;
  }
}

String camBuf = "";
bool inFrame = false;
void handleCamCommands() {
  while (CAM_SERIAL.available() > 0) {
    char c = CAM_SERIAL.read();
    if (c == PROTOCOL_START) {
      inFrame = true;
      camBuf = "";
      continue;
    }
    if (c == PROTOCOL_END) {
      inFrame = false;
      camBuf.trim();
      Serial.println("[UART] Received from CAM: " + camBuf);
      if (camBuf.startsWith("PLA_LIST:"))
        processMultiPlateResult(camBuf.substring(9));
      else if (camBuf.startsWith("PLATE:"))
        processPlateResult(camBuf.substring(6));
      else if (camBuf.startsWith("QR_DATA:"))
        processQrResult(camBuf.substring(8));
      else if (camBuf.startsWith("SET_IP:")) {
        serverIP = camBuf.substring(7);
        serverReady = false;
        testServer();
      } else if (camBuf.startsWith("ERR_TIMEOUT")) {
        showTempMessage("Server Timeout", "Try Again", 3000);
      } else if (camBuf.startsWith("ERR_HTTP")) {
        showTempMessage("Server Error", "Check WiFi/IP", 3000);
      }
      camBuf = "";
      continue;
    }
    if (inFrame) {
      camBuf += c;
      if (camBuf.length() > MAX_CAM_BUF) { // Safety limit
        camBuf = "";
        inFrame = false;
      }
    }
  }
}

void processMultiPlateResult(String csv) {
  int start = 0, comma = csv.indexOf(',');
  while (comma != -1 || start < csv.length()) {
    String p =
        (comma != -1) ? csv.substring(start, comma) : csv.substring(start);
    p.trim();
    if (p.length() > 0) {
      BookingResult bk = checkBooking(p);
      if (bk.found && bk.paid) {
        processPlateResult(p + ",100");
        return;
      }
    }
    if (comma == -1)
      break;
    start = comma + 1;
    comma = csv.indexOf(',', start);
  }
  showTempMessage("NO BOOKING", "Check Details", 3000);
}

void processPlateResult(String data) {
  int commaIdx = data.indexOf(',');
  String plate = (commaIdx > 0) ? data.substring(0, commaIdx) : data;
  float confidence =
      (commaIdx > 0) ? (data.substring(commaIdx + 1).toFloat() / 100.0) : 0.85;

  showTempMessage("Verifying...", plate, 5000);
  BookingResult bk = checkBooking(plate);
  bool authorized = bk.found && bk.paid;

  if (serverReady) {
    HTTPClient http;
    http.begin("http://" + serverIP + ":5000/v1/proxy/latest-plate");
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(2000);
    String payload =
        "{\"plate\":\"" + plate + "\",\"confidence\":" + String(confidence) +
        ",\"isAuthorized\":" + String(authorized ? "true" : "false") + "}";
    http.POST(payload);
    http.end();
  }

  if (authorized) {
    int slot = (bk.slot > 0) ? bk.slot : (vehiclesParked + 1);
    showTempMessage("GRANTED S" + String(slot), "Welcome " + bk.name, 4000);
    writeEntryEvent(plate, bk.name, slot);
    openGate(GATE_OPEN_ANGLE); // Opens upward
  } else {
    showTempMessage("ACCESS DENIED", "No Booking Found", 4000);
  }
}

void processQrResult(String data) {
  int pStart = data.indexOf("CAR:") + 4;
  if (pStart < 4)
    return;
  int pEnd = data.indexOf("|", pStart);
  if (pEnd == -1)
    pEnd = data.length();
  String plate = data.substring(pStart, pEnd);

  showTempMessage("Verifying QR", plate, 3000);
  writeExitEvent(plate);
  showTempMessage("EXIT GRANTED", "Safe Journey", 4000);
  openGate(GATE_OPEN_ANGLE); // Opens upward
}

BookingResult checkBooking(String plate) {
  BookingResult b = {false, false, false, -1, ""};
  if (!serverReady)
    return b;

  HTTPClient http;
  http.begin("http://" + serverIP +
             ":5000/v1/proxy/check-booking?plate=" + plate);
  http.setTimeout(3000);

  if (http.GET() == 200) {
    DynamicJsonDocument doc(1024);
    DeserializationError err = deserializeJson(doc, http.getString());
    if (!err) {
      b.found = doc["found"] | false;
      if (b.found) {
        b.paid = doc["paid"] | false;
        b.active = doc["active"] | false;
        b.slot = doc["slot"] | -1;
        b.name = doc["name"].as<String>();
      }
    }
  }
  http.end();
  return b;
}

void writeEntryEvent(String plate, String name, int slot) {
  if (!serverReady)
    return;
  HTTPClient http;
  http.begin("http://" + serverIP + ":5000/v1/proxy/log-entry");
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);
  http.POST("{\"plate\":\"" + plate + "\",\"name\":\"" + name +
            "\",\"slot\":" + String(slot) + "}");
  http.end();
  expectedSlot = slot;
  entryDetectedAt = millis();
}

void writeExitEvent(String plate) {
  if (!serverReady)
    return;
  HTTPClient http;
  http.begin("http://" + serverIP + ":5000/v1/proxy/log-exit");
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);
  http.POST("{\"plate\":\"" + plate + "\"}");
  http.end();
}

void monitorWrongSlot() {
  if (expectedSlot <= 0 || millis() - entryDetectedAt > 45000) {
    expectedSlot = 0;
    return;
  }
  int sensors[4] = {IR_SLOT_1, IR_SLOT_2, IR_SLOT_3, IR_SLOT_4};
  for (int i = 0; i < 4; i++) {
    if (digitalRead(sensors[i]) == LOW) {
      if ((i + 1) != expectedSlot) {
        HTTPClient http;
        http.begin("http://" + serverIP + ":5000/v1/proxy/log-alert");
        http.addHeader("Content-Type", "application/json");
        http.setTimeout(2000);
        http.POST("{\"type\":\"WRONG_SLOT\",\"expected\":" +
                  String(expectedSlot) + ",\"actual\":" + String(i + 1) + "}");
        http.end();
      }
      expectedSlot = 0;
      break;
    }
  }
}

void openGate(int targetAngle) {
  Serial.printf("[GATE] Opening to angle %d...\n", targetAngle);
  gateServo.attach(SERVO_PIN);
  gateServo.write(targetAngle);
  gateOpen = true;
  gateCloseAt = millis() + 4000; // Keep open for 4s
}

void closeGate() {
  Serial.println("[GATE] Returning to Idle Upward state...");
  gateServo.attach(SERVO_PIN);
  gateServo.write(GATE_OPEN_ANGLE); // Rotate upward to idle open
  delay(500);                       // Small blocking for final move is okay
  gateServo.detach();
  gateOpen = false;
}

void initializeSlotsIfMissing() {
  HTTPClient http;
  http.begin("http://" + serverIP + ":5000/v1/proxy/init-slots");
  http.setTimeout(3000);
  http.POST("");
  http.end();
}

void showTempMessage(String l1, String l2, unsigned long duration) {
  overrideDisplay = true;
  tempMessageExpiry = millis() + duration;
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(l1);
  lcd.setCursor(0, 1);
  lcd.print(l2);
}

void clearTempMessage() {
  overrideDisplay = false;
  lcd.clear();
}

void updateDisplay() {
  if (overrideDisplay)
    return;
  if (millis() - lastDisplayChange >= 5000) {
    lastDisplayChange = millis();
    displayState = (displayState + 1) % 3;
    lcd.clear();
    if (displayState == 0) {
      lcd.print("Parked: " + String(vehiclesParked));
      lcd.setCursor(0, 1);
      lcd.print("Slots: " + String(slotsLeft));
    } else if (displayState == 1) {
      lcd.print("Visit Website:");
      lcd.setCursor(0, 1);
      lcd.print("parkin.vercel.app");
    } else {
      lcd.print("Creator:");
      lcd.setCursor(0, 1);
      lcd.print("SIDDHARTH");
    }
  }
}
