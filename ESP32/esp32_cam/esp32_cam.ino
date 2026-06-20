/*
  ╔═══════════════════════════════════════════════════════════════════╗
  ║   ParkIN — CAM MODULE   (AI-Thinker ESP32-CAM)              v5.0 ║
  ║   Architecture: STABLE & ROBUST (Non-blocking)                    ║
  ╚═══════════════════════════════════════════════════════════════════╝
*/

#include "esp_camera.h"
#include "esp_http_server.h"
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Preferences.h>
#include <time.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ════════════════════════════════════════════════════════════════════
//  1. NETWORK & SERVER CREDENTIALS
// ════════════════════════════════════════════════════════════════════
#define WIFI_SSID             "YOUR_WIFI_SSID"
#define WIFI_PASSWORD         "YOUR_WIFI_PASSWORD"
#define DEFAULT_SERVER_IP     "YOUR_SERVER_IP"

String server_ip = DEFAULT_SERVER_IP;
String OCR_ENDPOINT = "http://" + server_ip + ":5000/v1/plate-reader/";
String DETECTION_ENDPOINT = "http://" + server_ip + ":5000/detect-plate";
String QR_API_ENDPOINT = "http://" + server_ip + ":5000/v1/qr-reader/";
String SERVER_PING_URL = "http://" + server_ip + ":5000/";

WiFiUDP udp;
const unsigned int udpPort = 51234;
bool udpStarted = false;

// ════════════════════════════════════════════════════════════════════
//  2. DEV SERIAL Connection (Hardware UART2 - Pin 14, 15)
// ════════════════════════════════════════════════════════════════════
#define DEV_RX_PIN 15 
#define DEV_TX_PIN 14 
#define DEV_SERIAL Serial2
#define DEV_SERIAL_BAUD 9600 
#define DEV_SERIAL_PROTOCOL_START '['
#define DEV_SERIAL_PROTOCOL_END ']'

// ════════════════════════════════════════════════════════════════════
//  3. CAMERA PINS — AI-Thinker ESP32-CAM
// ════════════════════════════════════════════════════════════════════
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#define CMD_ENTRY "ENTRY"
#define CMD_EXIT "EXIT"

// ════════════════════════════════════════════════════════════════════
//  TUNING CONSTANTS
// ════════════════════════════════════════════════════════════════════
#define MIN_OCR_CONFIDENCE 45
#define HTTP_TIMEOUT_MS 30000 
#define QR_HTTP_TIMEOUT_MS 20000
#define SCAN_TIMEOUT_MS 30000
#define NETWORK_CHECK_INTERVAL 10000

// ════════════════════════════════════════════════════════════════════
//  GLOBAL STATE
// ════════════════════════════════════════════════════════════════════
Preferences preferences;
SemaphoreHandle_t serialMutex;
SemaphoreHandle_t camMutex;
unsigned long lastNetworkCheck = 0;
bool serverAvailable = false;
bool isScanning = false;

struct OcrResult {
  String plate;
  int confidence;
  String candidates;
  bool registered;
};

// ════════════════════════════════════════════════════════════════════
//  UTILITIES
// ════════════════════════════════════════════════════════════════════

void safePrint(const String &msg) {
  if (xSemaphoreTake(serialMutex, pdMS_TO_TICKS(100))) {
    Serial.println(msg);
    xSemaphoreGive(serialMutex);
  }
}

void updateServerIP(String ip) {
  server_ip = ip;
  OCR_ENDPOINT = "http://" + server_ip + ":5000/v1/plate-reader/";
  DETECTION_ENDPOINT = "http://" + server_ip + ":5000/detect-plate";
  QR_API_ENDPOINT = "http://" + server_ip + ":5000/v1/qr-reader/";
  SERVER_PING_URL = "http://" + server_ip + ":5000/";
  safePrint("[SYS] Endpoints dynamically updated to: " + server_ip);

  preferences.begin("server-cfg", false);
  preferences.putString("server_ip", server_ip);
  preferences.end();
}

void listenForServerIP() {
  if (WiFi.status() != WL_CONNECTED) return;

  if (!udpStarted) {
    udp.begin(udpPort);
    udpStarted = true;
    safePrint("[UDP] Listening for server IP beacon on port 51234...");
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
        if (newIP != server_ip && !newIP.isEmpty()) {
          safePrint("[UDP] Server IP Auto-Discovered: " + newIP);
          updateServerIP(newIP);
        }
      }
    }
  }
}

void sendToDev(const String &cmd) {
  DEV_SERIAL.print(DEV_SERIAL_PROTOCOL_START);
  DEV_SERIAL.print(cmd);
  DEV_SERIAL.println(DEV_SERIAL_PROTOCOL_END);
  safePrint("[→DEV UART] " + String(cmd));
}

// ════════════════════════════════════════════════════════════════════
//  CAMERA INIT
// ════════════════════════════════════════════════════════════════════
bool initCamera() {
  camera_config_t cfg;
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer = LEDC_TIMER_0;
  cfg.pin_d0 = Y2_GPIO_NUM;
  cfg.pin_d1 = Y3_GPIO_NUM;
  cfg.pin_d2 = Y4_GPIO_NUM;
  cfg.pin_d3 = Y5_GPIO_NUM;
  cfg.pin_d4 = Y6_GPIO_NUM;
  cfg.pin_d5 = Y7_GPIO_NUM;
  cfg.pin_d6 = Y8_GPIO_NUM;
  cfg.pin_d7 = Y9_GPIO_NUM;
  cfg.pin_xclk = XCLK_GPIO_NUM;
  cfg.pin_pclk = PCLK_GPIO_NUM;
  cfg.pin_vsync = VSYNC_GPIO_NUM;
  cfg.pin_href = HREF_GPIO_NUM;
  cfg.pin_sscb_sda = SIOD_GPIO_NUM;
  cfg.pin_sscb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn = PWDN_GPIO_NUM;
  cfg.pin_reset = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 20000000;
  cfg.pixel_format = PIXFORMAT_JPEG;
  cfg.frame_size = FRAMESIZE_CIF;  
  cfg.jpeg_quality = 12;           
  cfg.fb_count = 2;
  cfg.grab_mode = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&cfg) != ESP_OK) return false;
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, 1);
    s->set_hmirror(s, 0);
    s->set_contrast(s, 2);    
    s->set_brightness(s, 0);
    s->set_whitebal(s, 1);       
  }
  return true;
}

// Helper to read HTTP response body without hanging on Keep-Alive connections
String readHTTPResponseBody(WiFiClient &client) {
  int contentLength = -1;
  unsigned long timeout = millis();
  
  // Wait until bytes are available (up to 35 seconds for slow CPU inferences)
  while (!client.available() && client.connected()) {
    if (millis() - timeout > 35000) {
      safePrint("[HTTP] Wait timeout");
      return "";
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }

  // Parse headers
  while (client.connected() || client.available()) {
    String line = client.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) break; // End of headers
    
    if (line.startsWith("Content-Length:") || line.startsWith("content-length:")) {
      int colonIdx = line.indexOf(':');
      if (colonIdx != -1) {
        String lenStr = line.substring(colonIdx + 1);
        lenStr.trim();
        contentLength = lenStr.toInt();
      }
    }
  }

  // Read response body based on Content-Length
  String response = "";
  if (contentLength > 0) {
    response.reserve(contentLength);
    unsigned long bodyStart = millis();
    while (response.length() < contentLength && (client.connected() || client.available())) {
      if (millis() - bodyStart > 15000) { // 15s body read safety timeout
        safePrint("[HTTP] Body read timeout");
        break;
      }
      if (client.available()) {
        response += (char)client.read();
      } else {
        vTaskDelay(pdMS_TO_TICKS(5));
      }
    }
  } else {
    // Fallback if no Content-Length header was parsed
    response = client.readString();
  }
  return response;
}

OcrResult recognizePlate(camera_fb_t *fb) {
  OcrResult res = {"", 0, "", false};
  if (!fb || WiFi.status() != WL_CONNECTED) return res;

  WiFiClient client;
  client.setTimeout(HTTP_TIMEOUT_MS);  

  if (!client.connect(server_ip.c_str(), 5000)) {
    safePrint("[OCR] Server connection failed to " + server_ip);
    return res;
  }
  safePrint("[OCR] Connected! Sending image (" + String(fb->len) + " bytes)...");

  client.println("POST /v1/plate-reader/ HTTP/1.1");
  client.println("Host: " + server_ip);
  client.println("Content-Type: image/jpeg");
  client.print("Content-Length: ");
  client.println(fb->len);
  client.println("Connection: close");
  client.println();
  
  // Single, hardware-optimized zero-copy block stream write for maximum TCP transmission speed
  client.write(fb->buf, fb->len);

  String response = readHTTPResponseBody(client);
  client.stop();
  safePrint("[OCR] Response received!");

  DynamicJsonDocument doc(2048);
  DeserializationError error = deserializeJson(doc, response);

  if (error) {
    safePrint("[OCR] JSON Parsing Error: " + String(error.c_str()));
    safePrint("[OCR] Raw Response was: " + response);
  } else {
    JsonArray results = doc["results"].as<JsonArray>();
    String plateList = "";
    for (int i = 0; i < results.size() && i < 3; i++) {
      String p = results[i]["plate"].as<String>();
      int conf = (int)(results[i]["score"].as<float>() * 100);
      if (i == 0) {
        res.plate = p;
        res.confidence = conf;
        res.registered = results[i]["registered"] | false;
      }
      if (plateList != "") plateList += ",";
      plateList += p;
    }
    res.candidates = plateList;
    if (!res.plate.isEmpty()) {
      safePrint("[OCR] Result: " + res.plate + " (" + String(res.confidence) + "%) " + (res.registered ? "[REG]" : "[GUEST]"));
    } else {
      safePrint("[OCR] No plate detected");
    }
  }
  return res;
}

String decodeQRviaAPI(camera_fb_t *fb) {
  if (!fb || WiFi.status() != WL_CONNECTED) return "";

  WiFiClient client;
  client.setTimeout(QR_HTTP_TIMEOUT_MS);
  if (!client.connect(server_ip.c_str(), 5000)) {
    safePrint("[QR] Connection to server failed!");
    return "";
  }

  safePrint("[QR] Connected! Transmitting frame in stable chunks (" + String(fb->len) + " bytes)...");

  client.println("POST /v1/qr-reader/ HTTP/1.1");
  client.println("Host: " + server_ip);
  client.println("Content-Type: image/jpeg");
  client.print("Content-Length: ");
  client.println(fb->len);
  client.println("Connection: close");
  client.println();
  
  // High-reliability chunked transmission to prevent ESP32 Wi-Fi buffer overrun
  size_t size = fb->len;
  size_t offset = 0;
  while (offset < size) {
    size_t chunk = (size - offset > 1024) ? 1024 : (size - offset);
    client.write(fb->buf + offset, chunk);
    offset += chunk;
    vTaskDelay(pdMS_TO_TICKS(1)); // allow the Wi-Fi stack to process
  }

  safePrint("[QR] Transmission complete! Waiting for server decode response...");

  String response = readHTTPResponseBody(client);
  client.stop();

  safePrint("[QR] Server response received!");

  DynamicJsonDocument doc(1024);
  if (!deserializeJson(doc, response)) {
    JsonArray root = doc.as<JsonArray>();
    if (root.size() > 0) {
      String decoded = root[0]["symbol"][0]["data"].as<String>();
      safePrint("[QR] Decoded QR Content: " + decoded);
      return decoded;
    }
  }
  return "";
}

bool detectPlateWithAI(camera_fb_t *fb) {
  if (!fb || WiFi.status() != WL_CONNECTED) return false;

  WiFiClient client;
  client.setTimeout(10000); // 10s for fast detection

  if (!client.connect(server_ip.c_str(), 5000)) {
    return false;
  }

  client.println("POST /detect-plate HTTP/1.1");
  client.println("Host: " + server_ip);
  client.println("Content-Type: image/jpeg");
  client.print("Content-Length: ");
  client.println(fb->len);
  client.println("Connection: close");
  client.println();
  
  // Single, hardware-optimized zero-copy block stream write for maximum TCP transmission speed
  client.write(fb->buf, fb->len);

  String response = readHTTPResponseBody(client);
  client.stop();

  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, response);
  if (!error) {
    bool hasPlate = doc["has_plate"] | false;
    int count = doc["plates_detected"] | 0;
    if (hasPlate) {
      safePrint("[AI] Detection: " + String(count) + " plate(s) found");
    }
    return hasPlate;
  }
  return false;
}

void runEntryPipeline() {
  unsigned long start = millis();
  safePrint("[SCAN] Starting Entry Loop...");
  isScanning = true;
  String lastPlate = "";

  while (millis() - start < SCAN_TIMEOUT_MS) {
    camera_fb_t *fb = NULL;
    if (xSemaphoreTake(camMutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
      fb = esp_camera_fb_get();
      xSemaphoreGive(camMutex);
    }
    if (!fb) { vTaskDelay(pdMS_TO_TICKS(100)); continue; }

    OcrResult ocr = recognizePlate(fb);
    if (!ocr.plate.isEmpty()) {
      safePrint("[SCAN] Candidate: " + ocr.plate + " Conf: " + String(ocr.confidence));

      if (ocr.registered) {
         sendToDev(ocr.candidates.indexOf(',') != -1 ? "PLA_LIST:" + ocr.candidates : "PLATE:" + ocr.plate + ",100");
         esp_camera_fb_return(fb);
         isScanning = false;
         return;
      }
      if (ocr.confidence >= MIN_OCR_CONFIDENCE) {
        if (ocr.confidence >= 80 || ocr.plate == lastPlate) {
          sendToDev(ocr.candidates.indexOf(',') != -1 ? "PLA_LIST:" + ocr.candidates : "PLATE:" + ocr.plate + "," + String(ocr.confidence));
          esp_camera_fb_return(fb);
          isScanning = false;
          return;
        }
        lastPlate = ocr.plate;
      }
    } else {
      // safePrint("[SCAN] No plate found, skipping OCR");
    }
    esp_camera_fb_return(fb);
    vTaskDelay(pdMS_TO_TICKS(50));
  }
  sendToDev("ERR_TIMEOUT");
  isScanning = false;
}


void runExitPipeline() {
  unsigned long start = millis();
  safePrint("[SCAN] Starting Exit Loop...");
  isScanning = true;
  while (millis() - start < SCAN_TIMEOUT_MS) {
    camera_fb_t *fb = NULL;
    if (xSemaphoreTake(camMutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
      fb = esp_camera_fb_get();
      xSemaphoreGive(camMutex);
    }
    if (!fb) { vTaskDelay(pdMS_TO_TICKS(100)); continue; }

    String qrData = decodeQRviaAPI(fb);
    esp_camera_fb_return(fb);

    if (!qrData.isEmpty()) {
      // Reformat to include CAR: prefix if missing, satisfying Dev Board parser expectations
      if (qrData.indexOf("CAR:") == -1 && qrData.indexOf("ID:") != -1) {
        int idIdx = qrData.indexOf("ID:") + 3;
        String bookingId = qrData.substring(idIdx);
        bookingId.trim();
        qrData = "CAR:" + bookingId + "|ID:" + bookingId;
        safePrint("[QR] Reformatted for Dev Board: " + qrData);
      }
      sendToDev("QR_DATA:" + qrData);
      isScanning = false;
      return;
    }
    vTaskDelay(pdMS_TO_TICKS(500));
  }
  sendToDev("ERR_TIMEOUT");
  isScanning = false;
}

// ════════════════════════════════════════════════════════════════════
//  TASKS
// ════════════════════════════════════════════════════════════════════

void devCommsTask(void *pv) {
  String rxBuf = "";
  bool inFrame = false;
  while (true) {
    while (DEV_SERIAL.available()) {
      char c = DEV_SERIAL.read();
      if (c == DEV_SERIAL_PROTOCOL_START) { inFrame = true; rxBuf = ""; continue; }
      if (c == DEV_SERIAL_PROTOCOL_END) {
        inFrame = false; rxBuf.trim();
        if (rxBuf.startsWith("SET_IP:")) {
          server_ip = rxBuf.substring(7); server_ip.trim();
          OCR_ENDPOINT = "http://" + server_ip + ":5000/v1/plate-reader/";
          SERVER_PING_URL = "http://" + server_ip + ":5000/";
          safePrint("[SYS] IP Updated to: " + server_ip);
        }
        else {
          safePrint("[UART] Received: " + rxBuf);
          if (rxBuf == CMD_ENTRY) runEntryPipeline();
          else if (rxBuf == CMD_EXIT) runExitPipeline();
        }
        rxBuf = ""; continue;
      }
      if (inFrame && rxBuf.length() < 64) rxBuf += c;
    }
    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

void checkWiFi() {
  if (isScanning || millis() - lastNetworkCheck < NETWORK_CHECK_INTERVAL) return;
  lastNetworkCheck = millis();

  if (WiFi.status() != WL_CONNECTED) {
    safePrint("[WiFi] Connection lost. Reconnecting...");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  } else {
    WiFiClient client;
    client.setTimeout(2000); // Shorter timeout for background check
    if (client.connect(server_ip.c_str(), 5000)) {
      if (!serverAvailable) safePrint("[SYS] Server connection restored");
      serverAvailable = true;
    } else {
      if (serverAvailable) safePrint("[SYS] Server connection lost (Unreachable)");
      serverAvailable = false;
    }
    client.stop();
  }
}

void setup() {
  // STABILITY: Disable brownout detector to prevent reset loops during peak power draw (taking photos + transmitting via WiFi)
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  
  pinMode(32, OUTPUT); digitalWrite(32, LOW); // Wake camera
  pinMode(4, OUTPUT); digitalWrite(4, LOW);   // LED Off

  Serial.begin(115200);
  DEV_SERIAL.begin(DEV_SERIAL_BAUD, SERIAL_8N1, DEV_RX_PIN, DEV_TX_PIN);

  // Power stabilization delay
  delay(1000);

  serialMutex = xSemaphoreCreateMutex();
  camMutex = xSemaphoreCreateMutex();

  // Load saved server IP from flash memory (EEPROM)
  preferences.begin("server-cfg", true);
  String saved_ip = preferences.getString("server_ip", DEFAULT_SERVER_IP);
  preferences.end();
  updateServerIP(saved_ip);

  if (!initCamera()) { Serial.println("CAM INIT FAIL"); while (1); }

  WiFi.mode(WIFI_STA);
  // STABILITY: Disable modem sleep and limit Wi-Fi transmission power to prevent severe voltage dips
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) { delay(500); Serial.print("."); retries++; }
  Serial.println(WiFi.status() == WL_CONNECTED ? "\n[WiFi] OK" : "\n[WiFi] FAIL");

  xTaskCreatePinnedToCore(devCommsTask, "dev", 8192, NULL, 1, NULL, 1);
}

void loop() {
  checkWiFi();
  listenForServerIP();
  vTaskDelay(pdMS_TO_TICKS(1000));
}
