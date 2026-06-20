# Low-Cost ESP32-CAM Smart Parking ANPR System

This repository contains the complete implementation of the low-cost, resource-constrained Automated Number Plate Recognition (ANPR) smart parking system described in our research paper. The architecture uses a three-tier design to optimize compute, network bandwidth, and deployment cost.

## 📁 Repository Structure

*   `arduino/`
    *   `esp32_cam/` — Microcontroller code for the AI-Thinker ESP32-CAM (sensing, JPEG capture, Wi-Fi upload, and UDP server discovery).
    *   `esp32_devkit/` — Microcontroller code for the NodeMCU ESP32 DevKit v1 (peripheral controller for IR sensors, LCD, and SG90 servo gate).
*   `gateway_server/` — Lightweight x86/ARM gateway server running YOLOv8-nano (INT8), Hough-based deskewing, EasyOCR, and HSRP grammar correction.
*   `dataset/` — Contains the raw unzipped evaluation dataset (`ANPR_dataset/`), the zipped archive (`ANPR_dataset.zip`), and the `dataset.yaml` configuration.
*   `scripts/` — Evaluation, validation, and OpenVINO model compilation scripts.

---

## 🔌 Hardware Connections & Pins

The system uses a **Dual ESP32 Node** to separate image processing from peripheral control.

### UART Communication Link
*   **ESP32-CAM TX** (GPIO 14) $\rightarrow$ **ESP32 DevKit RX** (GPIO 16)
*   **ESP32-CAM RX** (GPIO 15) $\rightarrow$ **ESP32 DevKit TX** (GPIO 17)
*   Baud Rate: `9600` (Enclosed in protocol frames `[COMMAND]`)

### ESP32 DevKit v1 Peripherals
*   **Servo Motor (SG90):** GPIO 13 (PWM)
*   **LCD Display (16x2 I2C):** GPIO 21 (SDA), GPIO 22 (SCL)
*   **IR Slot Sensors (1–4):** GPIO 18, 14, 27, 26
*   **IR Gate Sensors (Entry/Exit):** GPIO 32, 33

---

## 🚀 Getting Started

### 1. Arduino Code Upload
1.  Open the Arduino IDE.
2.  Install the ESP32 board manager.
3.  Install the required libraries:
    *   `ArduinoJson`
    *   `ESP32Servo`
    *   `hd44780`
4.  Open `arduino/esp32_cam/esp32_cam.ino`, set your Wi-Fi credentials (`WIFI_SSID` / `WIFI_PASSWORD`), select **AI Thinker ESP32-CAM** as the board, and flash.
5.  Open `arduino/esp32_devkit/esp32_devkit.ino`, select **ESP32 Dev Module**, and flash.

### 2. Gateway Server Setup
1.  Navigate to `gateway_server/`.
2.  Install python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure Firebase:
    *   Download your Firebase Service Account JSON key.
    *   Place it in the directory and name it `credentials.json` (or update its path in `app.py`).
    *   Update `databaseURL` inside `app.py` with your database URL.
4.  Run the Flask API Server:
    ```bash
    python app.py
    ```

### 3. Dataset Configuration
*   The raw dataset is provided directly in `dataset/ANPR_dataset/` (split into `images/` and `labels/` subdirectories).
*   Alternatively, a zipped archive is available at `dataset/ANPR_dataset.zip`.
*   Use `dataset.yaml` to reference the train/val directories for YOLOv8 training.

### 4. Running Scripts
*   **Recreate Paper Figures:** Run `python scripts/generate_all_figures.py` to generate the high-resolution vector graphs shown in the paper.
*   **INT8 Quantization:** Use `python scripts/export_and_val_int8.py` to compile the standard ONNX model to an INT8 OpenVINO engine.

---

## 📊 Core Performance Metrics
*   **Detection Accuracy:** 96.50% mAP@50 (INT8 quantized YOLOv8-nano).
*   **End-to-End Latency:** 1,388 ms.
*   **OCR Accuracy:** 98.70% word-level accuracy (using context-aware state machine).
*   **Power Consumption:** 1.20 W peak, 1.27 Wh daily energy budget.
