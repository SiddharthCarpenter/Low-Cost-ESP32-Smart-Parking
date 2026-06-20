from flask import Flask, request, jsonify, send_from_directory, Response
import cv2
import numpy as np
import re
import time
import socket
import os
import uuid
import functools
import sys
import base64
import io
from PIL import Image
from collections import defaultdict
import torch
# Monkeypatch torch.load to fix weights_only issue in PyTorch 2.6+
orig_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO
import easyocr

import json
import urllib.request
import urllib.parse

firebase_enabled = False
db = None
firebase_token = None

def get_firebase_token_fallback():
    email = "YOUR_EMAIL@example.com"
    password = "YOUR_PASSWORD"
    api_key = "YOUR_FIREBASE_API_KEY"
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    try:
        data = json.dumps({"email": email, "password": password, "returnSecureToken": True}).encode()
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode()).get("idToken")
    except Exception as e:
        print(f"⚠️  Firebase Auth Fallback Failed: {e}")
        return None

try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
    if os.path.exists('serviceAccountKey.json'):
        if not firebase_admin._apps:
            cred = credentials.Certificate('serviceAccountKey.json')
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://YOUR-PROJECT-ID-default-rtdb.firebaseio.com/'
            })
        db = firebase_db
        firebase_enabled = True
        print("Firebase Neural Sync: ACTIVE (Admin SDK)")
    else:
        print("Firebase Service Account Key missing. Using REST API fallback...")
        firebase_token = get_firebase_token_fallback()
        if firebase_token:
            firebase_enabled = True
            print("Firebase Neural Sync: ACTIVE (REST Fallback)")
except Exception as e:
    print(f"Firebase Init Failed: {e}")

app = Flask(__name__)

@app.before_request
def before_request():
    print(f"📥 [{time.strftime('%H:%M:%S')}] {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ─── Security & Rate Limiting ──────────────────────────────────────────────────
API_KEY = "parkin_secure_2026"
request_counts = defaultdict(list)

def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key') or request.args.get('key')
        if key != API_KEY:
            # For now, we'll just log it to avoid breaking hardware immediately
            # In a real production app, we would return 401
            print(f"⚠️  Missing or invalid API key from {request.remote_addr}")
        return f(*args, **kwargs)
    return decorated

def rate_limit(limit=10, window=60):
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            now = time.time()
            ip = request.remote_addr
            request_counts[ip] = [t for t in request_counts[ip] if now - t < window]
            if len(request_counts[ip]) >= limit:
                return jsonify({"error": "Rate limit exceeded", "retry_after": int(window - (now - request_counts[ip][0]))}), 429
            request_counts[ip].append(now)
            return f(*args, **kwargs)
        return decorated
    return decorator

@app.route('/')
def health_check():
    import psutil
    return jsonify({
        "status": "alive",
        "timestamp": int(time.time()),
        "ip": request.remote_addr,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "models_loaded": plate_detector is not None and ocr_engine is not None
        }
    })

# ─── Model Loading ────────────────────────────────────────────────────────────
print("🔄 Loading YOLOv11 license plate detector...")

# Export YOLOv11 to ONNX for 3-5x faster inference on CPU
onnx_path = r'C:\Users\hp\OneDrive\Desktop\PARR\runs\detect\yolov8_anpr_custom2\weights\last_int8_openvino_model'
if not os.path.exists(onnx_path):
    print("🔄 Exporting YOLOv11 model to ONNX format for optimized CPU execution...")
    try:
        temp_model = YOLO('license-plate-finetune-v1n.pt')
        temp_model.export(format='onnx', imgsz=320, dynamic=True)
        print("✅ ONNX export complete!")
    except Exception as e:
        print(f"⚠️  ONNX export failed, falling back to PyTorch model: {e}")

if os.path.exists(onnx_path):
    print("⚡ Loading optimized OpenVINO INT8 YOLO model...")
    plate_detector = YOLO(onnx_path, task='detect')
else:
    print("⚠️  Using standard PyTorch YOLO model...")
    plate_detector = YOLO('license-plate-finetune-v1n.pt')

# 🔄 Loading EasyOCR engine with Alphanumeric Allowlist for SPEED
# Auto-enable GPU if CUDA is available, otherwise default to False
gpu_available = torch.cuda.is_available()
print(f"🧠 Torch CUDA available: {gpu_available}")
ocr_engine = easyocr.Reader(['en'], gpu=gpu_available)
OCR_ALLOWLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# print("🔄 Loading ESPCN Super-Resolution model...")
# try:
#     sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
#     sr_model.readModel('ESPCN_x4.pb')
#     sr_model.setModel('espcn', 4)
#     print("✅ Super-resolution ready")
# except Exception as e:
#     print(f"⚠️  SR model failed to load: {e}")
sr_model = None

# FIX: Pre-warm EasyOCR by running a dummy inference at startup.
# Without this, the first real readtext() call lazy-loads the LSTM model
# (~30-50MB), freezing Flask for 10-30s and causing ESP32 TCP timeouts.
print("🔄 Pre-warming EasyOCR engine (first inference loads LSTM model)...")
try:
    _dummy_img = np.zeros((50, 200, 3), dtype=np.uint8)
    cv2.putText(_dummy_img, "TEST123", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ocr_engine.readtext(_dummy_img, allowlist=OCR_ALLOWLIST)
    print("✅ EasyOCR warm-up complete!")
except Exception as e:
    print(f"⚠️  EasyOCR warm-up failed (non-critical): {e}")

print("✅ All models loaded!\n")

# ─── State Code Mapping ───────────────────────────────────────────────────────
INDIAN_STATES = {
    'AP','AR','AS','BR','CG','GA','GJ','HR','HP','JH','JK','KA','KL','LA',
    'MP','MH','MN','ML','MZ','NL','OD','PB','RJ','SK','TN','TS','TR','UP',
    'UK','WB','AN','CH','DN','DD','DL','LD','PY'
}

TO_LETTERS = {'0':'O','1':'I','8':'B','5':'S','6':'G','2':'Z','4':'A','7':'T','3':'E','9':'P'}
TO_NUMBERS = {'O':'0','I':'1','B':'8','S':'5','G':'6','Z':'2','L':'1','A':'4','T':'7','E':'3','X':'1','C':'0','D':'0','Q':'0','R':'4'}

# ─── Temporal Consensus (Sliding Window) ─────────────────────────────────────
plate_history = []
CONSENSUS_WINDOW = 3.0

def add_to_consensus(plate, confidence):
    now = time.time()
    plate_history.append((now, plate, confidence))
    plate_history[:] = [(t, p, c) for t, p, c in plate_history
                        if now - t <= CONSENSUS_WINDOW][-10:]

def get_consensus_best():
    if not plate_history:
        return None, 0
    scores = defaultdict(float)
    for _, plate, conf in plate_history:
        scores[plate] += conf
    best = max(scores.items(), key=lambda x: x[1])
    return best[0], best[1]

# ─── Firebase Database Check with Thread-Safe Cache ───────────────────────────
import threading
bookings_cache = {}
bookings_original_keys = {}
last_cache_time = 0.0
cache_lock = threading.Lock()

# ─── MJPEG Video Stream Server (Flask-Hosted, zero load on ESP32-CAM) ─────────
latest_frame_lock = threading.Lock()
try:
    placeholder_img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(placeholder_img, "ParkIN Live Feed", (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 30, 78), 2)
    cv2.putText(placeholder_img, "SYSTEM STANDBY - READY FOR SCANS", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    _, placeholder_bytes = cv2.imencode('.jpg', placeholder_img)
    latest_frame_bytes = placeholder_bytes.tobytes()
except Exception:
    latest_frame_bytes = b""

# ─── Latest Scan Fallback State (Thread-Safe WiFi Fallback Bypass) ────────────
latest_scan_lock = threading.Lock()
latest_scan = {
    "plate": "",
    "confidence": 0.0,
    "timestamp": 0
}

latest_qr_lock = threading.Lock()
latest_qr = {
    "data": "",
    "timestamp": 0
}

# Thread lock for pyzbar (ensures safety when called from multiple concurrent Waitress threads)
zbar_lock = threading.Lock()

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ─── Automatic UDP Server Discovery Beacon ────────────────────────────────────
def run_udp_discovery_beacon():
    print("📡 [UDP DISCOVERY] Starting server broadcast beacon on port 51234...", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while True:
        try:
            ip = get_local_ip()
            if ip != "127.0.0.1":
                msg = f"PARKIN_SERVER_IP:{ip}".encode('utf-8')
                sock.sendto(msg, ('255.255.255.255', 51234))
        except Exception as e:
            print(f"⚠️ [UDP DISCOVERY] Broadcast Error: {e}", flush=True)
        time.sleep(3) # Broadcast every 3 seconds

threading.Thread(target=run_udp_discovery_beacon, daemon=True).start()

system_fully_online = False

def update_bookings_cache_loop():
    global bookings_cache, bookings_original_keys, last_cache_time, firebase_token, system_fully_online
    # Give models a brief moment to initialize before first database poll
    time.sleep(0.5)
    
    # Broadcast stream url on startup
    try:
        local_ip = get_local_ip()
        stream_url = f"http://{local_ip}:5000/v1/proxy/stream"
        print(f"📡 [UPLINK] Stream server active at {stream_url}. Syncing with Firebase...")
        if db is not None:
            db.reference('/anpr/stream_url').set(stream_url)
        else:
            if not firebase_token:
                firebase_token = get_firebase_token_fallback()
            db_url = "https://YOUR-PROJECT-ID-default-rtdb.firebaseio.com"
            write_url = f"{db_url}/anpr/stream_url.json?auth={firebase_token}"
            req = urllib.request.Request(write_url, data=json.dumps(stream_url).encode(), method='PUT')
            urllib.request.urlopen(req)
        print("📡 [UPLINK] Firebase stream sync success!")
    except Exception as e:
        print(f"  ⚠️  Stream Sync Error: {e}", flush=True)

    while True:
        try:
            if not firebase_enabled:
                if not system_fully_online:
                    system_fully_online = True
                    print("\n✅ [SYSTEM READY] All models, Firebase caches, and network proxies are fully online and verified. Ready for detection!\n", flush=True)
                time.sleep(5)
                continue
                
            # Option 1: Admin SDK
            if db is not None:
                try:
                    ref = db.reference('/bookings')
                    data = ref.get()
                    if isinstance(data, dict):
                        new_cache = {}
                        new_keys = {}
                        for k, v in data.items():
                            if v:
                                clean = re.sub(r'[^A-Z0-9]', '', k.upper())
                                new_cache[clean] = v
                                new_keys[clean] = k
                        with cache_lock:
                            bookings_cache = new_cache
                            bookings_original_keys = new_keys
                        last_cache_time = time.time()
                        if not system_fully_online:
                            system_fully_online = True
                            print("\n✅ [SYSTEM READY] All models, Firebase caches, and network proxies are fully online and verified. Ready for detection!\n", flush=True)
                        print(f"🔄 [Background Cache] Loaded {len(bookings_cache)} active bookings via Admin SDK.", flush=True)
                    elif data is None:
                        with cache_lock:
                            bookings_cache = {}
                            bookings_original_keys = {}
                        last_cache_time = time.time()
                        if not system_fully_online:
                            system_fully_online = True
                            print("\n✅ [SYSTEM READY] All models, Firebase caches, and network proxies are fully online and verified. Ready for detection!\n", flush=True)
                except Exception as e:
                    print(f"  ⚠️ [Background Cache] Admin Fetch Error: {e}", flush=True)
                    
            # Option 2: REST Fallback
            else:
                try:
                    if not firebase_token:
                        firebase_token = get_firebase_token_fallback()
                    
                    db_url = "https://YOUR-PROJECT-ID-default-rtdb.firebaseio.com"
                    url = f"{db_url}/bookings.json?auth={firebase_token}"
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req) as resp:
                        data = json.loads(resp.read().decode())
                        if isinstance(data, dict):
                            new_cache = {}
                            new_keys = {}
                            for k, v in data.items():
                                if v:
                                    clean = re.sub(r'[^A-Z0-9]', '', k.upper())
                                    new_cache[clean] = v
                                    new_keys[clean] = k
                            with cache_lock:
                                bookings_cache = new_cache
                                bookings_original_keys = new_keys
                            last_cache_time = time.time()
                            if not system_fully_online:
                                system_fully_online = True
                                print("\n✅ [SYSTEM READY] All models, Firebase caches, and network proxies are fully online and verified. Ready for detection!\n", flush=True)
                            print(f"🔄 [Background Cache] Loaded {len(bookings_cache)} active bookings via REST Fallback.", flush=True)
                        elif data is None:
                            with cache_lock:
                                bookings_cache = {}
                                bookings_original_keys = {}
                            last_cache_time = time.time()
                            if not system_fully_online:
                                system_fully_online = True
                                print("\n✅ [SYSTEM READY] All models, Firebase caches, and network proxies are fully online and verified. Ready for detection!\n", flush=True)
                except Exception as e:
                    print(f"  ⚠️ [Background Cache] REST Fetch Error: {e}", flush=True)
                    firebase_token = None
        except Exception as ex:
            print(f"  ⚠️ [Background Cache] Thread Loop Error: {ex}", flush=True)
            
        time.sleep(10) # Refresh every 10 seconds asynchronously

# Launch background cache updater immediately
threading.Thread(target=update_bookings_cache_loop, daemon=True).start()

def find_registered_match(plate):
    clean_plate = re.sub(r'[^A-Z0-9]', '', plate.upper())
    
    def norm(s):
        trans = str.maketrans('OIBASZ', '018452')
        return s.translate(trans)
        
    norm_candidate = norm(clean_plate)
    
    with cache_lock:
        if clean_plate in bookings_cache:
            return clean_plate
        for registered_key in bookings_cache:
            norm_reg = norm(registered_key)
            if len(norm_reg) >= 6 and norm_reg in norm_candidate:
                return registered_key
            if len(norm_candidate) >= 6 and norm_candidate in norm_reg:
                return registered_key
    return None

def check_database_for_plate(plate):
    """Check if plate exists in Firebase bookings (Fully non-blocking, substring-aware lookup)."""
    return find_registered_match(plate) is not None

def get_fuzzy_variations(plate):
    """Generate likely variations of a plate based on common OCR confusions."""
    confusions = {
        '1': ['4', 'I', 'J', 'L'], '4': ['1', 'A', 'L'], 'I': ['1', 'J', 'T'], 'J': ['1', 'I'], 'L': ['1', '4'],
        '0': ['D', 'O', 'Q'], 'D': ['0', 'O', 'Q'], 'O': ['0', 'D', 'Q'], 'Q': ['0', 'D', 'O'],
        '8': ['B', 'G', '0'], 'B': ['8', '6', 'D'], 'G': ['8', '6'], '6': ['G', '5'],
        'S': ['5', '2'], '5': ['S', '6'],
        'Z': ['2', '7'], '2': ['Z', '5'], '7': ['T', 'Z'], 'T': ['7', 'I'],
        'A': ['4', 'H'], 'H': ['A', 'M', 'N'], 'M': ['N', 'H'], 'N': ['M', 'H'],
        'U': ['V', 'Y'], 'V': ['U', 'Y'], 'Y': ['V', 'U']
    }
    variations = set()
    plate_list = list(plate)
    
    # Try single-character swaps
    for i, char in enumerate(plate_list):
        if char in confusions:
            for alt in confusions[char]:
                new_plate = plate[:i] + alt + plate[i+1:]
                variations.add(new_plate)
            
    return variations

# ─── Plate Correction & Validation ───────────────────────────────────────────
def is_hsrp_format(text):
    """Validates Indian HSRP format: SS DD LL NNNN (Supports both 1 and 2 series letters)"""
    return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', text))

def correct_indian_plate(raw):
    """Correct common OCR errors for Indian plates (Supports 9 and 10 char HSRP)."""
    plate = re.sub(r'[^A-Z0-9]', '', raw.upper())
    # Strip blue IND security strip text if OCR captured it
    plate = re.sub(r'^(IND|1ND|IIND|ND|IN)', '', plate)

    # Remove screw-hole artifact (11 → 10)
    if len(plate) == 11:
        plate = plate[:10]

    # Indian HSRP can be 9 characters (single series letter e.g., TN33J1364) or 10 characters (double e.g., HR98AA0000)
    if len(plate) != 9 and len(plate) != 10:
        return plate

    # Segment: SS(0:2) DD(2:4) LL(4:-4) NNNN(-4:)
    state    = ''.join(TO_LETTERS.get(c, c) for c in plate[:2])
    district = ''.join(TO_NUMBERS.get(c, c) for c in plate[2:4])
    number   = ''.join(TO_NUMBERS.get(c, c) for c in plate[-4:])
    series   = ''.join(TO_LETTERS.get(c, c) for c in plate[4:-4])

    corrected = f"{state}{district}{series}{number}"

    # State code validation — try first-letter match
    if state not in INDIAN_STATES:
        # Common confusion: M/H/N at the start
        state_raw = plate[:2]
        if 'H' in state_raw or 'N' in state_raw:
            # Check if it looks like Maharashtra (MH) or Madhya Pradesh (MP)
            if 'M' in state_raw or state_raw.startswith('H'):
                # Try correcting to common states
                for s in ['MH', 'MP', 'HR', 'HP']:
                    if s[0] == state[0] or s[1] == state[1]:
                        corrected = s + corrected[2:]
                        state = s
                        break

        # Fallback first-letter match
        if state not in INDIAN_STATES:
            for valid_state in INDIAN_STATES:
                if valid_state[0] == state[0]:
                    corrected = valid_state + corrected[2:]
                    state = valid_state
                    break

    # M/N/H confusion fix for MP plates (check the raw state chars only)
    state_raw = plate[:2]
    if state in ('HP', 'HR', 'MH') and ('M' in state_raw or 'N' in state_raw):
        corrected = 'MP' + corrected[2:]

    return corrected

# ─── Frame Quality Assessment ─────────────────────────────────────────────────
def assess_frame_quality(img):
    """
    Calculate frame sharpness using Laplacian variance.
    Returns: (quality_score, is_acceptable)
    >100 = sharp, 50-100 = acceptable, <50 = blurry (skip)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_acceptable = laplacian_var > 50
    return laplacian_var, is_acceptable

# ─── Pre-Detection Upscaling ──────────────────────────────────────────────────
def upscale_for_detection(img, target_width=640):
    """
    Upscale entire frame BEFORE YOLO detection to help with distant plates.
    - width < 640      → AI Super-Resolution (ESPCN), fallback to Lanczos
    - 640 <= w < target → Lanczos to target_width
    - width >= target  → return as-is
    """
    h, w = img.shape[:2]

    if w >= target_width:
        return img

# if w < 640 and sr_model is not None:
#     try:
#         upscaled = sr_model.upsample(img)
#         print(f"  📈 SR Upscale: {w}x{h} → {upscaled.shape[1]}x{upscaled.shape[0]}")
#         return upscaled
#     except Exception as e:
#         print(f"  ⚠️  SR failed: {e}")

    scale    = target_width / w
    new_h    = int(h * scale)
    upscaled = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_LINEAR) # Linear is faster than Lanczos for detection
    print(f"  📈 Quick Upscale: {w}x{h} → {target_width}x{new_h}")
    return upscaled

def deskew_hough(crop):
    """
    Straighten the license plate crop using the Hough Line Transform.
    Detects dominant horizontal lines and rotates the image to align them perfectly.
    """
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 30)
        
        if lines is None:
            return crop
            
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = theta * 180.0 / np.pi
            if 70.0 <= angle_deg <= 110.0:
                skew = angle_deg - 90.0
                angles.append(skew)
                
        if not angles:
            return crop
            
        median_skew = np.median(angles)
        if abs(median_skew) < 1.0 or abs(median_skew) > 25.0:
            return crop
            
        h, w = crop.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_skew, 1.0)
        deskewed = cv2.warpAffine(crop, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed
    except Exception as e:
        print(f"Hough Deskew Error: {e}")
        return crop

# ─── Plate Enhancement ────────────────────────────────────────────────────────
def enhance_plate(img):
    """Ultra-fast preprocessing for OCR."""
    try:
        # 1. Convert to grayscale immediately
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # 2. Fast Histogram Equalization (Better than CLAHE for speed sometimes, but CLAHE is okay)
        # Using a fixed clipLimit for speed
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 3. Simple thresholding is often faster and better for EasyOCR than sharpening
        return enhanced
    except Exception as e:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

def remove_ind_strip(img):
    """Removes the blue IND security strip (leftmost ~5%)."""
    h, w = img.shape[:2]
    return img[:, int(w * 0.05):]

# ─── Core Detection Pipeline ──────────────────────────────────────────────────
def extract_plate_v11(img, snap_id):
    """Optimized plate detection pipeline: Fast, Lazy, and Efficient."""
    if img is None:
        return []

    # Blistering speed optimization: Downscale huge input images to 640px width
    h_orig, w_orig = img.shape[:2]
    max_w = 640
    if w_orig > max_w:
        scale = float(max_w) / w_orig
        img = cv2.resize(img, (max_w, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
        print(f"  📐 Downscaled input image for blistering speed: {w_orig}x{h_orig} -> {max_w}x{img.shape[0]}", flush=True)

    # Step 1: Frame Quality Gate
    quality_score, is_acceptable = assess_frame_quality(img)
    print(f"  📊 Frame quality: {quality_score:.1f}", flush=True)
    if not is_acceptable:
        print(f"  ⏭️  Skipping blurry frame", flush=True)
        return []

    # Step 2: YOLO Inference (Optimized Resolution for Extreme Speed)
    # Using 640 resolution ensures robust plate detection even from 20cm far
    results = plate_detector(img, conf=0.15, imgsz=640, verbose=False)
    candidates = {}

    if not results or len(results[0].boxes) == 0:
        return []

    # Step 3: Process Detections (Fast-Exit if possible)
    # Limit to top 2 detections to avoid wasting CPU cycles on background false positives!
    for idx, box in enumerate(results[0].boxes[:2]):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Filter out IND strip or square false positives using aspect ratio
        box_w = x2 - x1
        box_h = y2 - y1
        if box_h > 0 and (box_w / box_h) < 1.5:
            continue
            
        # Crop with minimal padding
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        h_det, w_det = img.shape[:2]
        crop = img[max(0, y1-pad_h):min(h_det, y2+pad_h), max(0, x1-pad_w):min(w_det, x2+pad_w)]

        if crop.size == 0: continue

        # Apply Hough Skew Correction to flatten any tilt
        crop_deskewed = deskew_hough(crop)

        # Step 4: Lazy OCR Stages
        clean = remove_ind_strip(crop_deskewed)
        
        # Stage A: Enhanced Gray (The "Fast Path")
        img_a = enhance_plate(clean)
        
        # Run OCR on Stage A - Using direct recognition to completely bypass CRAFT text detector (4x faster)
        h_a, w_a = img_a.shape[:2]
        ocr_results = ocr_engine.recognize(
            img_a,
            horizontal_list=[[0, w_a, 0, h_a]],
            free_list=[],
            allowlist=OCR_ALLOWLIST,
            decoder='greedy'
        )
        
        for bbox, text, prob in ocr_results:
            print(f"    🔤 [Gray] {text} ({prob:.2f})")
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(cleaned) < 7: continue
            
            reg_match = find_registered_match(cleaned)
            if reg_match:
                print(f"  🔥 FAST-PASS: {reg_match} registered (matched directly from OCR: {text})!")
                return [{"plate": reg_match, "score": 1.0, "snapId": snap_id, "registered": True}]
                
            corrected = correct_indian_plate(cleaned)
            reg_match = find_registered_match(corrected)
            
            if reg_match:
                print(f"  🔥 FAST-PASS: {reg_match} registered (matched from corrected: {corrected})!")
                return [{"plate": reg_match, "score": 1.0, "snapId": snap_id, "registered": True}]
            
            # If high confidence or valid format, add to candidates
            is_hsrp = is_hsrp_format(corrected)
            score = prob + (0.2 if is_hsrp else 0)
            candidates[corrected] = score
            add_to_consensus(corrected, score)

        # Stage B: Binary Threshold (Skip if Stage A was "Good Enough")
        # Optimization: Only run Stage B if Stage A found NOTHING
        if not candidates:
            # Stage B is only reached for truly difficult plates
            img_b = cv2.adaptiveThreshold(img_a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            h_b, w_b = img_b.shape[:2]
            ocr_results_b = ocr_engine.recognize(
                img_b,
                horizontal_list=[[0, w_b, 0, h_b]],
                free_list=[],
                allowlist=OCR_ALLOWLIST,
                decoder='greedy'
            )
            for bbox_b, text, prob_b in ocr_results_b:
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(cleaned) < 7: continue
                
                reg_match = find_registered_match(cleaned)
                if reg_match:
                    print(f"  🔥 FAST-PASS: {reg_match} registered (matched directly from OCR: {text})!")
                    return [{"plate": reg_match, "score": 1.0, "snapId": snap_id, "registered": True}]
                    
                corrected = correct_indian_plate(cleaned)
                reg_match = find_registered_match(corrected)
                if reg_match:
                    print(f"  🔥 FAST-PASS: {reg_match} registered (matched from corrected: {corrected})!")
                    return [{"plate": reg_match, "score": 1.0, "snapId": snap_id, "registered": True}]
                candidates[corrected] = max(candidates.get(corrected, 0), 0.5)

    # Step 5: Fuzzy Matching Fallback (Runs if no high-confidence candidate OR if none of the candidates are registered)
    top_registered = any(check_database_for_plate(p) for p in candidates)
    if not candidates or max(candidates.values(), default=0) < 0.8 or not top_registered:
        for plate in list(candidates.keys()):
            for var in get_fuzzy_variations(plate):
                reg_match = find_registered_match(var)
                if reg_match:
                    print(f"  🧠 FUZZY MATCH: {plate} -> {reg_match} (Database matched!)")
                    return [{"plate": reg_match, "score": 0.95, "snapId": snap_id, "registered": True, "fuzzy": True}]

    # Step 6: Final Results (Top 3)
    best_con, _ = get_consensus_best()
    if best_con and best_con not in candidates:
        candidates[best_con] = 0.6

    sorted_plates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    final_results = []
    for p, s in sorted_plates[:3]:
        reg_match = find_registered_match(p)
        final_results.append({
            "plate": reg_match if reg_match else p, 
            "score": round(s, 2), 
            "snapId": snap_id,
            "registered": reg_match is not None
        })

    if final_results:
        print(f"  🏆 Best Result: {final_results[0]['plate']} ({final_results[0]['score']}) {'[REGISTERED]' if final_results[0]['registered'] else ''}", flush=True)
    else:
        print("  ❌ No plates found in this frame.", flush=True)

    sys.stdout.flush()
    return final_results

# ─── Flask Routes ─────────────────────────────────────────────────────────────
@app.route('/detect-plate', methods=['POST'])
def detect_plate():
    try:
        # Accept multiple formats: multipart, JSON (base64), or raw binary
        img_bytes = None
        
        if 'image' in request.files:
            # From website file upload
            file = request.files['image']
            img_bytes = file.read()
        elif request.is_json and 'image' in request.json:
            # From ESP32 base64
            image_data = request.json['image']
            img_bytes = base64.b64decode(image_data)
        elif request.content_type == 'image/jpeg':
            # From ESP32 raw binary (more efficient)
            img_bytes = request.data
        else:
            return jsonify({'error': 'No image provided or unsupported format'}), 400
        
        # Convert bytes to numpy array for OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
            
        # Run detection
        results = plate_detector(image, conf=0.25, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2)
                    },
                    'confidence': round(conf * 100, 2)
                })
        
        return jsonify({
            'success': True,
            'plates_detected': len(detections),
            'detections': detections,
            'has_plate': len(detections) > 0
        })
        
    except Exception as e:
        print(f"⚠️ Detection Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/datasets/<path:filename>')
def serve_datasets(filename):
    return send_from_directory('datasets', filename)

@app.route('/v1/plate-reader/', methods=['POST'])
@rate_limit(limit=30, window=60) # 30 requests per minute
@require_api_key
def plate_reader():
    global system_fully_online
    if not system_fully_online:
        print("⏳ [SYSTEM STANDBY] Server is performing initial system checks and synchronizing Firebase database. Pausing ANPR scan until fully verified...", flush=True)
        return jsonify({"results": []})

    try:
        content_type = request.content_type or ''
        print(f"📥 [{time.strftime('%H:%M:%S')}] /v1/plate-reader/ | Content-Type: '{content_type}'", flush=True)
        
        # Robust case-insensitive check to prevent falling back to form-parsing on raw/custom HTTP headers
        if 'image/jpeg' in content_type.lower():
            img_bytes = request.data
        else:
            file = request.files.get('upload')
            if not file:
                print("⚠️  No 'upload' file and Content-Type is not image/jpeg", flush=True)
                return jsonify({"results": []})
            img_bytes = file.read()

        if not img_bytes or len(img_bytes) == 0:
            print("⚠️  Empty image bytes received!", flush=True)
            return jsonify({"results": []})

        with latest_frame_lock:
            latest_frame_bytes = img_bytes

        print(f"📸 Received {len(img_bytes)} bytes image. Processing ANPR...", flush=True)
        nparr    = np.frombuffer(img_bytes, np.uint8)
        dec_img  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if dec_img is None:
            print("⚠️  Failed to decode image! Mismatched JPEG format.", flush=True)
            return jsonify({"results": []})

        # Helper to run blocking CPU/IO tasks asynchronously
        def run_in_background(target, *args, **kwargs):
            t = threading.Thread(target=target, args=args, kwargs=kwargs)
            t.daemon = True
            t.start()

        # Save incoming image for debugging asynchronously (non-blocking for OneDrive / slow HDDs)
        def save_debug_image(img):
            try:
                if not os.path.exists('datasets/debug'):
                    os.makedirs('datasets/debug')
                cv2.imwrite('datasets/debug/latest_request.jpg', img)
                print("💾 Saved received image asynchronously for diagnostics", flush=True)
            except Exception as ex:
                print(f"⚠️  Failed to save debug image: {ex}", flush=True)
        run_in_background(save_debug_image, dec_img)

        snap_id       = uuid.uuid4().hex[:6]
        plate_results = extract_plate_v11(dec_img, snap_id)

        # enforce booked-users-only access: Filter out plates not registered in bookings
        if plate_results:
            unregistered = [r['plate'] for r in plate_results if not r.get('registered', False)]
            if unregistered:
                print(f"⚠️  [ANPR] Rejected unregistered plates: {unregistered} (No booking found in Firebase!)", flush=True)
            plate_results = [r for r in plate_results if r.get('registered', False)]

        if plate_results:
            best  = plate_results[0]
            # Update thread-safe latest_scan storage for WiFi fallback bypass ONLY for verified/registered plates
            with latest_scan_lock:
                latest_scan["plate"] = best["plate"]
                latest_scan["confidence"] = best["score"]
                latest_scan["timestamp"] = int(time.time())
            fname = f"auto_train/{best['plate']}_{best['snapId']}.jpg"
            server_ip = request.host.split(':')[0]
            snap_url  = f"http://{server_ip}:5000/datasets/{fname}"

            # Async background execution for training image save and Firebase synchronizations
            def async_io_tasks(img, filename, best_plate, snap_u, results_list):
                try:
                    # 1. Save training image
                    if not os.path.exists('datasets/auto_train'):
                        os.makedirs('datasets/auto_train')
                    cv2.imwrite(os.path.join('datasets', filename), img)
                except Exception as ex:
                    print(f"⚠️  Failed to save train image: {ex}", flush=True)

                if firebase_enabled:
                    try:
                        is_valid = is_hsrp_format(best_plate)
                        payload = {
                            "plate":             best_plate,
                            "confidence":        results_list[0]['score'],
                            "candidates":        [r['plate'] for r in results_list],
                            "regexValid":        True,
                            "aiValid":           is_valid,
                            "isFullyAuthorized": is_valid,
                            "timestamp":         int(time.time())
                        }
                        
                        with cache_lock:
                            db_key = bookings_original_keys.get(best_plate, best_plate)
                        
                        import urllib.parse
                        if db:
                            db.reference('/anpr/latest_plate').set(payload)
                            db.reference(f'/bookings/{db_key}/snapshotUrl').set(snap_u)
                        else:
                            # REST Fallback for writing
                            global firebase_token
                            db_url = "https://YOUR-PROJECT-ID-default-rtdb.firebaseio.com"
                            
                            # 1. Update latest_plate
                            write_url = f"{db_url}/anpr/latest_plate.json?auth={firebase_token}"
                            req = urllib.request.Request(write_url, data=json.dumps(payload).encode(), method='PUT')
                            urllib.request.urlopen(req)
                            
                            # 2. Update snapshotUrl under the correct original key
                            safe_key = urllib.parse.quote(db_key)
                            snap_url_endpoint = f"{db_url}/bookings/{safe_key}/snapshotUrl.json?auth={firebase_token}"
                            req2 = urllib.request.Request(snap_url_endpoint, data=json.dumps(snap_u).encode(), method='PUT')
                            urllib.request.urlopen(req2)
                        print("📡 Firebase update complete (async)", flush=True)
                    except Exception as e:
                        print(f"  ⚠️  Firebase sync error: {e}", flush=True)

            run_in_background(async_io_tasks, dec_img, fname, best['plate'], snap_url, plate_results)

            for res in plate_results:
                res['url'] = snap_url

        return jsonify({"results": plate_results})

    except Exception as e:
        import traceback
        print(f"❌ ERROR in /v1/plate-reader/: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "results": []}), 500

# ─── QR Code Decoder ──────────────────────────────────────────────────────────
qr_decoder = cv2.QRCodeDetector()

@app.route('/v1/qr-reader/', methods=['POST'])
def qr_reader():
    global system_fully_online
    if not system_fully_online:
        print("⏳ [SYSTEM STANDBY] Server is performing initial system checks and synchronizing Firebase database. Pausing QR decode until fully verified...", flush=True)
        return jsonify([{"symbol": [{"data": ""}]}])

    print("\n📥 [QR_READER] Request received on /v1/qr-reader/", flush=True)
    if request.content_type == 'image/jpeg':
        img_bytes = request.data
    else:
        file = request.files.get('file') or request.files.get('upload')
        if not file:
            print("❌ [QR_READER] No file or image data found in request!", flush=True)
            return jsonify([{"symbol": [{"data": ""}]}])
        img_bytes = file.read()

    print(f"📥 [QR_READER] Successfully read {len(img_bytes)} bytes of image buffer", flush=True)

    if img_bytes:
        with latest_frame_lock:
            latest_frame_bytes = img_bytes

    nparr    = np.frombuffer(img_bytes, np.uint8)
    dec_img  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if dec_img is None:
        print("❌ [QR_READER] Failed to decode image bytes into CV2 frame!", flush=True)
        return jsonify([{"symbol": [{"data": ""}]}])

    # Ultimate Unbreakable QR Code Decoding Engine (8-Pass All-PyZbar Suite):
    # 1. We apply CLAHE (Adaptive Histogram Equalization) to instantly solve shadows, glare, and low-contrast.
    # 2. We use Lanczos4 interpolation for high-fidelity edge-retaining upscaling (essential for 400x296).
    # 3. We completely bypass slow OpenCV engines.
    # 4. If all passes fail, we save the frame to disk so we can visually inspect what the camera sees.

    # Pre-import pyzbar decoder
    from pyzbar.pyzbar import decode as zbar_decode
    
    data = None
    gray = cv2.cvtColor(dec_img, cv2.COLOR_BGR2GRAY)
    
    # Pre-calculate high-contrast CLAHE image
    clahe_engine = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_gray = clahe_engine.apply(gray)

    # 🚀 Pass 1: Standard Grayscale (~1ms)
    try:
        print("🔍 [QR_READER] Pass 1: pyzbar on Grayscale...", flush=True)
        with zbar_lock:
            decoded_objs = zbar_decode(gray)
        if decoded_objs:
            data = decoded_objs[0].data.decode('utf-8')
            print(f"🎯 [QR_READER] SUCCESS: Decoded via Pass 1 (Grayscale) -> '{data}'", flush=True)
    except Exception as e:
        print(f"  ⚠️ pyzbar Pass 1 error: {e}", flush=True)

    # 🚀 Pass 2: High-Contrast CLAHE Grayscale (~1.5ms)
    if not data:
        try:
            print("🔍 [QR_READER] Pass 2: pyzbar on High-Contrast CLAHE...", flush=True)
            with zbar_lock:
                decoded_objs = zbar_decode(clahe_gray)
            if decoded_objs:
                data = decoded_objs[0].data.decode('utf-8')
                print(f"🎯 [QR_READER] SUCCESS: Decoded via Pass 2 (CLAHE) -> '{data}'", flush=True)
        except Exception as e:
            print(f"  ⚠️ pyzbar Pass 2 error: {e}", flush=True)

    # 🚀 Pass 3: Sharpened CLAHE Grayscale (Counteract lens/motion blur, ~2ms)
    if not data:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(clahe_gray, -1, kernel)
        try:
            print("🔍 [QR_READER] Pass 3: pyzbar on Sharpened CLAHE...", flush=True)
            with zbar_lock:
                decoded_objs = zbar_decode(sharpened)
            if decoded_objs:
                data = decoded_objs[0].data.decode('utf-8')
                print(f"🎯 [QR_READER] SUCCESS: Decoded via Pass 3 (Sharpened CLAHE) -> '{data}'", flush=True)
        except Exception:
            pass

    # 🚀 Pass 4: Adaptive Threshold on CLAHE Grayscale (Filters strong reflections/shadows, ~2.5ms)
    if not data:
        thresh = cv2.adaptiveThreshold(clahe_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        try:
            print("🔍 [QR_READER] Pass 4: pyzbar on Adaptive Binarized CLAHE...", flush=True)
            with zbar_lock:
                decoded_objs = zbar_decode(thresh)
            if decoded_objs:
                data = decoded_objs[0].data.decode('utf-8')
                print(f"🎯 [QR_READER] SUCCESS: Decoded via Pass 4 (Binarized CLAHE) -> '{data}'", flush=True)
        except Exception:
            pass

    # 🚀 Pass 5: 2x Interpolated Grayscale (For small QR codes, ~0.5ms)
    if not data:
        h_g, w_g = gray.shape[:2]
        large_2x = cv2.resize(gray, (w_g * 2, h_g * 2), interpolation=cv2.INTER_CUBIC)
        try:
            print("🔍 [QR_READER] Pass 5: pyzbar on 2x Cubic Grayscale...", flush=True)
            with zbar_lock:
                decoded_objs = zbar_decode(large_2x)
            if decoded_objs:
                data = decoded_objs[0].data.decode('utf-8')
                print(f"🎯 [QR_READER] SUCCESS: Decoded via Pass 5 (2x Cubic) -> '{data}'", flush=True)
        except Exception:
            pass

    # 🚀 Pass 6: 2x Interpolated CLAHE Grayscale (Enlarged + high contrast, ~0.5ms)
    if not data:
        large_clahe_2x = cv2.resize(clahe_gray, (w_g * 2, h_g * 2), interpolation=cv2.INTER_CUBIC)
        try:
            print("🔍 [QR_READER] Pass 6: pyzbar on 2x Cubic CLAHE...", flush=True)
            with zbar_lock:
                decoded_objs = zbar_decode(large_clahe_2x)
            if decoded_objs:
                data = decoded_objs[0].data.decode('utf-8')
                print(f"🎯 [QR_READER] SUCCESS: Decoded via Pass 6 (2x Cubic CLAHE) -> '{data}'", flush=True)
        except Exception:
            pass

    # 🚀 Pass 7: 3x Interpolated Grayscale (For extremely tiny/distant QR codes, ~1ms)
    if not data:
        large_3x = cv2.resize(gray, (w_g * 3, h_g * 3), interpolation=cv2.INTER_CUBIC)
        try:
            print("🔍 [QR_READER] Pass 7: pyzbar on 3x Cubic Grayscale...", flush=True)
            with zbar_lock:
                decoded_objs = zbar_decode(large_3x)
            if decoded_objs:
                data = decoded_objs[0].data.decode('utf-8')
                print(f"🎯 [QR_READER] SUCCESS: Decoded via Pass 7 (3x Cubic) -> '{data}'", flush=True)
        except Exception:
            pass

    # 🚀 Pass 8: 3x Interpolated CLAHE Grayscale (Ultimate deep-recovery pass, ~1ms)
    if not data:
        large_clahe_3x = cv2.resize(clahe_gray, (w_g * 3, h_g * 3), interpolation=cv2.INTER_CUBIC)
        try:
            print("🔍 [QR_READER] Pass 8: pyzbar on 3x Cubic CLAHE...", flush=True)
            with zbar_lock:
                decoded_objs = zbar_decode(large_clahe_3x)
            if decoded_objs:
                data = decoded_objs[0].data.decode('utf-8')
                print(f"🎯 [QR_READER] SUCCESS: Decoded via Pass 8 (3x Cubic CLAHE) -> '{data}'", flush=True)
        except Exception:
            pass

    if not data:
        # Save the failed image frame to disk so we can visually inspect exactly what the camera sees
        try:
            if not os.path.exists('datasets/failed_qr'):
                os.makedirs('datasets/failed_qr')
            cv2.imwrite('datasets/failed_qr/last_failed_qr.jpg', dec_img)
            print("💾 [QR_READER] Saved last failed QR frame to 'datasets/failed_qr/last_failed_qr.jpg' for diagnostics.", flush=True)
        except Exception as e:
            print(f"  ⚠️ Failed to save failed QR frame: {e}", flush=True)
        print("❌ [QR_READER] Failed to decode QR code on all optimized passes!", flush=True)

    if data:
        # Reformat QR data for Dev Board compatibility if it has ID: but not CAR:
        if "CAR:" not in data and "ID:" in data:
            match = re.search(r'ID:([^\s|]+)', data)
            if match:
                booking_id = match.group(1).strip()
                data = f"CAR:{booking_id}|ID:{booking_id}"
                print(f"🔄 [QR_READER] Server-side reformatted for Dev Board compatibility: '{data}'", flush=True)

        # Update thread-safe latest_qr storage for WiFi fallback bypass
        with latest_qr_lock:
            latest_qr["data"] = data
            latest_qr["timestamp"] = int(time.time())
            
        try:
            plate_match = re.search(r'CAR:([^|]+)', data)
            plate   = plate_match.group(1) if plate_match else "GUEST"
            snap_id = uuid.uuid4().hex[:6]
            fname   = f"checkout_evidence/{plate}_{snap_id}.jpg"
            server_ip = request.host.split(':')[0]
            snap_url  = f"http://{server_ip}:5000/datasets/{fname}"

            # Async background execution for saving checkout evidence and Firebase updates (no block!)
            def async_qr_tasks(img, filename, best_plate, snap_u):
                try:
                    if not os.path.exists('datasets/checkout_evidence'):
                        os.makedirs('datasets/checkout_evidence')
                    cv2.imwrite(os.path.join('datasets', filename), img)
                except Exception as ex:
                    print(f"⚠️  Failed to save QR checkout image: {ex}", flush=True)

                if firebase_enabled and best_plate != "GUEST":
                    try:
                        with cache_lock:
                            db_key = bookings_original_keys.get(best_plate, best_plate)
                        
                        if db:
                            db.reference(f'/bookings/{db_key}/checkoutPhoto').set(snap_u)
                        else:
                            global firebase_token
                            if not firebase_token:
                                firebase_token = get_firebase_token_fallback()
                            db_url = "https://YOUR-PROJECT-ID-default-rtdb.firebaseio.com"
                            import urllib.parse
                            safe_key = urllib.parse.quote(db_key)
                            checkout_url_endpoint = f"{db_url}/bookings/{safe_key}/checkoutPhoto.json?auth={firebase_token}"
                            req = urllib.request.Request(checkout_url_endpoint, data=json.dumps(snap_u).encode(), method='PUT')
                            urllib.request.urlopen(req)
                        print("📡 Firebase checkoutPhoto sync complete (async)", flush=True)
                    except Exception as e:
                        print(f"  ⚠️  QR evidence error: {e}", flush=True)

            # Delegate heavy disk IO and REST networks to background thread
            t = threading.Thread(target=async_qr_tasks, args=(dec_img, fname, plate, snap_url))
            t.daemon = True
            t.start()

        except Exception as e:
            print(f"  ⚠️  QR execution setup error: {e}", flush=True)

        return jsonify([{"symbol": [{"data": data}]}])

    return jsonify([{"symbol": [{"data": ""}]}])

# ─── New Fallback Polling Endpoints for Dev Board WiFi Bypass ─────────────────
@app.route('/v1/proxy/stream', methods=['GET'])
def video_feed():
    def gen():
        last_sent = None
        try:
            while True:
                with latest_frame_lock:
                    frame = latest_frame_bytes
                if frame and frame != last_sent:
                    last_sent = frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.05) # ~20 FPS polling of buffer
        except GeneratorExit:
            print("🔌 [STREAM] Browser client disconnected, closing thread stream and releasing worker thread.", flush=True)
        except Exception as e:
            print(f"⚠️ [STREAM] Stream thread exception: {e}", flush=True)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/v1/proxy/get-latest-plate', methods=['GET'])
def proxy_get_latest_plate():
    with latest_scan_lock:
        return jsonify(latest_scan)

@app.route('/v1/proxy/get-latest-qr', methods=['GET'])
def proxy_get_latest_qr():
    with latest_qr_lock:
        return jsonify(latest_qr)

# ─── Firebase Bridge for Dev Module ───────────────────────────────────────────
@app.route('/v1/trigger-dev/', methods=['GET', 'POST'])
def trigger_dev():
    cmd = request.args.get('cmd')
    if not cmd:
        return jsonify({"status": "error", "message": "Missing cmd parameter"}), 400

    print(f"[BRIDGE] Forwarding to Dev via Firebase: {cmd}")
    payload = {
        "cmd": cmd,
        "timestamp": int(time.time()),
        "rand": uuid.uuid4().hex[:8]
    }
    
    if firebase_enabled:
        try:
            if db:
                db.reference('/dev_commands/latest').set(payload)
            else:
                global firebase_token
                db_url = "https://YOUR-PROJECT-ID-default-rtdb.firebaseio.com"
                write_url = f"{db_url}/dev_commands/latest.json?auth={firebase_token}"
                req = urllib.request.Request(write_url, data=json.dumps(payload).encode(), method='PUT')
                urllib.request.urlopen(req)
        except Exception as e:
            print(f"  ⚠️  Firebase Bridge Error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "ok"})

# ─── New Proxy Endpoints for SSL-Free Hardware with REST Fallback ───────────

def firebase_rest_request(path, method='GET', data=None):
    global firebase_token
    db_url = "https://YOUR-PROJECT-ID-default-rtdb.firebaseio.com"
    if not path.startswith('/'):
        path = '/' + path
    url = f"{db_url}{path}.json"
    if firebase_token:
        url += f"?auth={firebase_token}"
        
    try:
        req_data = json.dumps(data).encode() if data is not None else None
        headers = {'Content-Type': 'application/json'} if data is not None else {}
        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        with urllib.request.urlopen(req) as resp:
            resp_data = resp.read().decode()
            return json.loads(resp_data) if resp_data else None
    except Exception as e:
        print(f"  ⚠️ Firebase REST Error [{method} {path}]: {e}")
        return None

def overtime_monitoring_loop():
    print("⏰ [OVERTIME MONITOR] Starting background tracking thread...", flush=True)
    time.sleep(15)  # Wait for cache to warm up
    while True:
        try:
            # Safely fetch active bookings and original keys mapping from thread cache
            with cache_lock:
                active_bookings = list(bookings_cache.items())
                keys_map = dict(bookings_original_keys)
                
            now_ts = int(time.time())
            
            for clean_plate, data in active_bookings:
                if not isinstance(data, dict):
                    continue
                
                # Check if user has an active checked-in status or has entry_time set without checkout
                status = data.get("status")
                is_active = data.get("active", False)
                
                # Checked-in states
                if status == "checked-in" or (is_active and status != "completed" and status != "waiting"):
                    entry_time = data.get("entry_time", 0)
                    
                    # Convert duration to minutes or use bookedMinutes directly
                    booked_mins = data.get("bookedMinutes", 0)
                    if not booked_mins:
                        booked_mins = int(data.get("duration", 1)) * 60
                        
                    mobile = data.get("mobile") or data.get("phone") or "9999999999"
                    name = data.get("name", "Driver")
                    plate = data.get("plate", clean_plate)
                    slot_id = data.get("slotId") or f"SLOT-{data.get('slot', 'X')}"
                    
                    if entry_time > 0 and booked_mins > 0:
                        expiry_time = entry_time + (booked_mins * 60)
                        
                        if now_ts > expiry_time:
                            overstay_seconds = now_ts - expiry_time
                            overstay_mins = int(overstay_seconds / 60)
                            if overstay_mins == 0:
                                overstay_mins = 1  # Round up to 1 min if overstayed
                            
                            fine_amount = overstay_mins * 3
                            
                            # Check if already notified
                            already_notified = data.get("overtimeNotified", False)
                            
                            update_fields = {
                                "finePending": True,
                                "fine": fine_amount,
                                "extraMinutes": overstay_mins
                            }
                            
                            original_key = keys_map.get(clean_plate, clean_plate)
                            
                            # First time exceeding: log message and simulated SMS broadcast
                            if not already_notified:
                                update_fields["overtimeNotified"] = True
                                
                                # Send dynamic SMS warning log message
                                sms_text = f"Warning: Dear {name}, your parking slot {slot_id} time for vehicle {plate} is up! Surcharge of 3 rupees per minute is active starting now. Please check out or renew."
                                print(f"📱 [AUTOMATED SMS DISPATCH TO {mobile}]: {sms_text}", flush=True)
                                
                                # Automated HTTP trigger to Textbelt (Free API - 1 per day limit)
                                try:
                                    textbelt_url = "https://textbelt.com/text"
                                    data = urllib.parse.urlencode({
                                        'phone': mobile,
                                        'message': sms_text,
                                        'key': 'textbelt'
                                    }).encode('utf-8')
                                    req = urllib.request.Request(textbelt_url, data=data)
                                    
                                    # Actually send the request
                                    with urllib.request.urlopen(req, timeout=5) as response:
                                        response_data = response.read()
                                        print(f"  ✅ Textbelt Success: {response_data}", flush=True)
                                except Exception as e:
                                    print(f"  ⚠️  Textbelt API Error: {e}", flush=True)
                                
                                # Push message to general logs for Security Guard & Dashboard
                                log_payload = {
                                    "type": "SYSTEM_EVENT",
                                    "timestamp": now_ts * 1000,
                                    "msg": f"SMS WARNING sent to {mobile} ({name}): S{slot_id} expired. Surcharge: ₹3/min is active."
                                }
                                
                                # Also push to a sub-node 'sms_notifications' under bookings for direct display
                                notification_payload = {
                                    "recipient": mobile,
                                    "message": sms_text,
                                    "timestamp": now_ts * 1000,
                                    "plate": plate,
                                    "slotId": slot_id,
                                    "fineRate": "₹3/min"
                                }
                                
                                if db:
                                    db.reference('/logs').push(log_payload)
                                    db.reference(f'/bookings/{original_key}/sms_notifications').push(notification_payload)
                                else:
                                    # REST fallback
                                    firebase_rest_request('/logs', 'POST', log_payload)
                                    firebase_rest_request(f'/bookings/{original_key}/sms_notifications', 'POST', notification_payload)
                            
                            # Update fine values under current booking key
                            if db:
                                db.reference(f'/bookings/{original_key}').update(update_fields)
                            else:
                                firebase_rest_request(f'/bookings/{original_key}', 'PATCH', update_fields)
                                
        except Exception as e:
            print(f"⚠️ [OVERTIME MONITOR] Error in loop: {e}", flush=True)
            
        time.sleep(10)  # Monitor every 10 seconds

# Start overtime tracker background thread
threading.Thread(target=overtime_monitoring_loop, daemon=True).start()

@app.route('/v1/proxy/send-sms', methods=['POST'])
def proxy_send_sms():
    data = request.json
    mobile = data.get('mobile')
    plate = data.get('plate')
    message = data.get('message')
    
    if not mobile or not message:
        return jsonify({"status": "error", "message": "Missing mobile or message"}), 400
        
    try:
        # Automated HTTP trigger to Fast2SMS Gateway
        print(f"📱 [MANUAL SMS DISPATCH TO {mobile}]: {message}", flush=True)
        
        try:
            # Textbelt Free API (1 per day limit)
            textbelt_url = "https://textbelt.com/text"
            api_data = urllib.parse.urlencode({
                'phone': mobile,
                'message': message,
                'key': 'textbelt'
            }).encode('utf-8')
            req = urllib.request.Request(textbelt_url, data=api_data)
            
            # Actually send the request
            with urllib.request.urlopen(req, timeout=5) as response:
                response_data = response.read().decode('utf-8')
                print(f"  ✅ Textbelt Success: {response_data}", flush=True)
                
                # Check if Textbelt rejected due to quota
                resp_json = json.loads(response_data)
                if not resp_json.get('success'):
                    return jsonify({"status": "error", "message": f"Textbelt Quota Exceeded: {resp_json.get('error')}"}), 502
                    
        except Exception as e:
            print(f"  ⚠️  Textbelt API Error: {e}", flush=True)
            return jsonify({"status": "error", "message": f"Gateway Error: {str(e)}."}), 502
        
        # Log it to the database
        now_ts = int(time.time())
        log_payload = {
            "type": "SYSTEM_EVENT",
            "timestamp": now_ts * 1000,
            "msg": f"Manual SMS sent to {mobile} for vehicle {plate}: {message}"
        }
        
        if db:
            db.reference('/logs').push(log_payload)
        else:
            firebase_rest_request('/logs', 'POST', log_payload)
            
        return jsonify({"status": "success", "message": "SMS dispatched via Fast2SMS gateway"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/v1/proxy/check-booking', methods=['GET'])
def proxy_check_booking():
    plate = request.args.get('plate')
    if not plate: return jsonify({"found": False})
    
    try:
        if db:
            ref = db.reference(f'/bookings/{plate}')
            data = ref.get()
            if data:
                is_active = data.get("active", False)
                status = data.get("status", "")
                if is_active or status in ("active", "checked-in", "completed"):
                    print(f"⚠️ [ACCESS RESTRICTED] Vehicle {plate} attempted entry but is already inside (status: {status}) or completed!", flush=True)
                    return jsonify({
                        "found": True,
                        "paid": False,
                        "active": is_active,
                        "slot": data.get("slot", -1),
                        "name": data.get("name", "Unknown"),
                        "entry_time": data.get("entry_time", 0),
                        "bookedMinutes": data.get("bookedMinutes", 0),
                        "finePending": data.get("finePending", False)
                    })
                return jsonify({
                    "found": True,
                    "paid": data.get("paid", False),
                    "active": data.get("active", False),
                    "slot": data.get("slot", -1),
                    "name": data.get("name", "Unknown"),
                    "entry_time": data.get("entry_time", 0),
                    "bookedMinutes": data.get("bookedMinutes", 0),
                    "finePending": data.get("finePending", False)
                })
        else:
            # REST Fallback
            data = firebase_rest_request(f'/bookings/{plate}', 'GET')
            if data:
                is_active = data.get("active", False)
                status = data.get("status", "")
                if is_active or status in ("active", "checked-in", "completed"):
                    print(f"⚠️ [ACCESS RESTRICTED] Vehicle {plate} attempted entry but is already inside (status: {status}) or completed!", flush=True)
                    return jsonify({
                        "found": True,
                        "paid": False,
                        "active": is_active,
                        "slot": data.get("slot", -1),
                        "name": data.get("name", "Unknown"),
                        "entry_time": data.get("entry_time", 0),
                        "bookedMinutes": data.get("bookedMinutes", 0),
                        "finePending": data.get("finePending", False)
                    })
                # Ensure fields are correctly structured in response
                return jsonify({
                    "found": True,
                    "paid": data.get("paid", False),
                    "active": data.get("active", False),
                    "slot": data.get("slot", -1),
                    "name": data.get("name", "Unknown"),
                    "entry_time": data.get("entry_time", 0),
                    "bookedMinutes": data.get("bookedMinutes", 0),
                    "finePending": data.get("finePending", False)
                })
    except Exception as e:
        print(f"  ⚠️  Proxy Check Error: {e}")
    
    return jsonify({"found": False})

@app.route('/v1/proxy/log-entry', methods=['POST'])
def proxy_log_entry():
    data = request.json
    plate = data.get('plate')
    if not plate: return jsonify({"status": "error", "message": "Missing plate"}), 400
    
    ts = int(time.time())
    try:
        if db:
            # 1. Log the entry detection event
            db.reference('/anpr/detections').push({
                "plate": plate,
                "type": "ENTRY",
                "timestamp": ts
            })
            
            # 2. Update the booking status
            with cache_lock:
                db_key = bookings_original_keys.get(plate, plate)
            
            booking_ref = db.reference(f'/bookings/{db_key}')
            booking = booking_ref.get()
            
            if not booking:
                return jsonify({"status": "error", "message": "No booking found"}), 404
            
            booking_ref.update({
                "status": "checked-in",
                "active": true,
                "entry_time": ts
            })
            
            # 3. Update the slot status if assigned
            slot_id = booking.get('slot')
            if slot_id:
                # Update hardware node
                db.reference(f'/parking/slots/S{slot_id}').update({
                    "active": True,
                    "plate": plate
                })
                # Update dashboard node
                db.reference(f'/slots/slot{slot_id}').update({
                    "occupied": True,
                    "status": "occupied",
                    "plate": plate,
                    "currentVehicle": plate,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
        else:
            # REST Fallback
            firebase_rest_request('/anpr/detections', 'POST', {
                "plate": plate,
                "type": "ENTRY",
                "timestamp": ts
            })
            
            # Get original key from cache if possible
            with cache_lock:
                db_key = bookings_original_keys.get(plate, plate)
            
            booking = firebase_rest_request(f'/bookings/{db_key}', 'GET')
            if not booking:
                return jsonify({"status": "error", "message": "No booking found"}), 404
                
            firebase_rest_request(f'/bookings/{db_key}', 'PATCH', {
                "status": "checked-in",
                "active": True,
                "entry_time": ts
            })
            
            slot_id = booking.get('slot')
            if slot_id:
                firebase_rest_request(f'/parking/slots/S{slot_id}', 'PATCH', {
                    "active": True,
                    "plate": plate
                })
                firebase_rest_request(f'/slots/slot{slot_id}', 'PATCH', {
                    "occupied": True,
                    "status": "occupied",
                    "plate": plate,
                    "currentVehicle": plate,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
                
        print(f"✅ [PROXY] Successfully logged entry for {plate} (Slot: {booking.get('slot', 'N/A')})", flush=True)
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"  ⚠️  Proxy Entry Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/v1/proxy/log-exit', methods=['POST'])
def proxy_log_exit():
    data = request.json
    plate = data.get('plate')
    if not plate: return jsonify({"status": "error"})
    
    ts = int(time.time())
    try:
        if db:
            # Log detection
            db.reference('/anpr/detections').push({
                "plate": plate,
                "type": "EXIT",
                "timestamp": ts
            })
            # Get slot info to vacate
            ref = db.reference(f'/bookings/{plate}')
            booking = ref.get()
            vacated_slot = booking.get('slot', 0) if booking else 0
            
            # Calculate early exit refund (80% of 50/hr rate for unused full hours)
            refund_amount = 0.0
            mins_used = 0
            if booking:
                entry_time = booking.get('entry_time', 0)
                mins_booked = booking.get('bookedMinutes', 0)
                if entry_time > 0 and mins_booked > 0:
                    mins_used = int((ts - entry_time) / 60)
                    if mins_used < mins_booked:
                        unused_mins = mins_booked - mins_used
                        unused_hours = int(unused_mins / 60)
                        if unused_hours >= 1:
                            refund_amount = float(unused_hours * 50 * 0.8)
            
            # Update booking - refundAmount is set to 0.0 initially, forcing manual application
            ref.update({
                "active": False,
                "status": "completed",
                "actualExitTime": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "refundAmount": 0.0,
                "minutesUsed": mins_used
            })
            
            if vacated_slot > 0:
                # Update original slot
                db.reference(f'/parking/slots/S{vacated_slot}').update({
                    "active": False,
                    "plate": ""
                })
                # Update React Dashboard Slot Node
                db.reference(f'/slots/slot{vacated_slot}').update({
                    "occupied": False,
                    "status": "available",
                    "plate": "",
                    "name": "",
                    "currentVehicle": None,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
                # Promote from waiting list
                promote_waiting_list(vacated_slot)
        else:
            # REST Fallback
            firebase_rest_request('/anpr/detections', 'POST', {
                "plate": plate,
                "type": "EXIT",
                "timestamp": ts
            })
            booking = firebase_rest_request(f'/bookings/{plate}', 'GET')
            vacated_slot = booking.get('slot', 0) if booking else 0
            
            refund_amount = 0.0
            mins_used = 0
            if booking:
                entry_time = booking.get('entry_time', 0)
                mins_booked = booking.get('bookedMinutes', 0)
                if entry_time > 0 and mins_booked > 0:
                    mins_used = int((ts - entry_time) / 60)
                    if mins_used < mins_booked:
                        unused_mins = mins_booked - mins_used
                        unused_hours = int(unused_mins / 60)
                        if unused_hours >= 1:
                            refund_amount = float(unused_hours * 50 * 0.8)

            firebase_rest_request(f'/bookings/{plate}', 'PATCH', {
                "active": False,
                "status": "completed",
                "actualExitTime": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "refundAmount": refund_amount,
                "minutesUsed": mins_used
            })
            
            if vacated_slot > 0:
                # Update original slot
                firebase_rest_request(f'/parking/slots/S{vacated_slot}', 'PATCH', {
                    "active": False,
                    "plate": ""
                })
                # Update React Dashboard Slot Node
                firebase_rest_request(f'/slots/slot{vacated_slot}', 'PATCH', {
                    "occupied": False,
                    "status": "available",
                    "plate": "",
                    "name": "",
                    "currentVehicle": None,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
                promote_waiting_list(vacated_slot)
                
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"  ⚠️  Proxy Exit Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

def promote_waiting_list(slot):
    try:
        if db:
            ref = db.reference('/waiting_list')
            waiting = ref.get()
        else:
            waiting = firebase_rest_request('/waiting_list', 'GET')
            
        if waiting:
            first_key = next(iter(waiting))
            path = f"/bookings/{first_key}"
            if db:
                db.reference(path).update({
                    "slot": slot,
                    "slotId": f"SLOT-{slot}",
                    "status": "pending",
                    "isWaiting": False
                })
                db.reference(f"/parking/slots/S{slot}").update({
                    "active": True,
                    "plate": first_key
                })
                # React Node
                db.reference(f"/slots/slot{slot}").update({
                    "occupied": True,
                    "status": "occupied",
                    "plate": first_key,
                    "currentVehicle": first_key,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
                db.reference(f"/waiting_list/{first_key}").delete()
            else:
                firebase_rest_request(path, 'PATCH', {
                    "slot": slot,
                    "slotId": f"SLOT-{slot}",
                    "status": "pending",
                    "isWaiting": False
                })
                firebase_rest_request(f"/parking/slots/S{slot}", 'PATCH', {
                    "active": True,
                    "plate": first_key
                })
                # React Node
                firebase_rest_request(f"/slots/slot{slot}", 'PATCH', {
                    "occupied": True,
                    "status": "occupied",
                    "plate": first_key,
                    "currentVehicle": first_key,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
                firebase_rest_request(f"/waiting_list/{first_key}", 'DELETE')
                
            print(f"  ✨ Promoted {first_key} to Slot S{slot}")
    except Exception as e:
        print(f"  ⚠️  Promotion Error: {e}")

@app.route('/v1/proxy/update-slot', methods=['POST'])
def proxy_update_slot():
    data = request.json
    slot = data.get('slot')
    if not slot: return jsonify({"status": "error"})
    
    active_status = data.get('active', False)
    plate_val = data.get('plate', "")
    
    try:
        if db:
            db.reference(f'/parking/slots/S{slot}').update({
                "active": active_status,
                "plate": plate_val
            })
            # React Node
            db.reference(f'/slots/slot{slot}').update({
                "occupied": active_status,
                "status": "occupied" if active_status else "available",
                "plate": plate_val,
                "currentVehicle": plate_val if active_status else None,
                "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            })
        else:
            firebase_rest_request(f'/parking/slots/S{slot}', 'PATCH', {
                "active": active_status,
                "plate": plate_val
            })
            # React Node
            firebase_rest_request(f'/slots/slot{slot}', 'PATCH', {
                "occupied": active_status,
                "status": "occupied" if active_status else "available",
                "plate": plate_val,
                "currentVehicle": plate_val if active_status else None,
                "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            })
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/v1/proxy/get-settings', methods=['GET'])
def proxy_get_settings():
    try:
        if db:
            settings = db.reference('/settings').get()
            return jsonify(settings or {})
        else:
            settings = firebase_rest_request('/settings', 'GET')
            return jsonify(settings or {})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/v1/proxy/get-free-slots', methods=['GET'])
def proxy_get_free_slots():
    try:
        if db:
            slots = db.reference('/slots').get()
        else:
            slots = firebase_rest_request('/slots', 'GET')
            
        if slots:
            if isinstance(slots, dict):
                free_count = sum(1 for s in slots.values() if s and s.get('status', 'available') == 'available')
                total_count = len(slots)
            elif isinstance(slots, list):
                free_count = sum(1 for s in slots if s and s.get('status', 'available') == 'available')
                total_count = len(slots)
            else:
                free_count = 10
                total_count = 10
            return jsonify({"free_slots": free_count, "total_slots": total_count})
    except Exception as e:
        print(f"  ⚠️ Error counting free slots: {e}")
    return jsonify({"free_slots": 10, "total_slots": 10})

@app.route('/v1/proxy/init-slots', methods=['POST'])
def proxy_init_slots():
    try:
        if db:
            ref = db.reference('/parking/slots/S1/active')
            is_init = ref.get() is not None
        else:
            is_init = firebase_rest_request('/parking/slots/S1/active', 'GET') is not None
            
        if not is_init:
            total = 10
            slots_data = {}
            for i in range(1, total + 1):
                slots_data[f"S{i}"] = {
                    "active": False,
                    "plate": ""
                }
            if db:
                for i in range(1, total + 1):
                    db.reference(f'/parking/slots/S{i}').set({
                        "active": False,
                        "plate": ""
                    })
                db.reference('/parking/free_slots').set(total)
            else:
                firebase_rest_request('/parking/slots', 'PATCH', slots_data)
                firebase_rest_request('/parking/free_slots', 'PUT', total)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/v1/proxy/log-alert', methods=['POST'])
def proxy_log_alert():
    data = request.json
    try:
        if db:
            data['timestamp'] = int(time.time())
            db.reference('/parking/alerts').push(data)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/v1/proxy/latest-plate', methods=['POST'])
def proxy_latest_plate():
    data = request.json
    try:
        if db:
            data['timestamp'] = int(time.time())
            db.reference('/anpr/latest_plate').set(data)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ─── Plate-Based Exit Processing (Replaces QR exit) ──────────────────────────
#
#   POST /v1/proxy/plate-exit
#   Body: { "plate": "MH12AB1234" }
#   Logic:
#     1. Fuzzy-match plate against active bookings (handles 1-char OCR errors).
#     2. If booking.finePending → return fine details (force fine payment first).
#     3. If booking.status == "checked-in" → log exit, calculate refund, free slot.
#     4. If booking.status == "pending/active" (hasn't entered yet) → also allow
#        exit-as-no-entry (mark no-show, process refund immediately).
#   Returns full booking snapshot + refund calculation to the frontend.

PLATE_EXIT_FINE_RATE    = 3    # ₹ per minute overtime
PLATE_EXIT_REFUND_RATE  = 0.80 # 80% early-exit refund multiplier on unused full hours
NO_SHOW_PLATFORM_FEE   = 0.05 # 5% platform fee on no-show refund

def fuzzy_plate_match(target: str, candidates: dict) -> tuple[str | None, dict | None]:
    """
    Tries to match `target` plate to a booking key in `candidates`.
    Accepts direct match first, then 1-character substitution (OCR tolerance).
    Returns (matched_key, booking_dict) or (None, None).
    """
    clean_target = re.sub(r'[^A-Z0-9]', '', target.upper())
    # Direct match
    if clean_target in candidates:
        return clean_target, candidates[clean_target]
    # 1-char fuzzy match
    for key, val in candidates.items():
        clean_key = re.sub(r'[^A-Z0-9]', '', key.upper())
        if len(clean_key) == len(clean_target):
            diffs = sum(1 for a, b in zip(clean_key, clean_target) if a != b)
            if diffs <= 1:
                return key, val
    return None, None


@app.route('/v1/proxy/plate-exit', methods=['POST'])
def plate_exit():
    """
    Plate-based exit processing. Replaces the QR-based exit pipeline.
    Body JSON: { "plate": "MH12AB1234" }
    """
    data  = request.json or {}
    plate = re.sub(r'[^A-Z0-9]', '', (data.get('plate') or '').upper())
    if not plate:
        return jsonify({"status": "error", "message": "Missing plate"}), 400

    ts = int(time.time())
    now_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    # ── Fetch all active bookings ─────────────────────────────────────────────
    try:
        if db:
            raw = db.reference('/bookings').get()
        else:
            raw = firebase_rest_request('/bookings', 'GET')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    if not raw:
        return jsonify({"status": "error", "message": "No bookings found"}), 404

    db_key, booking = fuzzy_plate_match(plate, raw if isinstance(raw, dict) else {})

    if not booking:
        return jsonify({"status": "error", "message": f"No booking found for plate {plate}"}), 404

    booking_status  = booking.get('status', '')
    has_fine        = booking.get('finePending', False)
    fine_amount     = booking.get('fine', 0)
    extra_minutes   = booking.get('extraMinutes', 0)
    entry_time      = booking.get('entry_time', 0)
    booked_mins     = booking.get('bookedMinutes', 0)
    slot            = booking.get('slot', 0)
    slot_id         = booking.get('slotId', f'SLOT-{slot}')
    price           = booking.get('price', 0)
    has_referral    = booking.get('hasReferral', False)

    # ── CASE A: Fine pending — must pay before exit ───────────────────────────
    if has_fine:
        return jsonify({
            "status":        "fine_pending",
            "booking":       booking,
            "plate":         plate,
            "db_key":        db_key,
            "fine_amount":   fine_amount,
            "extra_minutes": extra_minutes,
            "message":       f"Overtime fine of ₹{fine_amount} must be cleared before exit."
        })

    # ── CASE B: Normal exit (vehicle was checked-in) ──────────────────────────
    if booking_status == 'checked-in':
        mins_used     = max(1, int((ts - entry_time) / 60)) if entry_time > 0 else 0
        refund_amount = 0.0
        refund_note   = ""

        if mins_used < booked_mins and booked_mins > 0:
            unused_mins  = booked_mins - mins_used
            unused_hours = int(unused_mins / 60)
            if unused_hours >= 1:
                refund_amount = round(unused_hours * 50 * PLATE_EXIT_REFUND_RATE, 2)
                refund_note   = f"Early exit: ₹{refund_amount:.2f} refund for {unused_hours} unused hour(s)"

        update_payload = {
            "status":         "completed",
            "active":         False,
            "actualExitTime": now_iso,
            "minutesUsed":    mins_used,
            "refundAmount":   refund_amount,
        }
        try:
            if db:
                db.reference(f'/bookings/{db_key}').update(update_payload)
                if slot > 0:
                    db.reference(f'/slots/slot{slot}').update({
                        "occupied": False, "status": "available",
                        "plate": "", "name": "", "currentVehicle": None,
                        "lastUpdated": now_iso
                    })
                    db.reference(f'/parking/slots/S{slot}').update({"active": False, "plate": ""})
                db.reference('/anpr/detections').push(
                    {"plate": plate, "type": "PLATE_EXIT", "timestamp": ts})
                if refund_amount > 0:
                    db.reference(f'/refunds/{db_key}').set({
                        "plate": plate, "type": "early_exit",
                        "amount": refund_amount, "timestamp": ts,
                        "note": refund_note, "status": "pending"
                    })
            else:
                firebase_rest_request(f'/bookings/{db_key}', 'PATCH', update_payload)
                if slot > 0:
                    firebase_rest_request(f'/slots/slot{slot}', 'PATCH', {
                        "occupied": False, "status": "available", "plate": "",
                        "name": "", "currentVehicle": None, "lastUpdated": now_iso
                    })
                    firebase_rest_request(f'/parking/slots/S{slot}', 'PATCH',
                        {"active": False, "plate": ""})
                if refund_amount > 0:
                    firebase_rest_request(f'/refunds/{db_key}', 'PUT', {
                        "plate": plate, "type": "early_exit",
                        "amount": refund_amount, "timestamp": ts,
                        "note": refund_note, "status": "pending"
                    })
            # Promote waiting list
            if slot > 0:
                try:
                    promote_waiting_list(slot)
                except Exception:
                    pass
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

        print(f"✅ [PLATE-EXIT] {plate} exited | {mins_used} min | Refund: ₹{refund_amount:.2f}", flush=True)
        return jsonify({
            "status":        "exit_ok",
            "plate":         plate,
            "booking":       booking,
            "mins_used":     mins_used,
            "refund_amount": refund_amount,
            "refund_note":   refund_note,
            "message":       f"Exit authorized. {refund_note if refund_note else 'Safe travels!'}"
        })

    # ── CASE C: Booking pending (never entered) → immediate no-show refund ────
    if booking_status in ('pending', 'active', 'waiting'):
        fee_pct       = 0.0 if has_referral else NO_SHOW_PLATFORM_FEE
        refund_amount = round(price * (1.0 - fee_pct), 2)
        fee_retained  = round(price * fee_pct, 2)
        note = (
            f"No-show exit. Full refund ₹{refund_amount:.2f} (referral applied)."
            if has_referral else
            f"No-show exit. Refund ₹{refund_amount:.2f} ({int(fee_pct*100)}% fee ₹{fee_retained:.2f} retained)."
        )
        update_payload = {
            "status": "no-show-refunded", "active": False,
            "actualExitTime": now_iso, "refundAmount": refund_amount,
            "noShowFee": fee_retained
        }
        try:
            if db:
                db.reference(f'/bookings/{db_key}').update(update_payload)
                if slot > 0:
                    db.reference(f'/slots/slot{slot}').update({
                        "occupied": False, "status": "available",
                        "plate": "", "name": "", "currentVehicle": None,
                        "lastUpdated": now_iso
                    })
                    db.reference(f'/parking/slots/S{slot}').update({"active": False, "plate": ""})
                db.reference(f'/refunds/{db_key}').set({
                    "plate": plate, "type": "no_show", "amount": refund_amount,
                    "fee_retained": fee_retained, "timestamp": ts,
                    "note": note, "status": "pending"
                })
                if slot > 0:
                    try: promote_waiting_list(slot)
                    except Exception: pass
            else:
                firebase_rest_request(f'/bookings/{db_key}', 'PATCH', update_payload)
                firebase_rest_request(f'/refunds/{db_key}', 'PUT', {
                    "plate": plate, "type": "no_show", "amount": refund_amount,
                    "fee_retained": fee_retained, "timestamp": ts,
                    "note": note, "status": "pending"
                })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

        print(f"✅ [PLATE-EXIT] {plate} no-show exit | Refund: ₹{refund_amount:.2f} | Fee: ₹{fee_retained:.2f}", flush=True)
        return jsonify({
            "status":        "no_show_refund",
            "plate":         plate,
            "refund_amount": refund_amount,
            "fee_retained":  fee_retained,
            "has_referral":  has_referral,
            "message":       note
        })

    return jsonify({"status": "error", "message": f"Booking already completed or in unknown state: {booking_status}"}), 400


# ─── Referral Code Validation ──────────────────────────────────────────────────
@app.route('/v1/referral/validate', methods=['POST'])
def validate_referral():
    """
    Validates a community referral code.
    Body JSON: { "code": "ABC123" }
    Returns: { "valid": true/false, "community_spot": "..." }
    """
    data = request.json or {}
    code = (data.get('code') or '').strip().upper()
    if not code:
        return jsonify({"valid": False, "message": "No code provided"})
    try:
        if db:
            ref_data = db.reference(f'/referral_codes/{code}').get()
        else:
            ref_data = firebase_rest_request(f'/referral_codes/{code}', 'GET')
        if ref_data and ref_data.get('active', False):
            return jsonify({
                "valid":          True,
                "community_spot": ref_data.get('spot_name', 'Community Spot'),
                "discount":       "5% no-show fee waived",
                "message":        f"✅ Referral valid! 5% no-show fee waived."
            })
        return jsonify({"valid": False, "message": "Invalid or expired referral code."})
    except Exception as e:
        return jsonify({"valid": False, "message": str(e)})


# ─── Refund Status Lookup ──────────────────────────────────────────────────────
@app.route('/v1/refund/status/<plate>', methods=['GET'])
def refund_status(plate):
    """
    Returns the pending/completed refund record for a plate.
    GET /v1/refund/status/MH12AB1234
    """
    clean_plate = re.sub(r'[^A-Z0-9]', '', plate.upper())
    try:
        if db:
            data = db.reference(f'/refunds/{clean_plate}').get()
        else:
            data = firebase_rest_request(f'/refunds/{clean_plate}', 'GET')
        if data:
            return jsonify({"status": "found", "refund": data})
        return jsonify({"status": "not_found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── No-Show Auto-Refund Background Monitor ───────────────────────────────────
#
#   Runs every 60 seconds in a daemon thread.
#   Checks all bookings where:
#     - status is "pending" or "active" (never entered)
#     - entry_time == 0 (gate was never physically triggered)
#     - current_time > booking_created + noShowWindowHours * 3600
#   On match:
#     - Calculates refund: 95% (or 100% with community referral)
#     - Updates booking to "no-show-refunded"
#     - Writes refund record to /refunds/{plate}
#     - Frees the slot
#     - Promotes waiting list

def no_show_refund_monitor():
    """
    Background daemon thread — auto-processes no-show refunds every 60 seconds.
    """
    import threading as _threading
    print("🔄 [NO-SHOW MONITOR] Started — checking every 60s", flush=True)

    while True:
        try:
            now_ts  = int(time.time())
            now_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

            if db:
                raw = db.reference('/bookings').get()
            else:
                raw = firebase_rest_request('/bookings', 'GET')

            if not isinstance(raw, dict):
                time.sleep(60)
                continue

            refunded_count = 0
            for db_key, booking in raw.items():
                if not isinstance(booking, dict):
                    continue

                status      = booking.get('status', '')
                entry_time  = booking.get('entry_time', 0)
                created_at  = booking.get('createdAt') or booking.get('timestamp')
                no_show_hrs = booking.get('noShowWindowHours', 24)
                price       = booking.get('price', 0)
                has_referral= booking.get('hasReferral', False)
                slot        = booking.get('slot', 0)
                plate       = booking.get('plate', db_key)

                # Only process pending/active bookings that were never entered
                if status not in ('pending', 'active'):
                    continue
                if entry_time and entry_time > 0:
                    continue
                if not created_at or price <= 0:
                    continue

                # Parse created_at (ISO string or unix timestamp)
                try:
                    if isinstance(created_at, (int, float)):
                        created_ts = int(created_at)
                        if created_ts > 1e12:  # milliseconds
                            created_ts //= 1000
                    else:
                        from datetime import datetime, timezone
                        dt = datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                        created_ts = int(dt.timestamp())
                except Exception:
                    continue

                window_secs  = int(no_show_hrs) * 3600
                deadline_ts  = created_ts + window_secs

                if now_ts < deadline_ts:
                    continue  # Window not yet expired

                # ── Process no-show refund ────────────────────────────────────
                fee_pct       = 0.0 if has_referral else NO_SHOW_PLATFORM_FEE
                refund_amount = round(price * (1.0 - fee_pct), 2)
                fee_retained  = round(price * fee_pct, 2)
                note = (
                    f"Auto no-show: full refund ₹{refund_amount:.2f} (referral applied)."
                    if has_referral else
                    f"Auto no-show: refund ₹{refund_amount:.2f}, fee ₹{fee_retained:.2f} retained."
                )

                update_payload = {
                    "status":        "no-show-refunded",
                    "active":        False,
                    "refundAmount":  refund_amount,
                    "noShowFee":     fee_retained,
                    "processedAt":   now_iso,
                }
                refund_record = {
                    "plate":        plate,
                    "type":         "no_show_auto",
                    "amount":       refund_amount,
                    "fee_retained": fee_retained,
                    "timestamp":    now_ts,
                    "note":         note,
                    "status":       "pending",
                    "auto":         True,
                }

                try:
                    if db:
                        db.reference(f'/bookings/{db_key}').update(update_payload)
                        db.reference(f'/refunds/{re.sub(chr(91)+chr(93), "", plate)}').set(refund_record)
                        if slot > 0:
                            db.reference(f'/slots/slot{slot}').update({
                                "occupied": False, "status": "available",
                                "plate": "", "name": "", "currentVehicle": None,
                                "lastUpdated": now_iso
                            })
                            db.reference(f'/parking/slots/S{slot}').update({"active": False, "plate": ""})
                            try: promote_waiting_list(slot)
                            except Exception: pass
                    else:
                        firebase_rest_request(f'/bookings/{db_key}', 'PATCH', update_payload)
                        firebase_rest_request(f'/refunds/{plate}', 'PUT', refund_record)

                    refunded_count += 1
                    print(f"  ✅ [NO-SHOW] Auto-refunded {plate}: ₹{refund_amount:.2f} (fee: ₹{fee_retained:.2f})", flush=True)

                except Exception as e:
                    print(f"  ⚠️ [NO-SHOW] Error refunding {plate}: {e}", flush=True)

            if refunded_count:
                print(f"  📊 [NO-SHOW] Cycle complete — {refunded_count} refund(s) processed", flush=True)

        except Exception as e:
            print(f"⚠️ [NO-SHOW MONITOR] Cycle error: {e}", flush=True)

        time.sleep(60)


# Start the no-show monitor daemon thread on import
import threading as _threading
_no_show_thread = _threading.Thread(target=no_show_refund_monitor, daemon=True, name="NoShowMonitor")
_no_show_thread.start()

# ─── Walk-In Mode (Gap #6: Unregistered / Guest Vehicle Entry) ────────────────

#
#   Allows vehicles WITHOUT a pre-booking to enter.
#   On entry:
#     → Dynamically assigns the best available slot.
#     → Creates a guest booking record in Firebase.
#     → Generates a unique guest QR token for exit.
#     → Returns slot number + token to the caller (NodeMCU).
#   On exit (via standard QR scan):
#     → Calculates stay duration and amount owed (₹50/hr base rate).
#     → Generates a payment link entry in Firebase for dashboard.
#     → Marks guest booking as completed.

WALK_IN_RATE_PER_HOUR = 50  # ₹ per hour for unregistered / walk-in vehicles

@app.route('/v1/walk-in/entry', methods=['POST'])
def walk_in_entry():
    """
    Register a walk-in (unregistered) vehicle and assign a slot.
    Body JSON: { "plate": "MH12AB1234" }  (plate from ANPR, may be partial/unverified)
    Returns:   { "status": "ok", "slot": 2, "token": "WI-XXXXXX", "rate": 50 }
    """
    data  = request.json or {}
    plate = re.sub(r'[^A-Z0-9]', '', (data.get('plate') or 'GUEST').upper())
    if not plate:
        plate = 'GUEST'

    ts    = int(time.time())
    token = f"WI-{uuid.uuid4().hex[:6].upper()}"  # Unique walk-in QR token

    # ── Find a free slot from Firebase ──────────────────────────────────
    assigned_slot = None
    try:
        if db:
            slots_data = db.reference('/slots').get()
        else:
            slots_data = firebase_rest_request('/slots', 'GET')

        if isinstance(slots_data, dict):
            for slot_key, slot_val in slots_data.items():
                if slot_val and slot_val.get('status', 'available') == 'available':
                    assigned_slot = int(re.sub(r'[^0-9]', '', slot_key) or 0)
                    if assigned_slot > 0:
                        break
        elif isinstance(slots_data, list):
            for idx, slot_val in enumerate(slots_data):
                if slot_val and slot_val.get('status', 'available') == 'available':
                    assigned_slot = idx + 1
                    break
    except Exception as e:
        print(f"  ⚠️ [WALK-IN] Slot fetch error: {e}", flush=True)

    if assigned_slot is None:
        print(f"  ⚠️ [WALK-IN] No free slots available for guest {plate}", flush=True)
        return jsonify({"status": "full", "message": "Parking lot is full. No slots available."}), 200

    # ── Create guest booking record ──────────────────────────────────────
    guest_booking = {
        "plate":        plate,
        "name":         f"Walk-In Guest",
        "token":        token,
        "slot":         assigned_slot,
        "slotId":       f"SLOT-{assigned_slot}",
        "status":       "checked-in",
        "active":       True,
        "isWalkIn":     True,
        "paid":         False,
        "entry_time":   ts,
        "bookedMinutes": 0,        # 0 = open-ended, charged on exit
        "ratePerHour":  WALK_IN_RATE_PER_HOUR,
        "mobile":       data.get('mobile', ''),
        "createdAt":    ts
    }

    try:
        if db:
            db.reference(f'/walk_ins/{token}').set(guest_booking)
            db.reference(f'/slots/slot{assigned_slot}').update({
                "occupied":       True,
                "status":         "occupied",
                "plate":          plate,
                "currentVehicle": plate,
                "isWalkIn":       True,
                "lastUpdated":    time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            })
            db.reference(f'/parking/slots/S{assigned_slot}').update({
                "active": True,
                "plate":  plate
            })
        else:
            firebase_rest_request(f'/walk_ins/{token}', 'PUT', guest_booking)
            firebase_rest_request(f'/slots/slot{assigned_slot}', 'PATCH', {
                "occupied": True, "status": "occupied",
                "plate": plate, "currentVehicle": plate,
                "isWalkIn": True,
                "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            })
            firebase_rest_request(f'/parking/slots/S{assigned_slot}', 'PATCH', {
                "active": True, "plate": plate
            })
    except Exception as e:
        print(f"  ⚠️ [WALK-IN] Firebase write error: {e}", flush=True)

    print(f"✅ [WALK-IN] Guest {plate} assigned Slot {assigned_slot} | Token: {token}", flush=True)
    return jsonify({
        "status":  "ok",
        "plate":   plate,
        "slot":    assigned_slot,
        "token":   token,
        "rate":    WALK_IN_RATE_PER_HOUR,
        "message": f"Welcome! You have been assigned Slot {assigned_slot}."
    })


@app.route('/v1/walk-in/exit', methods=['POST'])
def walk_in_exit():
    """
    Process walk-in vehicle exit.
    Body JSON: { "token": "WI-XXXXXX" }
    Returns:   { "status": "ok", "amount_due": 150, "plate": "...", "minutes_used": 180 }
    """
    data  = request.json or {}
    token = data.get('token', '').strip().upper()
    if not token:
        return jsonify({"status": "error", "message": "Missing token"}), 400

    ts = int(time.time())

    try:
        if db:
            booking = db.reference(f'/walk_ins/{token}').get()
        else:
            booking = firebase_rest_request(f'/walk_ins/{token}', 'GET')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    if not booking:
        return jsonify({"status": "error", "message": "Token not found"}), 404

    entry_time    = booking.get('entry_time', ts)
    mins_used     = max(1, int((ts - entry_time) / 60))
    hours_used    = mins_used / 60.0
    amount_due    = round(hours_used * WALK_IN_RATE_PER_HOUR, 2)
    plate         = booking.get('plate', 'GUEST')
    assigned_slot = booking.get('slot', 0)

    # ── Update guest booking as completed ────────────────────────────────
    exit_update = {
        "active":          False,
        "status":          "completed",
        "exit_time":       ts,
        "minutesUsed":     mins_used,
        "amountDue":       amount_due,
        "paymentPending":  True,
        "actualExitTime":  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }

    try:
        if db:
            db.reference(f'/walk_ins/{token}').update(exit_update)
            if assigned_slot > 0:
                db.reference(f'/slots/slot{assigned_slot}').update({
                    "occupied": False, "status": "available",
                    "plate": "", "currentVehicle": None,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
                db.reference(f'/parking/slots/S{assigned_slot}').update({
                    "active": False, "plate": ""
                })
        else:
            firebase_rest_request(f'/walk_ins/{token}', 'PATCH', exit_update)
            if assigned_slot > 0:
                firebase_rest_request(f'/slots/slot{assigned_slot}', 'PATCH', {
                    "occupied": False, "status": "available",
                    "plate": "", "currentVehicle": None,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
                firebase_rest_request(f'/parking/slots/S{assigned_slot}', 'PATCH', {
                    "active": False, "plate": ""
                })
    except Exception as e:
        print(f"  ⚠️ [WALK-IN EXIT] Firebase error: {e}", flush=True)

    print(f"✅ [WALK-IN EXIT] {plate} | {mins_used} min | ₹{amount_due:.2f} due | Token: {token}", flush=True)

    return jsonify({
        "status":       "ok",
        "plate":        plate,
        "token":        token,
        "minutes_used": mins_used,
        "amount_due":   amount_due,
        "message":      f"Thank you! Amount due: ₹{amount_due:.2f} for {mins_used} minutes."
    })


# ─── Automated Post-Exit Payment Notification (Gap #3) ────────────────────────
#
#   Called automatically from proxy_log_exit after every registered exit.
#   Calculates:
#     - Minutes actually used (actual_mins = exit_time - entry_time)
#     - Base charge   = (actual_mins / 60) × ₹50
#     - Overtime fine = overstay_mins × ₹3  (if applicable, from overtimeMonitor)
#     - Refund credit = unused full hours × ₹50 × 80%  (if left early)
#     - Net payable   = base_charge + fine - refund_credit
#   Writes full payment summary to Firebase under /bookings/{key}/paymentSummary.
#   Logs a simulated payment link to console.

RATE_PER_HOUR   = 50    # ₹ per hour base rate (pre-booking)
REFUND_RATE     = 0.80  # 80% refund on unused full hours
FINE_PER_MINUTE = 3     # ₹ per minute overtime fine

def calculate_and_push_payment_summary(plate, booking, actual_exit_ts):
    """
    Generates and pushes a full payment summary to Firebase for a booking.
    Called asynchronously after a registered vehicle exits.
    """
    try:
        entry_time    = booking.get('entry_time', 0)
        booked_mins   = booking.get('bookedMinutes', 0) or (int(booking.get('duration', 1)) * 60)
        fine_pending  = booking.get('finePending', False)
        existing_fine = booking.get('fine', 0)
        mobile        = booking.get('mobile') or booking.get('phone') or ''
        name          = booking.get('name', 'Driver')
        slot_id       = booking.get('slotId') or f"SLOT-{booking.get('slot', 'X')}"

        if entry_time <= 0 or booked_mins <= 0:
            return  # Not enough data to calculate

        actual_mins  = max(1, int((actual_exit_ts - entry_time) / 60))
        base_charge  = round((actual_mins / 60.0) * RATE_PER_HOUR, 2)

        # Refund for early exit (unused complete hours only)
        refund_credit = 0.0
        if actual_mins < booked_mins:
            unused_mins  = booked_mins - actual_mins
            unused_hours = int(unused_mins / 60)
            if unused_hours >= 1:
                refund_credit = round(unused_hours * RATE_PER_HOUR * REFUND_RATE, 2)

        # Overtime fine (already tracked by overtimeMonitor; use stored value)
        fine_amount = existing_fine if fine_pending else 0

        net_payable = round(base_charge + fine_amount - refund_credit, 2)
        net_payable = max(0.0, net_payable)  # Cannot be negative

        payment_summary = {
            "plate":          plate,
            "name":           name,
            "slotId":         slot_id,
            "entry_time":     entry_time,
            "exit_time":      actual_exit_ts,
            "actual_minutes": actual_mins,
            "booked_minutes": booked_mins,
            "base_charge":    base_charge,
            "overtime_fine":  fine_amount,
            "refund_credit":  refund_credit,
            "net_payable":    net_payable,
            "currency":       "INR",
            "rate_per_hour":  RATE_PER_HOUR,
            "payment_status": "PENDING" if net_payable > 0 else "SETTLED",
            "generated_at":   actual_exit_ts
        }

        # Console payment link simulation (replace with Razorpay API for production)
        print(f"\n  💳 [PAYMENT SUMMARY] ─────────────────────────────", flush=True)
        print(f"     Vehicle   : {plate} ({name})", flush=True)
        print(f"     Slot      : {slot_id}", flush=True)
        print(f"     Duration  : {actual_mins} min (booked: {booked_mins} min)", flush=True)
        print(f"     Base      : ₹{base_charge:.2f}", flush=True)
        if fine_amount > 0:
            print(f"     Fine      : ₹{fine_amount:.2f} (overtime)", flush=True)
        if refund_credit > 0:
            print(f"     Refund    : -₹{refund_credit:.2f} (early exit credit)", flush=True)
        print(f"     NET DUE   : ₹{net_payable:.2f}", flush=True)
        if mobile:
            # In production: call Razorpay Payment Links API here
            # razorpay_client.payment_link.create({...})
            print(f"     📱 Payment notification → {mobile}", flush=True)
        print(f"  ──────────────────────────────────────────────────\n", flush=True)

        # Push to Firebase
        with cache_lock:
            db_key = bookings_original_keys.get(plate, plate)

        if db:
            db.reference(f'/bookings/{db_key}/paymentSummary').set(payment_summary)
        else:
            firebase_rest_request(f'/bookings/{db_key}/paymentSummary', 'PUT', payment_summary)

        print(f"  📡 [PAYMENT] Firebase payment summary pushed for {plate}", flush=True)

    except Exception as e:
        print(f"  ⚠️ [PAYMENT] Error generating summary: {e}", flush=True)


# Patch proxy_log_exit to call payment summary generator asynchronously
_original_proxy_log_exit = proxy_log_exit.__wrapped__ if hasattr(proxy_log_exit, '__wrapped__') else None

@app.route('/v1/proxy/log-exit-v2', methods=['POST'])
def proxy_log_exit_v2():
    """
    Enhanced exit logging with full payment summary generation.
    Drop-in replacement for /v1/proxy/log-exit.
    Body JSON: { "plate": "MH12AB1234" }
    """
    data  = request.json or {}
    plate = data.get('plate')
    if not plate:
        return jsonify({"status": "error", "message": "Missing plate"}), 400

    ts = int(time.time())

    try:
        # Fetch full booking data for payment calculation
        with cache_lock:
            booking_data = bookings_cache.get(re.sub(r'[^A-Z0-9]', '', plate.upper()), {})
            db_key       = bookings_original_keys.get(re.sub(r'[^A-Z0-9]', '', plate.upper()), plate)

        if db:
            db.reference('/anpr/detections').push({
                "plate": plate, "type": "EXIT", "timestamp": ts
            })
            ref     = db.reference(f'/bookings/{db_key}')
            booking = ref.get() or booking_data

            vacated_slot = booking.get('slot', 0) if booking else 0

            # Calculate early exit refund
            refund_amount = 0.0
            mins_used     = 0
            if booking:
                entry_time  = booking.get('entry_time', 0)
                mins_booked = booking.get('bookedMinutes', 0)
                if entry_time > 0 and mins_booked > 0:
                    mins_used = int((ts - entry_time) / 60)
                    if mins_used < mins_booked:
                        unused_mins  = mins_booked - mins_used
                        unused_hours = int(unused_mins / 60)
                        if unused_hours >= 1:
                            refund_amount = float(unused_hours * RATE_PER_HOUR * REFUND_RATE)

            ref.update({
                "active":         False,
                "status":         "completed",
                "actualExitTime": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "refundAmount":   refund_amount,
                "minutesUsed":    mins_used
            })

            if vacated_slot > 0:
                db.reference(f'/parking/slots/S{vacated_slot}').update({"active": False, "plate": ""})
                db.reference(f'/slots/slot{vacated_slot}').update({
                    "occupied": False, "status": "available",
                    "plate": "", "name": "", "currentVehicle": None,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
                promote_waiting_list(vacated_slot)

            # Push payment summary asynchronously
            def async_payment(bk, pl, exit_ts):
                calculate_and_push_payment_summary(pl, bk, exit_ts)
            threading.Thread(
                target=async_payment,
                args=(booking or {}, plate, ts),
                daemon=True
            ).start()

        else:
            # REST fallback path
            firebase_rest_request('/anpr/detections', 'POST', {
                "plate": plate, "type": "EXIT", "timestamp": ts
            })
            booking      = firebase_rest_request(f'/bookings/{db_key}', 'GET') or booking_data
            vacated_slot = booking.get('slot', 0) if booking else 0

            refund_amount = 0.0
            mins_used     = 0
            if booking:
                entry_time  = booking.get('entry_time', 0)
                mins_booked = booking.get('bookedMinutes', 0)
                if entry_time > 0 and mins_booked > 0:
                    mins_used = int((ts - entry_time) / 60)
                    if mins_used < mins_booked:
                        unused_mins  = mins_booked - mins_used
                        unused_hours = int(unused_mins / 60)
                        if unused_hours >= 1:
                            refund_amount = float(unused_hours * RATE_PER_HOUR * REFUND_RATE)

            firebase_rest_request(f'/bookings/{db_key}', 'PATCH', {
                "active": False, "status": "completed",
                "actualExitTime": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "refundAmount":   refund_amount,
                "minutesUsed":    mins_used
            })

            if vacated_slot > 0:
                firebase_rest_request(f'/parking/slots/S{vacated_slot}', 'PATCH', {"active": False, "plate": ""})
                firebase_rest_request(f'/slots/slot{vacated_slot}', 'PATCH', {
                    "occupied": False, "status": "available", "plate": "", "name": "",
                    "currentVehicle": None,
                    "lastUpdated": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                })
                promote_waiting_list(vacated_slot)

            def async_payment_rest(bk, pl, exit_ts):
                calculate_and_push_payment_summary(pl, bk, exit_ts)
            threading.Thread(
                target=async_payment_rest,
                args=(booking or {}, plate, ts),
                daemon=True
            ).start()

        print(f"✅ [EXIT-V2] Logged exit for {plate} with payment summary", flush=True)
        return jsonify({"status": "ok"})

    except Exception as e:
        import traceback
        print(f"  ⚠️ [EXIT-V2] Error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── Walk-In Status Lookup ─────────────────────────────────────────────────────
@app.route('/v1/walk-in/status', methods=['GET'])
def walk_in_status():
    """Lookup a walk-in session by token. Used by dashboard or NodeMCU."""
    token = request.args.get('token', '').strip().upper()
    if not token:
        return jsonify({"status": "error", "message": "Missing token"}), 400
    try:
        if db:
            booking = db.reference(f'/walk_ins/{token}').get()
        else:
            booking = firebase_rest_request(f'/walk_ins/{token}', 'GET')
        if booking:
            return jsonify({"status": "ok", "booking": booking})
        return jsonify({"status": "not_found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        from waitress import serve
        print("\n" + "=" * 70)
        print("  🚗 ParkIN Detection Server — STABLE WAITRESS EDITION")
        print("  📈 Optimized TCP Handling | Concurrent Processing")
        print("=" * 70 + "\n")
        serve(app, host='0.0.0.0', port=5000, threads=12)
    except ImportError:
        print("\n" + "=" * 70)
        print("  🚗 ParkIN Detection Server — FLASK DEV MODE")
        print("  ⚠️  Waitress not found. Using fallback server.")
        print("=" * 70 + "\n")
        app.run(host='0.0.0.0', port=5000, threaded=True)
