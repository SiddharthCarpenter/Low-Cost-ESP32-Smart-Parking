from ultralytics import YOLO

# 1. Load the original PyTorch model
print("Loading FP32 PyTorch model...")
model = YOLO("runs/detect/yolov8_anpr_custom2/weights/last.pt")

# 2. Export to INT8 OpenVINO
print("\n--- EXPORTING TO INT8 OPENVINO ---")
exported_path = model.export(format="openvino", int8=True, data="ANPR_dataset/data.yaml", imgsz=640)
print(f"Model successfully exported to: {exported_path}")

# 3. Load the new INT8 TFLite model
print("\n--- LOADING INT8 MODEL ---")
int8_model = YOLO(exported_path)

# 4. Validate the INT8 model to get accuracy and latency
print("\n--- VALIDATING INT8 MODEL ---")
metrics = int8_model.val(data="ANPR_dataset/data.yaml", imgsz=640)

print("\n================ INT8 FINAL RESULTS ================")
print(f"mAP50 (Accuracy): {metrics.box.map50 * 100:.2f}%")
print(f"mAP50-95:         {metrics.box.map * 100:.2f}%")
print(f"Precision:        {metrics.box.mp * 100:.2f}%")
print(f"Recall:           {metrics.box.mr * 100:.2f}%")
print("====================================================")
