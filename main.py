# 📁 container_ocr_v2/main.py

from ultralytics import YOLO
import cv2
import os
import pytesseract
import re
from utils import crop_image_with_box, draw_box_and_label
from preprocessing import clean_ocr_image

# --- CONFIG ---
MODEL_PATH = "/Users/tenaity/Documents/MSE/asignment/yolo_runs/container-code-detector-v1/weights/best.pt"
IMAGE_PATH = '/Users/tenaity/Documents/MSE/asignment/test_image'
DEVICE = "mps"  # For Macbook M1/M2, use "mps" for Metal Performance Shaders
OUTPUT_DIR = "output"

# --- INITIALIZE ---
model = YOLO(MODEL_PATH)
model.to(DEVICE)

# --- OCR LOGIC ---
def extract_text(img):
    config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./\n"
    text = pytesseract.image_to_string(img, config=config)
    return text

def extract_fields(text):
    container_code = re.findall(r"[A-Z]{4}\s?\d{6}\s?\d", text.upper())
    tare_weight = re.findall(r"TARE[^\d]*(\d+[\.,]?\d*)\s*(KGS|KG)?", text.upper())
    gross_weight = re.findall(r"GROSS[^\d]*(\d+[\.,]?\d*)\s*(KGS|KG)?", text.upper())
    return {
        "container_code": container_code[0] if container_code else "",
        "tare_weight": tare_weight[0][0] if tare_weight else "",
        "gross_weight": gross_weight[0][0] if gross_weight else ""
    }

# --- PROCESS ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

for fname in os.listdir(IMAGE_PATH):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_PATH, fname)
    image = cv2.imread(img_path)
    print(f"\n📷 Processing {fname}...")

    detections = model(image)[0]
    result = {"image": fname, "fields": []}

    for box in detections.boxes:
        cls = int(box.cls.item())
        label = model.names[cls]
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]

        cropped = crop_image_with_box(image, xyxy)
        cleaned = clean_ocr_image(cropped)
        text = extract_text(cleaned)
        fields = extract_fields(text)

        result["fields"].append({"label": label, "text": text.strip(), **fields})
        draw_box_and_label(image, xyxy, label, fields)

    results.append(result)

    output_file = os.path.join(OUTPUT_DIR, f"annotated_{fname}")
    cv2.imwrite(output_file, image)
    print(f"[INFO] Saved to {output_file}")

print("\n✅ DONE. Check output/ folder.")