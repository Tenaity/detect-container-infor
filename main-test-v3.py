import os
import cv2
import pytesseract
import torch
import time
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from preprocessing import clean_ocr_image
from utils import crop_image_with_box

# --- CONFIG ---
MODEL_PATH = "/Users/tenaity/Documents/MSE/asignment/yolo_runs/v3/weights/best.pt"
IMAGE_PATH = '/Users/tenaity/Documents/MSE/asignment/test_image'
DEVICE = "mps"
OUTPUT_DIR = "output"
CSV_OUTPUT = "ocr_results.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Initialize models ---
print("[INFO] Loading detection model...")
model = YOLO(MODEL_PATH).to(DEVICE)

print("[INFO] Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
trocr_model.to(DEVICE).eval() # type: ignore

# --- OCR functions ---
def tesseract_ocr(image):
    config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return pytesseract.image_to_string(image, config=config).strip()

def trocr_ocr(image):
    if image.ndim == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = image
    pil_img = Image.fromarray(img_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE) # type: ignore
    with torch.no_grad():
        generated_ids = trocr_model.generate(**inputs) # type: ignore
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip() # type: ignore

# --- Main pipeline ---
records = []
for fname in os.listdir(IMAGE_PATH):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_PATH, fname)
    image = cv2.imread(img_path)
    detections = model(image)[0]

    field_text = {"dv": "", "owner": "", "serial": "", "size": ""}
    tess_time_total = 0
    trocr_time_total = 0

    for box in detections.boxes:
        cls = int(box.cls.item())
        label = model.names[cls]
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        cropped = crop_image_with_box(image, xyxy)
        cleaned = clean_ocr_image(cropped)

        start = time.time()
        tess_text = tesseract_ocr(cleaned)
        tess_time_total += time.time() - start

        start = time.time()
        trocr_text = trocr_ocr(cleaned)
        trocr_time_total += time.time() - start

        field_text[label] = trocr_text if len(trocr_text) > len(tess_text) else tess_text

    container_code = field_text["dv"] + field_text["owner"] + field_text["serial"]
    record = {
        "image": fname,
        "dv": field_text["dv"],
        "owner": field_text["owner"],
        "serial": field_text["serial"],
        "size": field_text["size"],
        "container_code": container_code,
        "tesseract_time": round(tess_time_total, 3),
        "trocr_time": round(trocr_time_total, 3)
    }
    records.append(record)

# Export CSV
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, CSV_OUTPUT), index=False)
print(f"[INFO] Exported to {os.path.join(OUTPUT_DIR, CSV_OUTPUT)}")