import os
import cv2
import pytesseract
import torch
import time
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from utils import crop_image_with_box
from preprocessing import clean_ocr_image

# --- CONFIG ---
MODEL_PATH = "/Users/tenaity/Documents/MSE/asignment/yolo_runs/v3/weights/best.pt"
IMAGE_PATH = "/Users/tenaity/Documents/MSE/asignment/test_image"
DEVICE = "mps"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Initialize models ---
print("[INFO] Loading models...")
model = YOLO(MODEL_PATH).to(DEVICE)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').eval()
# easyocr_reader = easyocr.Reader(['en'], gpu=False)

# --- OCR functions ---
def tesseract_ocr(image):
    config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return pytesseract.image_to_string(image, config=config).strip()

# def easyocr_ocr(image):
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     result = easyocr_reader.readtext(rgb, detail=0)
#     return ' '.join(result).strip() # type: ignore

def trocr_ocr(image):
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt") # type: ignore
    with torch.no_grad():
        generated_ids = trocr_model.generate(**inputs) # type: ignore
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip() # type: ignore

# --- Main processing ---
results = []
for fname in os.listdir(IMAGE_PATH):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"📷 Processing: {fname}")
    true_code = os.path.splitext(fname)[0]
    image = cv2.imread(os.path.join(IMAGE_PATH, fname))
    detections = model(image)[0]

    for box in detections.boxes:
        label = model.names[int(box.cls.item())]
        if label not in ["dv", "owner", "serial", "size"]:
            continue

        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        cropped = crop_image_with_box(image, xyxy)
        cleaned = clean_ocr_image(cropped)

        # Tesseract
        t1 = time.perf_counter()
        tess = tesseract_ocr(cleaned)
        t2 = time.perf_counter()

        # EasyOCR
        # t3 = time.perf_counter()
        # easy = easyocr_ocr(cropped)
        # t4 = time.perf_counter()

        # TrOCR
        t5 = time.perf_counter()
        trocr = trocr_ocr(cropped)
        t6 = time.perf_counter()

        results.append({
            "image": fname,
            "ground_truth": true_code,
            "field": label,
            "tesseract": tess,
            "tesseract_time": round(t2 - t1, 3),
            # "easyocr": easy,
            # "easyocr_time": round(t4 - t3, 3),
            "trocr": trocr,
            "trocr_time": round(t6 - t5, 3),
        })

# Save results
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "ocr_results_benchmark.csv")
df.to_csv(csv_path, index=False)
print(f"✅ DONE. Results saved to: {csv_path}")