import cv2
import pytesseract
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt

# Load TrOCR
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
model.eval()

# Clean OCR image
def clean_ocr_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Tesseract OCR
def tesseract_ocr(image):
    config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return pytesseract.image_to_string(image, config=config).strip()

# TrOCR
def trocr_ocr(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    with torch.no_grad():
        pixel_values = processor(images=img, return_tensors="pt").pixel_values # type: ignore
        generated_ids = model.generate(pixel_values) # type: ignore
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip() # type: ignore

# Main
path = "/Users/tenaity/Documents/MSE/asignment/test_image/oolu1760406.jpg" 
image = cv2.imread(path)
cleaned = clean_ocr_image(image)

# OCR
tess_result = tesseract_ocr(cleaned)
trocr_result = trocr_ocr(cleaned)

# Display
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image"); axs[0].axis("off")
axs[1].imshow(cleaned, cmap="gray")
axs[1].set_title(f"Tesseract: {tess_result}"); axs[1].axis("off")
axs[2].imshow(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB))
axs[2].set_title(f"TrOCR: {trocr_result}"); axs[2].axis("off")
plt.tight_layout()
plt.show()