import cv2

def clean_ocr_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding and noise reduction
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh