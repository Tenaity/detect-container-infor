import cv2

def crop_image_with_box(img, box, padding=5):
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, w)
    y2 = min(y2 + padding, h)
    return img[y1:y2, x1:x2]

# def draw_box_and_label(img, box, label):
#     x1, y1, x2, y2 = map(int, box)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(img, label, (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_box_and_label(image, xyxy, label, fields=None):
    x1, y1, x2, y2 = xyxy
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    text_lines = [f"{label}"]
    if fields:
        if fields.get("container_code"):
            text_lines.append(f"Code: {fields['container_code']}")
        if fields.get("gross_weight"):
            text_lines.append(f"Gross: {fields['gross_weight']}kg")
        if fields.get("tare_weight"):
            text_lines.append(f"Tare: {fields['tare_weight']}kg")

    y_offset = y1 - 10
    for line in text_lines:
        cv2.putText(image, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset -= 20  # mỗi dòng cách nhau 20px