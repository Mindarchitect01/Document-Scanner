import cv2
import numpy as np
from ultralytics import YOLO

def detect_receipt(image_path, use_yolo=True):
    """
    Detects receipt in image. Returns the image and the quadrilateral corners (numpy array of 4 points).
    Tries YOLOv8 first; if it fails or is disabled, falls back to OpenCV contour detection.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    pts = None

    # Attempt YOLOv8 detection
    if use_yolo:
        model = YOLO('yolov8n.pt')  # or a custom checkpoint
        results = model(image)
        # If YOLO detected something with sufficient confidence
        if results and len(results[0].boxes) > 0:
            # Take the first detected box (x1,y1,x2,y2)
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            # Construct 4 corner points of the bounding box
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int0)
            return image, pts

    # Fallback: OpenCV contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        # Use minAreaRect to get a rotated box, then boxPoints to get corners
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        pts = np.int0(box)
        return image, pts

    # If detection fails, return original image and None
    return image, None