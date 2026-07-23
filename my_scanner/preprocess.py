import cv2

def preprocess_image(image):
    """
    Converts to grayscale, blurs, and applies adaptive threshold to binarize.
    Returns a binary (black-white) image optimized for OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Denoise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 15)
    return thresh
