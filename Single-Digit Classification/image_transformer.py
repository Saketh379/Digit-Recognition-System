import cv2
import numpy as np

def transform_image(image, scale = 1.0, angle = 0):
    """
    Apply scaling, and rotation to an image.
    The final output will have the same dimensions as the input image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Apply scaling and rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    transformed = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0)
    return transformed
    