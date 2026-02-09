import cv2
import numpy as np

def correct_fisheye(frame, enabled=True, K=None, D=None, balance=0.0, dim=None):
    """
    Correct fisheye distortion in a frame.
    Args:
        frame: Input image (numpy array)
        enabled: If False, returns frame unchanged
        K: Camera matrix (3x3), if None uses default
        D: Distortion coefficients (4x1), if None uses default
        balance: Balance between crop and FOV (0=more crop, 1=more FOV)
        dim: Image dimensions (width, height), if None uses frame.shape
    Returns:
        Undistorted image (numpy array)
    """
    if not enabled:
        return frame
    h, w = frame.shape[:2]
    if dim is None:
        dim = (w, h)
    # Default calibration (for generic webcam, not perfect)
    if K is None:
        K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
    if D is None:
        D = np.array([[-0.1], [0.01], [0.0], [0.0]], dtype=np.float32)  # Mild fisheye
    # Generate new camera matrix
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted

def adjust_image(frame, brightness=1.0, contrast=1.0, saturation=1.0):
    """
    Adjust brightness, contrast, and saturation of an image.
    Args:
        frame: Input image (numpy array, BGR)
        brightness: Multiplier for brightness (1.0 = no change)
        contrast: Multiplier for contrast (1.0 = no change)
        saturation: Multiplier for saturation (1.0 = no change)
    Returns:
        Adjusted image (numpy array)
    """
    img = frame.astype(np.float32)
    # Brightness
    img = img * brightness
    # Contrast
    mean = np.mean(img, axis=(0,1), keepdims=True)
    img = (img - mean) * contrast + mean
    # Saturation (convert to HSV)
    img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= saturation
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img 