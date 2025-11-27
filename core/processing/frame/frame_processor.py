import cv2
import numpy as np
from torchvision.io import read_image
import torch

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("FrameProcessor")

class FrameProcessor:
    """
    Performs frame preprocessing:
    - read image file (Torch + OpenCV)
    - color space conversion
    - detect white object in HSV space
    - extract contours and corner points
    - prepare output frame for display
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess(self, img_path):
        """
        Process a single image.

        Returns:
        current_gray (np.ndarray)
        white_mask (np.ndarray)
        latest_corners (np.ndarray or None)
        output_frame (np.ndarray)
        quad_count (int)
        corner_count (int)
        """
        try:
            # --- 1. Scan image ---
            current_tensor = read_image(img_path).to(self.device).float() / 255.0
            current_frame_rgb = (current_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            current_frame = cv2.cvtColor(current_frame_rgb, cv2.COLOR_RGB2BGR)
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # --- 2. White object detection ---
            hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
            contours, _ = cv2.findContours(white_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            output_frame = current_frame.copy()
            quad_count = 0
            corner_count = 0
            MIN_AREA_THRESHOLD = 50
            MAX_ASPECT_RATIO = 5.0
            latest_corners = None

            # --- 3. Finding contours and corner points ---
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_AREA_THRESHOLD:
                    continue
                cv2.drawContours(output_frame, [contour], -1, (0, 255, 0), 2)
                rect = cv2.minAreaRect(contour)
                (center, (width, height), angle) = rect
                if min(width, height) > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                else:
                    aspect_ratio = MAX_ASPECT_RATIO + 1
                if aspect_ratio > MAX_ASPECT_RATIO:
                    continue
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                quad_count += 1
                corner_count += len(box)
                latest_corners = box.copy()
                for point in box:
                    x, y = point
                    cv2.circle(output_frame, (int(x), int(y)), 6, (0, 0, 255), -1)

            return current_gray, white_mask, latest_corners, output_frame, quad_count, corner_count

        except Exception as e:
            LOG.error(f"Error processing frame ({img_path}): {e}")
            return None, None, None, None, 0, 0
