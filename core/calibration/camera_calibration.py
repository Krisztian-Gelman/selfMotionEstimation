import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("Calibration")


class CameraCalibrator:
    """
    CameraCalibrator
    -----------------
    Handles classical chessboard-based camera calibration.
    Automatically searches for checkerboard patterns across multiple images
    and performs full calibration (K, D, RMS error, etc.).
    """

    def __init__(self,
                 image_dir: str = None,
                 scale_factor: float = 2.0,
                 max_pattern: tuple = (12, 8),
                 square_size: float = 22.0):
        """
        Initialize calibration parameters.

        :param image_dir: Path to folder containing calibration images.
        :param scale_factor: Pre-scaling factor for chessboard detection.
        :param max_pattern: Maximum checkerboard size (cols, rows).
        :param square_size: Physical size of one checker square (in mm).
        """
        self.image_dir = image_dir
        self.scale_factor = scale_factor
        self.max_pattern = max_pattern
        self.square_size = square_size

        self.possible_patterns = [
            (cols, rows)
            for cols in range(3, self.max_pattern[0] + 1)
            for rows in range(3, self.max_pattern[1] + 1)
        ]
        # Largest first
        self.possible_patterns.sort(key=lambda x: x[0] * x[1], reverse=True)

        self.objpoints = []
        self.imgpoints = []
        self.image_shape = None

        LOG.info(f"[CameraCalibrator] Initialized with path: {self.image_dir}")

    # ------------------------------------------------------------
    # Camera calibration helper
    # ------------------------------------------------------------
    def calibrate_camera(self, objpoints, imgpoints, image_shape):
        """
        Calibrate the camera based on the collected checkerboard points.

        Returns: ret, mtx, dist, rvecs, tvecs
        """
        LOG.info("\nCamera calibration in progress...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape, None, None
        )

        LOG.info("\nCalibration complete.")
        LOG.info(f"Reprojection error: {ret}")
        LOG.info(f"Camera matrix (mtx):\n{mtx}")
        LOG.info(f"Distortion coefficients (dist):\n{dist}")

        # Backprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error /= len(objpoints)
        LOG.info(f"Average backprojection error: {mean_error:.6f}")

        return ret, mtx, dist, rvecs, tvecs

    # ------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------
    def run(self):
        """
        Main calibration execution.
        Finds all PNG images, detects checkerboards, collects points,
        and performs camera calibration.
        """
        images = glob.glob(os.path.join(self.image_dir, "*.png"))
        if not images:
            raise FileNotFoundError(f"No PNG file in folder: {self.image_dir}")

        LOG.info(f"\nImages found: {len(images)} pcs")

        for fname in images:
            LOG.info(f"\nProcessing: {fname}")
            img = cv2.imread(fname)
            if img is None:
                LOG.error(f"Failed to read: {fname}")
                continue

            if self.image_shape is None:
                h, w = img.shape[:2]
                self.image_shape = (w, h)

            # Grayscale + zoom
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)

            # Contours
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_contours = img.copy()
            cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

            # Chessboard search (largest to smallest)
            best_pattern = None
            best_corners = None
            for pattern in self.possible_patterns:
                ret, corners = cv2.findChessboardCorners(gray, pattern, None)
                if ret:
                    best_pattern = pattern
                    best_corners = corners
                    LOG.info(f"Checkerboard found: {pattern[0]}×{pattern[1]}")
                    break

            img_chess = img.copy()
            if best_pattern is not None:
                # Scale back to original size
                corners_scaled = best_corners / self.scale_factor

                # Refine points with subpixels
                gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners_refined = cv2.cornerSubPix(
                    gray_orig,
                    corners_scaled,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )

                # Real coordinates (on Z=0 plane)
                cols, rows = best_pattern
                objp = np.zeros((cols * rows, 3), np.float32)
                objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size

                self.objpoints.append(objp)
                self.imgpoints.append(corners_refined)

                # Visualisation
                for pt in corners_scaled:
                    cv2.circle(img_chess, tuple(np.int32(pt[0])), 6, (0, 255, 0), -1)
                pts = np.int32(corners_scaled).reshape(-1, 2)
                cv2.polylines(img_chess, [pts], isClosed=True, color=(0, 0, 255), thickness=3)

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title("Original image")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
                plt.title("Contours")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(img_chess, cv2.COLOR_BGR2RGB))
                plt.title(f"Chessboard: {best_pattern[0]}×{best_pattern[1]}")
                plt.axis("off")

                plt.tight_layout()
                plt.show()
            else:
                LOG.info("No chessboard found with any of the patterns.")

        # ------------------------------------------------------------
        # Calibration
        # ------------------------------------------------------------
        if len(self.objpoints) >= 3:
            return self.calibrate_camera(self.objpoints, self.imgpoints, self.image_shape)
        else:
            LOG.info("\nNot enough chessboard samples for calibration.")
            return None, None, None, None, None
