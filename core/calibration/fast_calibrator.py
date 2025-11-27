import os
import cv2
import numpy as np

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("FastCalibrator")


class FastCalibrator:
    """
    Handles full camera calibration process from chessboard images.
    Includes corner detection, reprojection error filtering, and YAML export.
    """

    def __init__(self, calibration_source: list, output_dir: str):
        """
        :param calibration_source: list of image file paths for calibration
        """
        self.calibration_source = calibration_source
        self.output_dir = output_dir
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rms_error = None

    def perform_calibration(self, max_chessboard_size=(12, 8)):
        """Runs full camera calibration on the provided images."""
        MAX_W, MAX_H = max_chessboard_size

        if not self.calibration_source:
            LOG.error("No calibration images provided.")
            return None, None, None, None

        objpoints, imgpoints, processed = [], [], []
        objp = np.zeros((MAX_W * MAX_H, 3), np.float32)
        objp[:, :2] = np.mgrid[0:MAX_W, 0:MAX_H].T.reshape(-1, 2)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        LOG.info(f"Searching for chessboard in {len(self.calibration_source)} image...")

        first_img = cv2.imread(self.calibration_source[0])
        if first_img is None:
            LOG.error("The first calibration image cannot be read.")
            return None, None, None, None
        gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        for fname in self.calibration_source:
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)

            ret, corners = cv2.findChessboardCornersSB(gray, (MAX_W, MAX_H))
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objpoints.append(objp)
                imgpoints.append(corners2)
                processed.append(fname)
                LOG.info(f"Vertexes found: {os.path.basename(fname)}")
            else:
                LOG.info(f"No vertices found: {os.path.basename(fname)}")

        if not objpoints:
            LOG.error("We couldn't find enough valid images for calibration.")
            return None, None, None, None

        flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None, flags=flags)

        if not ret:
            LOG.error("Calibration failed.")
            return None, None, None, None

        self.camera_matrix = K
        self.dist_coeffs = D
        self.rms_error = ret

        LOG.info(f"Calibration successful. RMS error: {ret:.4f}")
        LOG.info(f"Camera matrix:\n{K}")
        LOG.info(f"Distortion coefficients:\n{D}")

        self.save_to_yaml(K, D, ret)
        return K, D, rvecs, tvecs

    def save_to_yaml(self, K, D, rms_error):
        """Saves calibration results to YAML file."""
        output_file = self.output_dir
        try:
            fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", K)
            fs.write("dist_coeffs", D)
            fs.write("rms_error", rms_error)
            fs.release()
            LOG.info(f"Calibration saved to YAML file: {output_file}")
        except Exception as e:
            LOG.error(f"Error saving calibration: {e}")
