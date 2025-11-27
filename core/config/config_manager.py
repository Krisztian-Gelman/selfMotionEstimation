import os

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("ConfigManager")

class ConfigManager:
    """
    Centralized configuration and path manager for the Self-Motion Estimation project.
    This version does not load external files — instead, it maintains unified absolute paths
    for all key directories and data resources used across modules.
    """

    def __init__(self):
        # --- Define project root ---
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        LOG.info(f"ConfigManager initialized. Root directory: {self.root_dir}")

        # --- Basic paths ---
        self.data_dir = os.path.join(self.root_dir, "data")
        self.config_dir = os.path.join(self.data_dir, "config")
        self.results_dir = os.path.join(self.data_dir, "results")
        self.log_dir = os.path.join(self.data_dir, "log", "output")
        self.sensor_dir = os.path.join(self.data_dir, "sensor")
        self.sensor_data_dir = os.path.join(self.sensor_dir, "sensor_data")
        self.sequence_dir = os.path.join(self.data_dir, "sequences")
        self.calibration_dir = os.path.join(self.data_dir, "calibration")

        # --- Main configuration files ---
        self.camera_calibration_file = os.path.join(self.config_dir, "camera_calibration_data.yaml")
        self.environment_file = os.path.join(self.root_dir, "env", "environment.yaml")
        self.estimated_positions_file = os.path.join(self.results_dir, "positions", "estimated_camera_pos.csv")

        # --- Main configuration directories ---
        self.calibration_video_source_down = os.path.join(self.calibration_dir,"orig_videos","down")
        self.calibration_video_source_front = os.path.join(self.calibration_dir,"orig_videos","front")
        self.calibration_images_source_down = os.path.join(self.calibration_dir,"png_dump","down")
        self.calibration_images_source_front = os.path.join(self.calibration_dir,"png_dump","front")

        self.input_video_source_down = os.path.join(self.sequence_dir,"orig_videos","down")
        self.input_video_source_front = os.path.join(self.sequence_dir,"orig_videos","front")
        self.input_images_source_down = os.path.join(self.sequence_dir,"png_dump","down")
        self.input_images_source_front = os.path.join(self.sequence_dir,"png_dump","front")

        # --- Run settings ---
        self.default_flow_method = "Lucas-Kanade"
        self.default_device = "cuda"

        # Check if folders exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create key folders if they are missing."""
        for directory in [self.log_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
            LOG.info(f"Ensured directory exists: {directory}")

    def summary(self):
        """Provides a summary of the main configuration paths."""
        summary_data = {
            "Root": self.root_dir,
            "Data": self.data_dir,
            "Config": self.config_dir,
            "Results": self.results_dir,
            "Logs": self.log_dir,
            "Camera Calibration": self.camera_calibration_file,
            "Environment": self.environment_file
        }
        LOG.info("Configuration Summary:")
        for key, val in summary_data.items():
            LOG.info(f"  {key}: {val}")
        return summary_data
