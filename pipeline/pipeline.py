import os
import cv2
import torch

from selfmotionestimation.core.calibration.fast_calibrator import FastCalibrator
from selfmotionestimation.core.evaluation.comparator import Comparator
from selfmotionestimation.core.evaluation.visualizer import Visualizer
from selfmotionestimation.core.processing.flow.optical_flow_methods import BaseOpticalFlow
from selfmotionestimation.core.processing.flow.feature_tracker import FeatureTracker
from selfmotionestimation.core.processing.frame.frame_processor import FrameProcessor
from selfmotionestimation.core.processing.sequence.sequence_controller import SequenceController
from selfmotionestimation.core.processing.visual.display_manager import DisplayManager
from selfmotionestimation.core.processing.input.input_manager import InputManager
from selfmotionestimation.core.processing.output.result_exporter import ResultExporter
from selfmotionestimation.core.processing.motion.motion_estimator import MotionEstimator
from selfmotionestimation.core.config.config_manager import ConfigManager
from selfmotionestimation.core.calibration.camera_calibration import CameraCalibrator
from selfmotionestimation.utils.png_dump import PNGDump
from selfmotionestimation.data.log.logger import Logger

"""
Project title: Self-motion and orientation estimation for aircraft
Version: 14.0
Created at: 30.10.2025
Last modified at: 27.10.2025
Author: Krisztián Gelman (alias John Rambo)

Project description:
Project descriptionThe goal of the project is to augment the capabilities of the widespread
GPS/IMU-based self-motion estimation algorithms with information available from onboard camera
images. The first part is to do a comprehensive literature review on the state-of-the-art and
then develop new algorithms for the task. Also, it is important to run tests to determine the
statistical properties of these algorithms. The test is run either on videos rendered by the
simulator or also on flight videos. Preliminary algorithms, scientific papers, the simulator,
and flight videos are available for the job. Unmanned Aerial Vehicles, UAV, sensor fusion,
collision avoidance, video and image processing.

Pipeline Module
--------------------
Pipeline Module – The brain of the program, where all data passes through.
This module is responsible for the entire processing chain, from calibration to image sequence processing.
It manages multiple modules and performs visual motion estimation through the processing components.
"""

LOG = Logger("PipeLine")
cfg = ConfigManager()

"""
CONFIGURATION
"""

CALIBRATION_VIDEO_FILE_DOWN = "down_16-Oct-2025 04-29-09.mp4"
CALIBRATION_IMAGES_DIR_DOWN = "16Oct2025"
INPUT_VIDEO_FILE_DOWN = "down_19-Aug-2025 05-52-16.mp4"
INPUT_IMAGES_DIR_DOWN = "19Aug2025"
DRONE_SENSOR_DATA = "Mission_Server_19-Aug-2025_05-51-41.mat"

class Pipeline:
    """
    Main process control class that coordinates the system operation:
    - Loading or performing camera calibration
    - Image sequence processing
    - Optical flow based motion estimation
    """

    def __init__(self, calibration: bool = True, png_dump: bool = True,
                 processing = True, visualisation: bool = True, comparing: bool = True):
        """
        Initializes the Pipeline, loads the calibration data, and starts the processing
        """

        # --- BASIC PARAMETERS ---
        self.method_type = BaseOpticalFlow.LUCAS_KANADE
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rms_error = None

        self.calibration_video_dir_down = os.path.join(cfg.calibration_video_source_down,CALIBRATION_VIDEO_FILE_DOWN)
        self.calibration_images_dir_down = os.path.join(cfg.calibration_images_source_down,CALIBRATION_IMAGES_DIR_DOWN)
        self.input_video_dir_down = os.path.join(cfg.input_video_source_down, INPUT_VIDEO_FILE_DOWN)
        self.input_images_dir_down = os.path.join(cfg.input_images_source_down, INPUT_IMAGES_DIR_DOWN)
        self.drone_sensor_data = os.path.join(cfg.sensor_data_dir,DRONE_SENSOR_DATA)

        self.file_paths = []

        # --- BASIC MODULES ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOG.info(f"PyTorch device set: {self.device}")

        self.calibration_dumper = PNGDump(
            video_path=self.calibration_video_dir_down,
            output_folder=self.calibration_images_dir_down,
            compression_level=0
        )

        self.calibrator = CameraCalibrator(image_dir=self.calibration_images_dir_down,
                                           scale_factor=1.5)

        self.sequence_dumper = PNGDump(
            video_path=self.input_video_dir_down,
            output_folder=self.input_images_dir_down,
            compression_level=0
        )

        """----------PROCESSING----------"""

        """ --- STEP 1: Calibrate camera --- """
        if calibration:
            self.calibration_dumper.extract_frames_lossless()
            self.calibrator.run()

        """ --- STEP 2: Dump input images --- """
        if png_dump:
            self.sequence_dumper.extract_frames_lossless()

        self.input_manager = InputManager()
        self.result_exporter = ResultExporter()
        self.motion_estimator = MotionEstimator(self.camera_matrix)
        self.tracker = FeatureTracker(self.method_type)
        self.frame_processor = FrameProcessor(self.device)
        self.display_manager = DisplayManager()

        # --- Sequence processing init ---
        self.sequence_controller = SequenceController(
            frame_processor=self.frame_processor,
            tracker=self.tracker,
            motion_estimator=self.motion_estimator,
            display_manager=self.display_manager,
            result_exporter=self.result_exporter,
            device=self.device,
            output_dir=cfg.estimated_positions_file
        )

        LOG.info("Pipeline initialized.")
        LOG.info(f"Optical Flow type: {self.method_type.value}")

        """--- STEP 3: Load or recalibrate camera calibration ---"""
        if not self.loadCalibration():
            LOG.warning("Calibration file not found or corrupt. Recalibration required.")
            self.calibration_dumper.extract_frames_lossless()
            self.calibrateCamera()
            self.loadCalibration()

        """ --- STEP 4: Loading and processing images ---"""
        if processing:
            self.file_paths = self.input_manager.load_input_source(self.input_images_dir_down)
            self.sequence_controller.process(
                self.file_paths,
                self.method_type,
                self.camera_matrix,
                self.dist_coeffs
            )

        """ --- STEP 5: After the processing is complete, we draw the estimated path ---"""
        if visualisation:
            viz = Visualizer(measured_file=self.drone_sensor_data,estimated_file=cfg.estimated_positions_file,output_dir=cfg.results_dir)
            viz.load_data()
            viz.plot_3d_trajectory(show=True, save=True)
            viz.plot_top_view(show=True, save=True)

        """ --- STEP 6: Compare Results ---"""
        if comparing:
            comp = Comparator(measured_file=self.drone_sensor_data,estimated_file=cfg.estimated_positions_file,output_dir=cfg.results_dir)
            comp.load_data()
            comp.compute_metrics()
            comp.report(save=True)
            comp.plot_analysis(show=True, save=True)

    # --------------------------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------------------------

    def calibrateCamera(self):
        """
        Runs the camera calibration process if the configuration file does not exist.
        """
        LOG.info("Starting calibration process...")
        self.file_paths = self.input_manager.load_input_source(
            self.calibration_images_dir_down
        )

        calibrator = FastCalibrator(self.file_paths, cfg.camera_calibration_file)
        self.camera_matrix, self.dist_coeffs, _, _ = calibrator.perform_calibration(max_chessboard_size=(12, 8))
        LOG.info("Calibration completed.")

    def loadCalibration(self, filename=cfg.camera_calibration_file):
        """
        Loads the camera calibration matrix (K), distortion coefficients (D)
        and RMS error from the specified YAML file and stores them in the self variables.
        """
        if not os.path.exists(filename):
            LOG.info(f"Error: Calibration file not found: {filename}")
            return False

        try:
            fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
            camera_matrix_node = fs.getNode("camera_matrix")
            dist_coeffs_node = fs.getNode("dist_coeffs")
            rms_error_node = fs.getNode("rms_error")

            if camera_matrix_node.empty() or dist_coeffs_node.empty() or rms_error_node.empty():
                LOG.info(f"Error: Required data (camera_matrix, dist_coeffs, rms_error) is missing in the file {filename}.")
                fs.release()
                return False

            self.camera_matrix = camera_matrix_node.mat()
            self.dist_coeffs = dist_coeffs_node.mat()
            self.rms_error = rms_error_node.real()
            fs.release()

            LOG.info(f"OK: Calibration data successfully loaded from file '{filename}'.")
            LOG.info(f"Kamera matrix: \n{self.camera_matrix}")
            LOG.info(f"Dist_coeffs: {self.dist_coeffs}")
            LOG.info(f"RMS error: {self.rms_error:.4f}")

            return True

        except Exception as e:
            LOG.info(f"Error: Error loading calibration data: {e}")
            return False
