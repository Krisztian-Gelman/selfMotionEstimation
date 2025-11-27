import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("SequenceController")


class SequenceController:
    """
    The class that controls the processing of image sequences

    Its task is:
    - processing a sequence of images
    - coordinating calls to FrameProcessor, FeatureTracker, MotionEstimator and DisplayManager
    - saving results
    """

    def __init__(self, frame_processor, tracker, motion_estimator, display_manager, result_exporter, device, output_dir: str):
        self.frame_processor = frame_processor
        self.tracker = tracker
        self.motion_estimator = motion_estimator
        self.display_manager = display_manager
        self.result_exporter = result_exporter
        self.device = device
        self.output_dir = output_dir

    # -------------------------------------------------------------------------
    def process(self, file_paths, flow_method, camera_matrix, dist_coeffs):
        """
        Process the entire image sequence.
        The operation is 1:1 identical to the original VideoHandler.process_image_sequence() method.
        """

        if not file_paths:
            LOG.error("Error: No images loaded in file_paths.")
            return [], []

        LOG.info(f"Image burst mode — processing {len(file_paths)} images with {flow_method.value}.")

        trajectories = []
        completed_trajectories = []
        camera_trajectory = [np.array([0, 0, 0])]

        trajectory_display_mode = "ALL"
        active_trajectory_index = 0

        for frame_idx, img_path in enumerate(file_paths):
            try:
                # --- Frame processing ---
                current_gray, white_mask, latest_corners, output_frame, quad_count, corner_count = \
                    self.frame_processor.preprocess(img_path)

                # --- Optical Flow Tracking ---
                self.tracker.track(current_gray, white_mask, latest_corners)
                trajectories = self.tracker.trajectories
                completed_trajectories = self.tracker.completed_trajectories

                # --- Camera motion estimation ---
                self.motion_estimator.K = camera_matrix
                self.motion_estimator.estimate(trajectories)
                camera_trajectory = self.motion_estimator.trajectory

                # --- Drawing trajectories ---
                for traj in trajectories:
                    if len(traj) > 1:
                        points = np.int32(
                            [pt.reshape(-1)[:2] for pt in traj if len(pt.reshape(-1)) >= 2]
                        )
                        for i in range(len(points) - 1):
                            cv2.line(output_frame, tuple(points[i]), tuple(points[i + 1]), (255, 255, 255), 2)

                # --- Information overlay ---
                cv2.putText(output_frame,
                            f"Quads: {quad_count} | Corners: {corner_count} | Active: {len(trajectories)} | Mode: {trajectory_display_mode}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # --- Display ---
                self.display_manager.display_frame(output_frame, frame_idx, self.device)

                # --- Sparse matplotlib update ---
                if frame_idx % 10 == 0:
                    self.display_manager.plot_trajectories(trajectories, completed_trajectories)

            except Exception as e:
                LOG.error(f"Error in image {frame_idx} ({os.path.basename(img_path)}): {e}")
                continue

            # --- Keystroke monitoring ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            all_traj_count = len(trajectories) + len(completed_trajectories)
            if key == ord('p'):
                trajectory_display_mode = "FILTERED"
                active_trajectory_index = (active_trajectory_index + 1) % max(1, all_traj_count)
                self.display_manager.plot_trajectories(trajectories, completed_trajectories)
            elif key == ord('a'):
                trajectory_display_mode = "ALL"
                self.display_manager.plot_trajectories(trajectories, completed_trajectories)
            elif key == ord('f'):
                trajectory_display_mode = "FILTERED"
                self.display_manager.plot_trajectories(trajectories, completed_trajectories)

        # --- End of processing ---
        cv2.destroyAllWindows()
        plt.ioff()

        self.result_exporter.export_camera_path(
            camera_trajectory,
            self.output_dir
        )

        self.display_manager.plot_trajectories(trajectories, completed_trajectories)

        return trajectories, completed_trajectories
