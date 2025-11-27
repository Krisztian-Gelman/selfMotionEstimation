import csv
import numpy as np

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("ResultExporter")


class ResultExporter:
    """
    Responsible for saving processed motion data (e.g. estimated camera trajectory).
    """

    def export_camera_path(self, trajectory, output_path: str):
        """
        Saves the estimated camera trajectory (x, y, z) into a CSV file.
        Performs scaling to normalize coordinates to [-1, +1].
        """
        try:
            trajectory_array = np.array(trajectory)
            if trajectory_array.shape[0] > 1:
                min_vals = np.min(trajectory_array, axis=0)
                max_vals = np.max(trajectory_array, axis=0)
                ranges = max_vals - min_vals
                ranges[ranges == 0] = 1

                scaled_trajectory = 2 * (trajectory_array - min_vals) / ranges - 1
                trajectory = scaled_trajectory.tolist()

                LOG.info("[SCALE] Camera track scaled to the range -1..+1 on all axes.")
            else:
                LOG.warning("[SCALE] Too few points to scale.")
        except Exception as e:
            LOG.error(f"[SCALE] Error scaling camera trajectory: {e}")

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['frame', 'x', 'y', 'z'])
                for idx, pos in enumerate(trajectory):
                    pos_list = [float(p) for p in np.ravel(pos)]
                    writer.writerow([idx] + pos_list)
                    LOG.info(f"[POS] {pos_list}")

            LOG.info(f"Camera track saved to CSV: {output_path}")
        except Exception as e:
            LOG.error(f"[EXPORT] Error saving camera trajectory: {e}")
