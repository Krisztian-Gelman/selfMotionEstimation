import numpy as np
import cv2

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("MotionEstimator")

class MotionEstimator:
    """
    Handles motion estimation between frames using essential matrix recovery.
    """

    def __init__(self, K):
        self.K = K
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory = [np.zeros(3)]

    def estimate(self, trajectories):
        """
        Updates camera position using essential matrix estimation between frame pairs.
        """
        if len(trajectories) == 0:
            LOG.warning("No trajectory available for motion estimation.")
            return None

        p0, p1 = [], []
        for traj in trajectories:
            if len(traj) >= 2:
                p0.append(traj[-2].reshape(-1))
                p1.append(traj[-1].reshape(-1))

        if len(p0) < 8:
            LOG.warning("Too few points to calculate essential matrix.")
            return None

        p0 = np.array(p0).reshape(-1, 1, 2)
        p1 = np.array(p1).reshape(-1, 1, 2)

        E, mask = cv2.findEssentialMat(p0, p1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            LOG.warning("Could not find essential matrix.")
            return None

        _, R, t, mask_pose = cv2.recoverPose(E, p0, p1, self.K)

        # Accumulate the movement
        self.cur_t += self.cur_R @ t
        self.cur_R = R @ self.cur_R

        pos = self.cur_t.flatten()
        self.trajectory.append(pos)

        #LOG.info(f"[POS] Frame {len(self.trajectory)-1}: {self.trajectory[-1]}")
        return self.trajectory[-1]
