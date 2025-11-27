import numpy as np
import cv2

from selfmotionestimation.data.log.logger import Logger
from selfmotionestimation.core.processing.flow.optical_flow_methods import BaseOpticalFlow

LOG = Logger("FeatureTracker")

class FeatureTracker:
    """
    Optical Flow implementations
    """

    def __init__(self, flow_method: BaseOpticalFlow):
        self.flow_method = flow_method
        self.old_gray = None
        self.p0 = None
        self.trajectories = []
        self.completed_trajectories = []

    # --------------------------------------------------------------------------
    # Lucas–Kanade Optical Flow
    # --------------------------------------------------------------------------
    def lucas_kanade_flow(self, current_gray, white_mask, latest_corners):
        MAX_CORNERS =  80

        # LK parameters (slightly stricter convergence)
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
        )

        # Quality thresholds
        FB_ERR_THRESH = getattr(self, 'fb_err_thresh', 1.2)  # px
        RANSAC_REPROJ = getattr(self, 'ransac_thresh', 2.0)  # px

        # 1) Initialization on the first frame
        if self.old_gray is None or (self.p0 is None or len(self.p0) == 0):
            feature_params = dict(
                maxCorners=MAX_CORNERS,
                qualityLevel=0.02,
                minDistance=10,
                blockSize=7
            )
            new_p0 = cv2.goodFeaturesToTrack(current_gray, mask=white_mask, **feature_params)

            if new_p0 is not None:
                if hasattr(self, 'trajectories') and self.trajectories:
                    self.completed_trajectories.extend(self.trajectories)
                self.p0 = new_p0
                self.trajectories = [[pt] for pt in self.p0]
                self.old_gray = current_gray.copy()
            else:
                # no good point – wait until the next frame
                self.old_gray = current_gray.copy()
                return

        #2) Follow if there is a previous frame and points
        if self.old_gray is not None and self.p0 is not None and len(self.p0) > 0:
            # Forward LK
            p1, st_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(self.old_gray, current_gray, self.p0, None, **lk_params)
            if p1 is None or st_fwd is None:
                self.old_gray = current_gray.copy()
                return

            st_fwd = st_fwd.reshape(-1).astype(bool)
            # Backward LK from forward estimates
            p0_bwd, st_bwd, err_bwd = cv2.calcOpticalFlowPyrLK(current_gray, self.old_gray, p1, None, **lk_params)
            st_bwd = (st_bwd.reshape(-1).astype(bool) if st_bwd is not None else np.zeros_like(st_fwd, dtype=bool))

            # Forward–Backward error
            fb_err = np.full(len(self.p0), np.inf, dtype=np.float32)
            ok_fb_idx = np.where(st_fwd & st_bwd)[0]
            if ok_fb_idx.size > 0:
                diff = (self.p0[ok_fb_idx] - p0_bwd[ok_fb_idx]).reshape(-1, 2)
                fb_err[ok_fb_idx] = np.linalg.norm(diff, axis=1)

            fb_mask = fb_err < FB_ERR_THRESH

            # Pre-filtered indexes (LK ok + FB ok)
            pre_idx = np.where(st_fwd & fb_mask)[0]

            # RANSAC inlier filtering (homography)
            if pre_idx.size >= 4:
                src = self.p0[pre_idx].reshape(-1, 1, 2)
                dst = p1[pre_idx].reshape(-1, 1, 2)
                H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ)
                if inl is not None:
                    inl = inl.ravel().astype(bool)
                    good_idx = pre_idx[inl]
                else:
                    good_idx = pre_idx
            else:
                good_idx = pre_idx

            # Trajectories refresh
            new_active_trajectories = []
            good_idx_set = set(good_idx.tolist())

            for i, traj in enumerate(self.trajectories):
                if i in good_idx_set:
                    new_pt = p1[i].reshape(1, 2)
                    traj.append(new_pt)
                    new_active_trajectories.append(traj)
                else:
                    self.completed_trajectories.append(traj)

            # New actives and the current point set
            if len(good_idx) > 0:
                self.trajectories = new_active_trajectories
                self.p0 = p1[good_idx].reshape(-1, 1, 2)
            else:
                self.trajectories = []
                self.p0 = None

        # 3) Redetection: fill the points in a diverse way
        current_p0_len = 0 if self.p0 is None else len(self.p0)
        num_to_add = MAX_CORNERS - current_p0_len
        if num_to_add > 0:
            # Mask copy
            det_mask = white_mask.copy()
            if self.p0 is not None and len(self.p0) > 0:
                for (x, y) in self.p0.reshape(-1, 2).astype(int):
                    cv2.circle(det_mask, (x, y), 12, 0, -1)  # 12 px radius ban

            feature_params_add = dict(
                maxCorners=num_to_add,
                qualityLevel=0.02,
                minDistance=10,
                blockSize=7
            )
            new_pts = cv2.goodFeaturesToTrack(current_gray, mask=det_mask, **feature_params_add)

            if new_pts is not None:
                if current_p0_len > 0:
                    self.p0 = np.vstack((self.p0, new_pts))
                else:
                    self.p0 = new_pts
                # Start new trajectories with the new points
                for pt in new_pts:
                    self.trajectories.append([pt])

        # 4) Update status for next iteration
        self.old_gray = current_gray.copy() if (self.p0 is not None and len(self.p0) > 0) else current_gray.copy()

    # --------------------------------------------------------------------------
    # Farneback Optical Flow
    # --------------------------------------------------------------------------
    def farneback_flow(self, current_gray, white_mask, latest_corners):
        MAX_CORNERS = 50

        if self.old_gray is None or self.p0 is None or len(self.p0) == 0:
            self.old_gray = current_gray.copy()
            feature_params = dict(maxCorners=MAX_CORNERS, qualityLevel=0.3, minDistance=15, blockSize=7)
            new_p0 = cv2.goodFeaturesToTrack(current_gray, mask=white_mask, **feature_params)
            if new_p0 is not None:
                if hasattr(self, 'trajectories') and self.trajectories:
                    self.completed_trajectories.extend(self.trajectories)
                self.p0 = new_p0
                self.trajectories = [[pt] for pt in self.p0]
            else:
                self.p0 = None
                self.trajectories = []
            return

        farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        flow_fwd = cv2.calcOpticalFlowFarneback(self.old_gray, current_gray, None, **farneback_params)
        flow_bwd = cv2.calcOpticalFlowFarneback(current_gray, self.old_gray, None, **farneback_params)

        pts = self.p0.reshape(-1, 2).astype(np.float32)
        H, W = current_gray.shape
        valid_mask = (pts[:, 0] >= 0) & (pts[:, 0] < W - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < H - 1)
        pts = pts[valid_mask]
        if len(pts) == 0:
            self.p0 = None
            self.trajectories = []
            self.old_gray = current_gray.copy()
            return

        map_x = pts[:, 0].reshape(-1, 1)
        map_y = pts[:, 1].reshape(-1, 1)
        fx = cv2.remap(flow_fwd[..., 0], map_x, map_y, cv2.INTER_LINEAR)
        fy = cv2.remap(flow_fwd[..., 1], map_x, map_y, cv2.INTER_LINEAR)
        flow_at_points = np.hstack([fx, fy])
        p1 = pts + flow_at_points

        map_x2 = p1[:, 0].reshape(-1, 1)
        map_y2 = p1[:, 1].reshape(-1, 1)
        bfx = cv2.remap(flow_bwd[..., 0], map_x2, map_y2, cv2.INTER_LINEAR)
        bfy = cv2.remap(flow_bwd[..., 1], map_x2, map_y2, cv2.INTER_LINEAR)
        back_flow = np.hstack([bfx, bfy])
        back_diff = np.linalg.norm(back_flow + flow_at_points, axis=1)
        back_threshold = np.mean(back_diff) + 3 * np.std(back_diff)
        consistent_mask = back_diff < back_threshold

        p1 = p1[consistent_mask]
        flow_at_points = flow_at_points[consistent_mask]
        valid_trajs = [self.trajectories[i] for i, m in enumerate(valid_mask) if m]
        valid_trajs = [valid_trajs[i] for i, c in enumerate(consistent_mask) if c]

        if len(p1) == 0:
            self.p0 = None
            self.trajectories = []
            self.old_gray = current_gray.copy()
            return

        motion_mag = np.linalg.norm(flow_at_points, axis=1)
        motion_threshold = np.mean(motion_mag) + 3 * np.std(motion_mag)
        stable_mask = motion_mag < motion_threshold
        p1 = p1[stable_mask]
        valid_trajs = [valid_trajs[i] for i, m in enumerate(stable_mask) if m]

        if len(p1) == 0:
            self.p0 = None
            self.trajectories = []
            self.old_gray = current_gray.copy()
            return

        new_trajectories = []
        for traj, new_pt in zip(valid_trajs, p1):
            traj.append(new_pt.reshape(1, 2))
            new_trajectories.append(traj)

        self.trajectories = new_trajectories
        self.p0 = p1.reshape(-1, 1, 2).astype(np.float32)

        current_p0_len = len(self.p0)
        if current_p0_len < 10:
            feature_params_add = dict(maxCorners=MAX_CORNERS - current_p0_len,
                                      qualityLevel=0.3,
                                      minDistance=15,
                                      blockSize=7)
            mask = white_mask.copy()
            for x, y in self.p0.reshape(-1, 2).astype(int):
                cv2.circle(mask, (x, y), 20, 0, -1)
            new_pts = cv2.goodFeaturesToTrack(current_gray, mask=mask, **feature_params_add)
            if new_pts is not None:
                self.p0 = np.vstack((self.p0, new_pts))
                for pt in new_pts:
                    self.trajectories.append([pt])

        self.old_gray = current_gray.copy()

    # Future development possible
    #def tv_l1_flow(self, current_gray, white_mask, latest_corners):
    #def deep_flow(self, current_gray, white_mask, latest_corners):

    # --------------------------------------------------------------------------
    # Flow method calling
    # --------------------------------------------------------------------------
    def track(self, current_gray, white_mask, latest_corners):
        if self.flow_method == BaseOpticalFlow.LUCAS_KANADE:
            self.lucas_kanade_flow(current_gray, white_mask, latest_corners)
        elif self.flow_method == BaseOpticalFlow.FARNEBACK:
            self.farneback_flow(current_gray, white_mask, latest_corners)
        else:
            LOG.warning(f"Not supported Optical Flow type: {self.flow_method}")

        return self.trajectories
