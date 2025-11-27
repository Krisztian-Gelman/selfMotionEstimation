import cv2
import numpy as np
import matplotlib.pyplot as plt

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("DisplayManager")


class DisplayManager:
    """
    The class responsible for displaying and plotting trajectories.
    The operation is 1:1 identical to the original code, only organized modularly.
    """

    def __init__(self):
        # Matplotlib configuration
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.invert_yaxis()
        DISPLAY_WIDTH = 1920
        DISPLAY_HEIGHT = 1080
        self.ax.set_xlim(0, DISPLAY_WIDTH)
        self.ax.set_ylim(DISPLAY_HEIGHT, 0)
        self.ax.set_title('Optical Flow Trajectories')
        self.ax.set_xlabel('X Coordinate (pixel)')
        self.ax.set_ylabel('Y Coordinate (pixel)')
        self.ax.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------------------
    def display_frame(self, frame_to_display, frame_idx, device, mode="NORMAL"):
        """
        Display a single frame in an OpenCV window.
        """
        #1. Tag Definition
        if mode == "NORMAL":
            label = f"Frame {frame_idx} (PyTorch {device})"
        elif mode == "CONSECUTIVE":
            label = f"Frame {frame_idx} -> Frame {frame_idx + 1} (CONSECUTIVE | {device})"
        else:
            label = f"Frame {frame_idx}"

        # 2. Text overlay
        cv2.putText(frame_to_display, label, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        DISPLAY_WIDTH = 640
        DISPLAY_HEIGHT = 400
        img_resized = cv2.resize(frame_to_display, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                                 interpolation=cv2.INTER_LINEAR)

        #3. Display
        cv2.imshow('Video Stream (Single Camera)', img_resized)

    # -------------------------------------------------------------------------
    def plot_trajectories(self, trajectories, completed_trajectories, max_points_to_plot=None):
        """
        Plots all active and completed trajectories using matplotlib.
        """
        if not hasattr(self, 'fig') or self.fig is None:
            LOG.info("Matplotlib Figure not yet initialized. Skipping plotting.")
            return

        all_trajectories = trajectories + completed_trajectories
        if not all_trajectories:
            LOG.info("No trajectories to plot.")
            return

        #1. Collect coordinates
        all_x, all_y = [], []
        for traj in all_trajectories:
            for point_array in traj:
                try:
                    point = point_array.reshape(-1)[:2]
                    if len(point) == 2:
                        all_x.append(point[0])
                        all_y.append(point[1])
                except:
                    continue

        if not all_x:
            LOG.info("No meaningful points to plot.")
            return

        #2. Axis boundaries and padding
        min_x, max_x = np.min(all_x), np.max(all_x)
        min_y, max_y = np.min(all_y), np.max(all_y)
        x_range = max_x - min_x or 10
        y_range = max_y - min_y or 10
        padding = 0.1

        new_xlim = (min_x - x_range * padding, max_x + x_range * padding)
        new_ylim = (min_y - y_range * padding, max_y + y_range * padding)

        LOG.info(f"Plotting: {len(all_trajectories)} trajectories (Active + Completed).")

        self.ax.clear()
        self.ax.invert_yaxis()
        self.ax.set_xlim(new_xlim[0], new_xlim[1])
        self.ax.set_ylim(new_ylim[1], new_ylim[0])
        self.ax.set_title('Optical Flow Trajectories')
        self.ax.set_xlabel('X Coordinate (pixel)')
        self.ax.set_ylabel('Y Coordinate (pixel)')
        self.ax.grid(True, linestyle=':', alpha=0.6)

        colors = plt.cm.get_cmap('hsv', len(all_trajectories))

        # 3. Drawing trajectories
        for i, traj in enumerate(all_trajectories):
            if len(traj) < 2:
                continue

            points = []
            for point_array in traj:
                try:
                    point = point_array.reshape(-1)[:2]
                    if len(point) == 2:
                        points.append(point)
                except:
                    continue

            if not points:
                continue

            points = np.array(points, dtype=np.float32)
            x_coords, y_coords = points[:, 0], points[:, 1]

            if max_points_to_plot and len(x_coords) > max_points_to_plot:
                step = max(1, len(x_coords) // max_points_to_plot)
                x_coords, y_coords = x_coords[::step], y_coords[::step]

            self.ax.plot(x_coords, y_coords,
                         color=colors(i),
                         linewidth=1,
                         marker='.',
                         markersize=4,
                         alpha=0.8)

            # start and end point
            self.ax.plot(x_coords[0], y_coords[0], 'o', color=colors(i), markersize=8, markeredgecolor='black')
            self.ax.plot(x_coords[-1], y_coords[-1], 'x', color=colors(i), markersize=10, markeredgecolor='black')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
