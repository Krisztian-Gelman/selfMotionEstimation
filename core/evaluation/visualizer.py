import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("Visualizer")


class Visualizer:
    """
    Visualizes measured (.mat) and estimated (.csv) drone trajectories in 3D and 2D.
    """

    def __init__(self, measured_file: str, estimated_file: str, output_dir: str):

        self.measured_file = measured_file
        self.estimated_file = estimated_file
        self.output_dir = output_dir

        LOG.info("Visualiser initialized.")
        LOG.info(f"Measured path file: {self.measured_file}")
        LOG.info(f"Estimated path file: {self.estimated_file}")

        self.measured_data = None
        self.estimated_data = None

    # ----------------------------------------------------------------------
    # >>> Data loading
    # ----------------------------------------------------------------------
    def load_data(self):
        """Loads measured and estimated data from supported formats (.mat, .csv)."""
        # ---- Measured data (MATLAB .mat file) ----
        try:
            if self.measured_file.endswith(".mat"):
                LOG.info("Loading measured data from MATLAB file...")
                mat_data = loadmat(self.measured_file)
                # Check if the key is included
                if "mission_server_data" not in mat_data:
                    LOG.error("Variable 'mission_server_data' not found in MATLAB file.")
                    return

                data = mat_data["mission_server_data"]
                # MATLAB data structure:
                # 1: time, 2–4: desired path (x,y,z), 6–8: measured path (x,y,z)
                meas_x = data[5, :]
                meas_y = data[6, :]
                meas_z = data[7, :]

                self.measured_data = pd.DataFrame({
                    "x": meas_x.flatten(),
                    "y": meas_y.flatten(),
                    "z": meas_z.flatten()
                })
                LOG.info(f"Measured trajectory loaded: {len(self.measured_data)} records from MATLAB file.")
            else:
                # If it is CSV
                self.measured_data = pd.read_csv(self.measured_file, sep=';', skipinitialspace=True)
                LOG.info(f"Measured trajectory loaded: {len(self.measured_data)} records from CSV file.")
        except Exception as e:
            LOG.error(f"Failed to load measured data: {e}")
            self.measured_data = None

        # ---- Estimated data (.csv file) ----
        try:
            self.estimated_data = pd.read_csv(self.estimated_file, sep=';', skipinitialspace=True)
            LOG.info(f"Estimated trajectory loaded: {len(self.estimated_data)} records.")
            LOG.info(f"Estimated columns: {list(self.estimated_data.columns)}")
        except Exception as e:
            LOG.error(f"Failed to load estimated data: {e}")
            self.estimated_data = None

    # ----------------------------------------------------------------------
    # >>> 3D Trajectory Drawing
    # ----------------------------------------------------------------------
    def plot_3d_trajectory(self, show=True, save=False):
        """Plots measured and estimated trajectories in 3D."""

        if self.measured_data is None or self.estimated_data is None:
            LOG.warning("Data not loaded yet. Calling load_data() automatically.")
            self.load_data()

        if self.measured_data is None or self.estimated_data is None:
            LOG.error("Cannot plot trajectory — one or more datasets are missing.")
            return

        LOG.info("Generating 3D trajectory plot...")

        meas_x, meas_y, meas_z = self.measured_data["x"], self.measured_data["y"], self.measured_data["z"]
        est_x, est_y, est_z = self.estimated_data["x"], self.estimated_data["y"], self.estimated_data["z"]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(meas_x, meas_y, meas_z, "r-", label="Measured Path", linewidth=1.5)
        ax.plot(est_x, est_y, est_z, "g-", label="Estimated Path", linewidth=1.5)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Drone 3D Trajectory")
        ax.legend()
        ax.grid(True)
        ax.set_box_aspect([1, 1, 0.6])

        if save:
            output_path = os.path.join(self.output_dir, "plots", "trajectory_3d.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            LOG.info(f"3D trajectory saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    # ----------------------------------------------------------------------
    # >>> 2D Top view
    # ----------------------------------------------------------------------
    def plot_top_view(self, show=True, save=False):
        """Plots top-down XY view of trajectories."""
        if self.measured_data is None or self.estimated_data is None:
            self.load_data()

        if self.measured_data is None or self.estimated_data is None:
            LOG.error("Cannot plot top view — data missing.")
            return

        LOG.info("Generating top-view plot...")

        meas_x, meas_y = self.measured_data["x"], self.measured_data["y"]
        est_x, est_y = self.estimated_data["x"], self.estimated_data["y"]

        plt.figure(figsize=(8, 8))
        plt.plot(meas_x, meas_y, "r-", label="Measured Path", linewidth=1.5)
        plt.plot(est_x, est_y, "g-", label="Estimated Path", linewidth=1.5)

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Drone Trajectory (Top View)")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        if save:
            output_path = os.path.join(self.output_dir, "plots", "trajectory_top_view.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            LOG.info(f"Top view saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()
