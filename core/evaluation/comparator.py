import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("Comparator")


class Comparator:
    """
    Compares measured (.mat) and estimated (.csv) drone trajectories numerically.
    Calculates RMS error, drift, and relative accuracy metrics.
    """

    def __init__(self, measured_file: str, estimated_file: str, output_dir: str):
        self.measured_file = measured_file
        self.estimated_file = estimated_file
        self.output_dir =output_dir

        LOG.info("Comparator initialized.")
        LOG.info(f"Measured path file: {self.measured_file}")
        LOG.info(f"Estimated path file: {self.estimated_file}")

        self.measured_data = None
        self.estimated_data = None
        self.metrics = {}

    # ----------------------------------------------------------------------
    # >>> Data loading
    # ----------------------------------------------------------------------
    def load_data(self):
        """Loads measured and estimated trajectory data."""
        try:
            if self.measured_file.endswith(".mat"):
                LOG.info("Loading measured data from MATLAB file...")
                mat_data = loadmat(self.measured_file)
                if "mission_server_data" not in mat_data:
                    LOG.error("Variable 'mission_server_data' not found in MATLAB file.")
                    return

                data = mat_data["mission_server_data"]
                meas_x, meas_y, meas_z = data[5, :], data[6, :], data[7, :]
                self.measured_data = pd.DataFrame({
                    "x": meas_x.flatten(),
                    "y": meas_y.flatten(),
                    "z": meas_z.flatten()
                })
                LOG.info(f"Measured trajectory loaded: {len(self.measured_data)} records.")
            else:
                self.measured_data = pd.read_csv(self.measured_file, sep=';', skipinitialspace=True)
                LOG.info(f"Measured trajectory loaded: {len(self.measured_data)} records from CSV.")
        except Exception as e:
            LOG.error(f"Failed to load measured data: {e}")
            self.measured_data = None

        try:
            self.estimated_data = pd.read_csv(self.estimated_file, sep=';', skipinitialspace=True)
            LOG.info(f"Estimated trajectory loaded: {len(self.estimated_data)} records from CSV.")
        except Exception as e:
            LOG.error(f"Failed to load estimated data: {e}")
            self.estimated_data = None

    # ----------------------------------------------------------------------
    # >>> Data matching
    # ----------------------------------------------------------------------
    def align_data(self):
        """Aligns the two trajectories to have equal length by interpolation or truncation."""
        if self.measured_data is None or self.estimated_data is None:
            LOG.warning("Data not loaded, calling load_data()...")
            self.load_data()

        if self.measured_data is None or self.estimated_data is None:
            LOG.error("Cannot align data – one or both datasets missing.")
            return

        n_meas = len(self.measured_data)
        n_est = len(self.estimated_data)

        if n_meas != n_est:
            LOG.warning(f"Different dataset lengths: measured={n_meas}, estimated={n_est}. Interpolating...")

            # Interpolation to the longer one
            min_len = min(n_meas, n_est)
            self.measured_data = self.measured_data.iloc[:min_len].reset_index(drop=True)
            self.estimated_data = self.estimated_data.iloc[:min_len].reset_index(drop=True)
        else:
            LOG.info("Datasets are aligned in length.")

    # ----------------------------------------------------------------------
    # >>> Calculating error metrics
    # ----------------------------------------------------------------------
    def compute_metrics(self):
        """Computes RMS error, MAE, drift, and relative error between trajectories."""
        if self.measured_data is None or self.estimated_data is None:
            self.load_data()
        self.align_data()

        meas = self.measured_data[["x", "y", "z"]].to_numpy()
        est = self.estimated_data[["x", "y", "z"]].to_numpy()

        # --- Differences ---
        diff = est - meas
        euclidean_errors = np.linalg.norm(diff, axis=1)

        # --- Basic error metrics ---
        rms_error = np.sqrt(mean_squared_error(meas, est))
        mae = mean_absolute_error(meas, est)
        max_dev = np.max(euclidean_errors)
        drift = np.linalg.norm(est[-1] - meas[-1])

        # --- Determining path lengths ---
        meas_path_length = np.sum(np.linalg.norm(np.diff(meas, axis=0), axis=1))
        est_path_length = np.sum(np.linalg.norm(np.diff(est, axis=0), axis=1))
        avg_path_length = (meas_path_length + est_path_length) / 2.0

        # --- Relative error (normalized) ---
        # The RMS error is expressed as a percentage of the average path length
        if avg_path_length > 0:
            rel_error = (rms_error / avg_path_length) * 100
        else:
            rel_error = np.nan

        # --- Storing results ---
        self.metrics = {
            "RMS_Error_m": rms_error,
            "MAE_m": mae,
            "Max_Deviation_m": max_dev,
            "Final_Drift_m": drift,
            "Measured_Path_Length_m": meas_path_length,
            "Estimated_Path_Length_m": est_path_length,
            "Relative_Error_percent": rel_error,
        }

        LOG.info("Trajectory comparison metrics calculated successfully.")
        return self.metrics

    # ----------------------------------------------------------------------
    # >>> Results report
    # ----------------------------------------------------------------------
    def report(self, save=True):
        """Prints and optionally saves the computed metrics."""
        if not self.metrics:
            self.compute_metrics()

        LOG.info("---- TRAJECTORY COMPARISON RESULTS ----")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                LOG.info(f"{key}: {value:.4f}")
            else:
                LOG.info(f"{key}: {value}")

        if save:
            out_dir = os.path.join(self.output_dir, "comparison")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "metrics", "metrics_summary.csv")

            pd.DataFrame([self.metrics]).to_csv(out_path, index=False, sep=';')
            LOG.info(f"Metrics saved to: {out_path}")

    # ----------------------------------------------------------------------
    # >>> Error visualisation
    # ----------------------------------------------------------------------
    def plot_analysis(self, show=True, save=True):
        """Plots the component-wise and total trajectory errors."""
        if not self.metrics:
            self.compute_metrics()

        # Check if the data is loaded
        if self.measured_data is None or self.estimated_data is None:
            self.load_data()
            self.align_data()

        meas = self.measured_data[["x", "y", "z"]].to_numpy()
        est = self.estimated_data[["x", "y", "z"]].to_numpy()
        diff = est - meas

        err_x, err_y, err_z = diff[:, 0], diff[:, 1], diff[:, 2]
        err_mag = np.linalg.norm(diff, axis=1)
        frames = np.arange(len(err_x))

        LOG.info("Generating trajectory error plots...")

        # --- Figure 1: Errors per axis ---
        plt.figure(figsize=(10, 6))
        plt.plot(frames, err_x, label="X error [m]", color="red", linewidth=1.0)
        plt.plot(frames, err_y, label="Y error [m]", color="green", linewidth=1.0)
        plt.plot(frames, err_z, label="Z error [m]", color="blue", linewidth=1.0)
        plt.xlabel("Frame index")
        plt.ylabel("Error [m]")
        plt.title("Component-wise Trajectory Error")
        plt.legend()
        plt.grid(True)

        if save:
            out_dir = os.path.join(self.output_dir, "comparison", "plots")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "component_errors.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            LOG.info(f"Component-wise error plot saved to: {out_path}")

        if show:
            plt.show()
        else:
            plt.close()

        # --- Figure 2: Euclidean (aggregated) error magnitude ---
        plt.figure(figsize=(10, 5))
        plt.plot(frames, err_mag, label="Total 3D Error [m]", color="purple", linewidth=1.0)
        plt.xlabel("Frame index")
        plt.ylabel("3D Error Magnitude [m]")
        plt.title("Total 3D Position Error over Time")
        plt.legend()
        plt.grid(True)

        if save:
            out_dir = os.path.join(self.output_dir, "comparison", "plots")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "total_3d_error.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            LOG.info(f"Total 3D error plot saved to: {out_path}")

        if show:
            plt.show()
        else:
            plt.close()

        LOG.info("Error plots generated successfully.")
