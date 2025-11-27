import cv2
import os

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("PNGdump")


class PNGDump:
    """
    PNGDump
    --------
    Extracts all frames from a video and saves them as lossless PNG images
    to a specified folder.

    Parameters
    ----------
    video_path : str
        Full path to the input video file.
    output_folder : str
        Path to the folder where images will be saved.
    compression_level : int
        PNG compression level (0=fastest/largest, 9=slowest/smallest).
        Quality is lossless at any level.

    Example
    -------
    >>> dumper = PNGDump(
    ...     video_path="path/to/video.mp4",
    ...     output_folder="path/to/output",
    ...     compression_level=0
    ... )
    >>> dumper.extract_frames_lossless()
    """

    def __init__(self,
                 video_path: str = None,
                 output_folder: str = None,
                 compression_level: int = 0):
        """
        Initializes PNGDump with paths and compression settings.
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.compression_level = compression_level

        LOG.info(f"[PNGDump] Initialized with video: {self.video_path}")
        LOG.info(f"[PNGDump] Output folder: {self.output_folder}")
        LOG.info(f"[PNGDump] Compression level: {self.compression_level}")

    # --------------------------------------------------------------------------
    # >>> MAIN FUNCTION <<<
    # --------------------------------------------------------------------------
    def extract_frames_lossless(self):
        """
        Extracts frames from the specified video file and saves them as PNGs.
        """

        # 1. Create the output directory if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            LOG.info(f"Created output directory: {self.output_folder}")

        # 2. Open the video file
        cap = cv2.VideoCapture(self.video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            LOG.error(f"Error: Could not open video file {self.video_path}")
            return

        # Define parameters for lossless PNG saving
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, self.compression_level]

        frame_count = 0
        LOG.info("Starting frame extraction...")

        # 3. Loop through the video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 4. Construct the filename and save the frame
            frame_filename = os.path.join(self.output_folder, f"frame_{frame_count:05d}.png")

            # Save the frame as an image file with explicit PNG parameters
            success = cv2.imwrite(frame_filename, frame, png_params)
            if not success:
                LOG.warning(f"Failed to save frame {frame_count} to {frame_filename}")

            frame_count += 1

        # 5. Release the video capture object
        cap.release()
        LOG.info(f"Frame extraction complete. Total frames saved: {frame_count} to {self.output_folder}")

        return frame_count
