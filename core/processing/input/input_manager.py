import os

from selfmotionestimation.data.log.logger import Logger

LOG = Logger("InputManager")


class InputManager:
    """
    Handles loading of image sequences or video files from a given source directory.
    """

    def __init__(self):
        self.file_paths = []

    def load_input_source(self, input_source: str):
        """
        Loads image or video files from the specified input directory.
        :param input_source: Path to the directory containing files.
        :return: List of file paths.
        """
        self.file_paths.clear()

        if not os.path.isdir(input_source):
            LOG.error(f"Error: The specified path ({input_source}) is not a valid folder.")
            return []

        allowed_extensions = {'.jpeg', '.jpg', '.png'}

        for filename in sorted(os.listdir(input_source)):
            full_path = os.path.join(input_source, filename)
            if os.path.isfile(full_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in allowed_extensions:
                    self.file_paths.append(full_path)

        LOG.info(f"Loaded files ({'Images'}): {len(self.file_paths)} pieces.")
        return self.file_paths
