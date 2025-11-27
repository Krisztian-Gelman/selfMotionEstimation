import os
import datetime
import logging

class Logger:
    """
    Unified logging system for the Self-Motion Estimation project.
    Usage:
        from selfmotionestimation.data.log.logger import logger
        LOG = logger("VideoHandler")
        LOG.info("Flow estimation started")
    """

    _initialized = False
    _log_path = None

    def __init__(self, module_name: str):
        """
        Initialize the logger for the given module.
        Each instance shares the same file and console handlers.
        """
        # --- Ensure log/output directory exists ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "output")
        os.makedirs(log_dir, exist_ok=True)

        # --- Create the log file only once per run ---
        if not Logger._initialized:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            Logger._log_path = os.path.join(log_dir, f"run_{timestamp}.log")

            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)

            formatter = logging.Formatter('[%(name)s] %(asctime)s - %(levelname)s - %(message)s')

            # File handler
            file_handler = logging.FileHandler(Logger._log_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

            Logger._initialized = True

        # --- Create or retrieve logger for this module ---
        self.LOG = logging.getLogger(module_name)
        self.LOG.setLevel(logging.INFO)
        self.LOG.propagate = True  # forward messages to root handlers

    # --- Wrapper methods ---
    def info(self, message: str):
        self.LOG.info(message)

    def warning(self, message: str):
        self.LOG.warning(message)

    def error(self, message: str):
        self.LOG.error(message)

    def debug(self, message: str):
        self.LOG.debug(message)

    def critical(self, message: str):
        self.LOG.critical(message)
