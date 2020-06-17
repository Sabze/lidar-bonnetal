
import logging
import os

def set_up_logger(logfile, logger_name):
    """Configure and set up the logger."""
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    fileHandler = logging.FileHandler(logfile, mode='a')
    fileHandler.setFormatter(formatter)
    log_setup.setLevel(logging.INFO)
    log_setup.addHandler(fileHandler)
    return logging.getLogger(logger_name)

class TrainLogger:
    """Log information after every epoch. """
    def __init__(self, log_dir):
        logfile = os.path.join(log_dir, "train_process.log")
        if os.path.exists(logfile):
            print("WARNING - logfile does already exist")
        self.logger = set_up_logger(logfile, "train")

    def warning(self, message, epoch):
        self.logger.warning("[{epoch}] - {message}".format(epoch=epoch, message=message))

    def info(self, message, epoch):
        self.logger.info("[{epoch}] - {message}".format(epoch=epoch, message=message))

    def success(self):
        self.logger.info("Success")