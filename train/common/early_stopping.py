
import numpy as np
import logging
import os
from .train_logger import set_up_logger


class EarlyStopping:
    """ Stops the training early if the a specified metrix doesn't improve after a given patience."""
    DEFAULT_PATIENCE = 7
    DEFAULT_DELTA = 0
    DEFAULT_VERBOSE = False
    TYPE = "iou"
    SUPPORTED_METRICS = ["iou", "loss"]
    def __init__(self, params, log_dir):
        """
        Args:
            params (dict): A dict with the following parameters:
                patience (int): How long to wait after last time validation loss improved.
                                Default: 7
                verbose (bool): If True, prints a message for each validation loss improvement.
                                Default: False
                delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
                type (str):    The type of metrix that has to improve has to be one of SUPPORTED_METRICS.
                                Default: "iou"
        """
        logfile = os.path.join(log_dir, "early_stopping.log")
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        if os.path.exists(logfile):
            print("WARNING - logfile does already exist")
        self.logger = set_up_logger(logfile, "es")
        if "patience" in params:
            self.patience = params["patience"]
        else:
            self.logger.logging.warning("Using default patience: {patience}".format(patience=self.DEFAULT_PATIENCE))
            self.patience = self.DEFAULT_PATIENCE

        if "delta" in params:
            self.delta = params["delta"]
        else:
            logging.warning("Using default delta: {delta}".format(delta=self.DEFAULT_DELTA))
            self.delta = self.DEFAULT_DELTA

        if "verbose" in params:
            self.verbose = params["verbose"]
        else:
            self.verbose = self.DEFAULT_VERBOSE
        if "type" in params:
            if params["type"] not in self.SUPPORTED_METRICS:
                logging.warning("The Early stopping metric {} is not supported, stopping".format(params["type"]))
                raise ValueError("The Early stopping metric {} is not supported.".format(params["type"]))
            self.type = params["type"]
        else:
            self.logger.warning("Using default type: {type}".format(type=self.TYPE))
            self.type = self.TYPE
        start_msg = "Using early stopping with patience = {patience},  delta = {delta}," \
                    " verbose = {verbose}, type= {type}".format(patience=self.patience, delta = self.delta,
                                                                verbose=self.verbose, type=self.type)
        self.logger.info(start_msg)
        print(start_msg)
        self.counter = 0
        self.early_stop = False
        if self.type == "loss":
            self.best_val = np.Inf
        else:
            self.best_val = 0

    def update(self, curr_val_value, epoch_nr=""):
        """Update the early-stopping with new value"""
        improved = False
        if self.type == "loss" and (curr_val_value + self.delta) < self.best_val:
            # The validation loss has improved
            improved = True
        elif self.type == "iou" and (curr_val_value - self.best_val) > self.delta:
            # The validation iou has improved
            improved = True
        if improved:
            update_msg = "[{epoch}] - Validation {type} improved ({old_best:.6f} --> " \
                         "{new_best:.6f}).".format(old_best=self.best_val, type=self.type, new_best=curr_val_value,
                                                   epoch=epoch_nr)
            if self.verbose:
                print(update_msg)
            self.logger.info(update_msg)
            self.best_val = curr_val_value # Save new best loss
            self.counter = 0    # Reset counter
        else:
            # No improvement, add to counter
            self.counter += 1
            self.logger.info("[{epoch}] - ({value:.4f}) Counter is {counter} out of patience {patience} ".format(counter=self.counter,
                                                                                               patience=self.patience,
                                                                                               epoch= epoch_nr, value=curr_val_value))
            # Check if we reached the patience
            if self.counter >= self.patience:
                self.logger.info("[{epoch}] - Stopping at best value: {min_loss:.6f} "
                             "(current value: {curr_loss:.6f})".format(epoch=epoch_nr, min_loss=self.best_val,
                                                                      curr_loss=curr_val_value))
                print("Validation metric has not improved in {patience} epochs,"
                      " stopping early".format(patience=self.patience))
                self.early_stop = True

