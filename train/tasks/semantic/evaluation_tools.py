
import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil

import glob
import warnings
import shutil
DATA_CONFIG = "data_cfg.yaml"
LABEL_PREDICTIONS = "label_predictions"
RESULT_FILE = "evaluation_results.txt"


def evaluate_model(dataset, model_dir, data_config=None, save=False):
    """ Run inference and save the output from evaluate_iou.py in a file called evaluation_results.txt.

    Args:
        dataset (str):      Path to dataset.
        model_dir (str):    Path to directory with model.
        data_config (str):  Path to the data config file, default uses the models data config file.
        save (bool):        True if the predictions should be saved, False otherwise.
    """
    print("Infering model: {}".format(os.path.basename(model_dir[:-1])))
    if not os.path.isdir(model_dir):
        print("{} is not a directory, skipping files..")
    else:
        pred_dir = os.path.join(model_dir, LABEL_PREDICTIONS)

        if data_config is None:
            # Use data configuration of model
            data_cfg = os.path.join(model_dir, DATA_CONFIG)
        else:
            # Use specified data configuration
            data_cfg = data_config
        if not os.path.exists(data_cfg):
            warnings.warn("data config file does not exist. Skipping model..")
        else:
            print("Using data config: {}".format(data_cfg))
            # Run inference
            if data_config is None:
                os.system("./infer.py -d {data} -l {log} -m {model}".format(data=dataset, log=pred_dir,
                                                                            model=model_dir))
            else:
                os.system("./infer.py -d {data} -l {log} -m {model} -dc {dc}".format(data=dataset, log=pred_dir,
                                                                            model=model_dir, dc=data_cfg))
            print("Evaluating...")
            output_file = os.path.join(model_dir, RESULT_FILE)
            with open(output_file, "a+") as output:

                # Evaluate the train data
                output.writelines(["_"*120+"\n", "_"*120+"\n"])
                output.write("  " *20 +"\n" + datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + "\n")
                output.writelines(["_" * 120 + "\n", "_" * 120 + "\n"])
                output.writelines(["\n" + "-" * 80 + "\n", "TRAIN\n", "-" * 80 + "\n"])
                output.flush()
                subprocess.call(["python", "./evaluate_iou.py", "-d", dataset, "-p", pred_dir, "-dc",
                                 data_cfg, '-s', "train"], stdout=output)

                # Evaluate the validation data
                output.writelines(["\n" + "-" * 80 + "\n", "VALIDATION\n", "-" * 80 + "\n"])
                output.flush()
                subprocess.call(["python", "./evaluate_iou.py", "-d", dataset, "-p", pred_dir, "-dc",
                                 data_cfg, '-s', "valid"], stdout=output)

                # Evaluate the test data
                output.writelines(["\n" + "-" * 80 + "\n", "TEST\n", "-" * 80 + "\n"])
                output.flush()
                subprocess.call(["python", "./evaluate_iou.py", "-d", dataset, "-p", pred_dir, "-dc",
                                 data_cfg, '-s', "test"], stdout=output)

            print("Saving results in: {}".format(output_file))
            print("Removing predictions: {}".format(pred_dir))
            if not save:
                shutil.rmtree(pred_dir)