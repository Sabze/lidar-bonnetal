
import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

from tasks.semantic.modules.user import *
import glob
import warnings
import pprint
import shutil
import evaluation_tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./evaluate_multiple.py", description="Evaluates several models.")
    parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to train with. No Default',
    )
    parser.add_argument(
      '--models', '-m',
      type=str,
      required=True,
      default=None,
      help='Directories to get the trained models.'
    )

    parser.add_argument(
        '--single', '-s',
        action='store_true',
        help='Specify if \'models\' is a file instead of a directory (a singel model is evaluated).'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Specify if you want to save the predictions'
    )

    parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      default=None,
      help='Specify if you want to use a different data config file than the one in the model directory.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("EVALUATE MULTIPLE:")
    print("dataset", FLAGS.dataset)
    print("model: ", FLAGS.models)
    print("Data config: ", FLAGS.data_cfg)
    print("Saving predictions: ", FLAGS.save)
    print("----------\n")

    if FLAGS.single:
        model_dirs = [FLAGS.models]
    else:
        model_dirs = sorted(glob.glob(FLAGS.models + "/*/"))
    print(model_dirs)
    for model_dir in model_dirs:
        evaluation_tools.evaluate_model(FLAGS.dataset, model_dir, data_config=FLAGS.data_cfg, save=FLAGS.save)