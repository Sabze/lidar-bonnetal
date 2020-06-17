import os
import argparse
import yaml
import datetime
import os.path
from shutil import copyfile
import subprocess
TEMPDIR = "config/temp"
import evaluation_tools
import glob
import pprint
import warnings
from modules.train_tracker import TrainTracker


def train_models(arch_files:list, dc:str, dataset:str, dirname:str, train_tracker:TrainTracker):
  """Train one model per arch file with the same data configuration"""
  for i, arch_file in enumerate(arch_files):
    print("Training with arch: {}".format(arch_file))
    arch_name = os.path.splitext(os.path.basename(arch_file))[0]
    logname = "{dir}/{arch_name}-{num}".format(arch_name=arch_name, num=train_tracker.counter, dir=dirname)
    logpath = os.path.join(logdir, logname)
    os.system("./train.py -d {data} -ac {arch} -dc {data_config} -l {log} ".format(data=dataset,
                                                                                   arch=arch_file,
                                                                                   data_config=dc,
                                                                                   log=logpath))
    train_tracker.update(logpath)
  return



if __name__ == '__main__':
  parser = argparse.ArgumentParser("./train_multiple_resolution.py", description='Train multiple models with '
                                                                                 'different config files (arch, data)'
                                                                                 'and different datasets. '
                                                                                 'This script does NOT work '
                                                                                 'directly from command-line, and you '
                                                                                 'MUST change the arguments in the '
                                                                                 'actual file.')
  parser.add_argument(
      '--logdir', '-l',
      type=str,
      default=os.path.expanduser("~") + '/logs/' +
      datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
      help='Directory to put the log data. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--evaluate', '-e',
      action='store_true',
      help='Specify if the models should be evaluated after trained'
  )

  warnings.warn("This script does NOT work directly from command-line,"
                " you have to specify the models IN the file itself!")

  # ---------- Change this section before you run the script! ----------

  # Specify the 16 channel arch config files
  low_res_models = sorted(glob.glob("config/arch/pretrained_models/remission/16/*"))

  # Specify the 64 channel arch config files
  high_res_models = sorted(glob.glob("config/arch/pretrained_models/remission/64/*"))

  # Specify the data of the 16 channel scans
  low_data = "/tf/data/SemanticKitti/180fov"

  # Specify the data of the 64 channel scans
  high_data = "/tf/data/SemanticKitti/64x360"


  # Specify the label config file of the 16 channel data
  low_dc = "config/labels/semantic-kitti-dwn.yaml"

  # Specify the label config file of the 64 channel data
  high_dc = "config/labels/semantic-kitti.yaml"

  # Specify the name of the logdir for the 16 channel files
  low_res_name = "16x1024"

  # Specify the name of the logdir for the 64 channel files
  high_res_name = "64x1024"

  # -------------------------------------------------------------------------

  FLAGS, unparsed = parser.parse_known_args()
  logdir = FLAGS.logdir

  print("Training low_res_models:")
  pprint.pprint(low_res_models)
  print("\nUsing high_res_models:")
  pprint.pprint(high_res_models)

  training_tracker = TrainTracker()
  train_models(low_res_models, low_data, low_dc, low_res_name, training_tracker)
  train_models(high_res_models, high_data, high_dc, high_res_name , training_tracker)

  pprint.pprint(training_tracker.logfiles)

  if FLAGS.evaluate:
      for model in training_tracker.logfiles:
          evaluation_tools.evaluate_model(FLAGS.dataset, model)

