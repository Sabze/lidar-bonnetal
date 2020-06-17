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


def train_models_with_pre(pretrained_models, arch_files, dc, dataset, dirname, train_tracker ):
    for pretrain in pretrained_models:
        pre_name = os.path.basename(pretrain[:-1])
        print("Pretraining model:", pre_name)
        for i, arch_file in enumerate(arch_files):
            print("Training with arch: {}".format(arch_file))
            arch_name = os.path.splitext(os.path.basename(arch_file))[0]
            logname = "{dir}/{pname}-{arch_name}-{num}".format(pname=pre_name, arch_name=arch_name, num=train_tracker.counter, dir=dirname)
            logpath = os.path.join(logdir, logname)
            os.system("./train.py -d {data} -ac {arch} -dc {data_config} -l {log} -p {pretrained}".format(data=dataset,
                                                                                                    arch=arch_file,
                                                                                                    data_config=dc,
                                                                                                    log=logpath,
                                                                                                    pretrained=pretrain))
            train_tracker.update(logpath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./train_multiple_pretrain.py", description='Train multiple models with different '
                                                                               'pretraining files and arch config '
                                                                               'files. This script does NOT work '
                                                                               'directly from command-line, and you '
                                                                               'MUST change the arguments in the '
                                                                               'actual file.')
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to train with. No Default',
  )
  parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      required=False,
      default='config/labels/semantic-kitti.yaml',
      help='Classification yaml cfg file. See /config/labels for sample. No default!',
  )
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

  FLAGS, unparsed = parser.parse_known_args()
  logdir = FLAGS.logdir

  warnings.warn("This script does not work directly from command-line,"
                " you have to specify the models IN the file itself!")


  # ---------- Change this section before you run the script! ----------

  # Specify the 21 models to be used as pretraining
  pretraining_21models = ["/tf/rangenet/lidar-bonnetal/logs/kitti/pretrained_models/darknet21"]

  # Speficy the 53 models to be used as pretraining
  pretraining_53models = ["/tf/rangenet/lidar-bonnetal/logs/kitti/pretrained_models/darknet53"]

  # Speficy the 21 arch config dir
  arch_21_files =  sorted(glob.glob("config/arch/uav/21/*"))

  # Speficy the 53 arch config dir
  arch_53_files = sorted(glob.glob("config/arch/uav/53/*"))

  # -------------------------------------------------------------------------

  dc = FLAGS.data_cfg

  print("Training 21 arch files:")
  pprint.pprint(arch_21_files)
  print("\nUsing pretrained 21 models:")
  pprint.pprint(pretraining_21models)
  print("\n\nTraining 53 arch files:")
  pprint.pprint(arch_53_files)
  print("\nUsing pretrained 53 models:")
  pprint.pprint(pretraining_53models)

  training_tracker = TrainTracker()
  train_models_with_pre(pretraining_21models, arch_21_files, dc, FLAGS.dataset, 21, training_tracker)
  train_models_with_pre(pretraining_53models, arch_53_files, dc, FLAGS.dataset, 53, training_tracker)

  pprint.pprint(training_tracker.logfiles)
  if FLAGS.evaluate:
      for model in training_tracker.logfiles:
          print(model)
          evaluation_tools.evaluate_model(FLAGS.dataset, model)

