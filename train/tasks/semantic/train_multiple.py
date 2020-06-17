import os
import argparse
import yaml
import datetime
import os.path
from shutil import copyfile
import subprocess
import evaluation_tools
import glob
TEMPDIR = "config/temp"

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./train_multiple.py", description='Train multiple models with the '
                                                                      'same data configuration file.')
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to train with. No Default',
  )
  parser.add_argument(
      '--arch_cfg_dir', '-ac',
      type=str,
      required=True,
      help='Path to the directory to the architecture yaml cfg files. See /config/arch for sample. No default!',
  )
  parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      required=False,
      default='config/labels/semantic-kitti-uav-custom.yaml',
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
      '--pretrained', '-p',
      type=str,
      required=False,
      default=None,
      help='Directory to get the pretrained model. If not passed, do from scratch!'
  )
  parser.add_argument(
      '--evaluate', '-e',
      action='store_true',
      help='Specify if the models should be evaluated after they are trained'
  )

  FLAGS, unparsed = parser.parse_known_args()
  arch_config_dir = FLAGS.arch_cfg_dir
  logdir = FLAGS.logdir
  if not os.path.isdir(arch_config_dir):
      raise ValueError("The config directory is not a directory {}".format(arch_config_dir))
  arch_files = sorted(glob.glob(arch_config_dir+"/*"))
  if len(arch_files) < 1 or len(arch_files) > 50:
      raise ValueError("Too many or too few file in folder {}".format(arch_files))
  print("Processing {} files".format(len(arch_files)))
  logdirs = []
  dc_config = FLAGS.data_cfg
  
  for i, arch_file in enumerate(arch_files):
      print("Training with arch: {}".format(arch_file))
      basename = os.path.splitext(os.path.basename(arch_file))[0]
      logname = "{arch_name}-{num}".format(arch_name=basename, num =i)
      logpath = os.path.join(logdir, logname)

      if FLAGS.pretrained is None:
          os.system("./train.py -d {data} -ac {arch} -dc {data_config} -l {log}".format(data=FLAGS.dataset,
                                                                                        arch=arch_file,
                                                                                        data_config=dc_config,
                                                                                        log = logpath))
      else:
          os.system("./train.py -d {data} -ac {arch} -dc {data_config} -l {log} -p {pretrained}".format(data=FLAGS.dataset,
                                                                                        arch=arch_file,
                                                                                        data_config=dc_config,
                                                                                        log=logpath,
                                                                                        pretrained=FLAGS.pretrained))
      logdirs.append(logpath)
  if FLAGS.evaluate:
      for model in logdirs:
          evaluation_tools.evaluate_model(FLAGS.dataset, model)
