import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from common.laserscan import LaserScan, SemLaserScan
import random
from torchvision import transforms

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

# TODO: Create a different parser/dataloader for the UAV data

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

class HeightChange(object):
    def __init__(self, params):
        self.max_height_diff, self.max_angle_diff, self.augm_prob, self.color_map = params

    def __call__(self, scan):
        prob = random.random()
        if prob > self.augm_prob:
            return scan
        else:
            diff = random.uniform(-self.max_height_diff, self.max_height_diff)
            points = scan.points + np.array([diff, 0, 0])
            theta = np.arcsin(scan.points[:, 2] / np.linalg.norm(scan.points, axis=1))
            fov_up = np.min([np.max(theta) * 180 / np.pi, scan.proj_fov_up + self.max_angle_diff])
            fov_down = np.max([np.min(theta) * 180 / np.pi, scan.proj_fov_down - self.max_angle_diff])
            new_laserscan = SemLaserScan(self.color_map,
                                         project=scan.project,
                                         H=scan.proj_H,
                                         W=scan.proj_W,
                                         fov_up=fov_up,
                                         fov_down=fov_down)
            new_laserscan.set_points(points, scan.remissions)
            return new_laserscan

class Flip(object):
    def __init__(self, params):
        self.augm_prob, self.color_map = params

    def __call__(self, scan):
        v_prob = random.random()
        h_prob = random.random()
        if v_prob > self.augm_prob and h_prob > self.augm_prob:
            return scan
        else:
            v_flip = True
            h_flip = True
            if v_prob > self.augm_prob:
                v_flip = False
            if h_flip > self.augm_prob:
                h_flip = False
            new_laserscan = SemLaserScan(self.color_map,
                                         project=scan.project,
                                         H=scan.proj_H,
                                         W=scan.proj_W,
                                         fov_up=scan.proj_fov_up,
                                         fov_down=scan.proj_fov_down, h_flip=h_flip, v_flip=v_flip)
            new_laserscan.set_points(scan.points, scan.remissions)
            return new_laserscan

class Translate(object):
    """ Change the view-point of the points.
    Default parameters changes the view point to the one for the UAV data."""
    def __init__(self, params):
        self.augment, self.x, self.z, self.angle = params

    def translate_points(self, points, yangle=-1.25*np.pi/2, x=13, z=-5):
        if self.augment:
            # Random augment
            xdiff = random.uniform(-self.x, self.x)
            zdiff = random.uniform(-self.z, self.z)
            angle_diff = random.uniform(-self.angle, self.angle)
            x += xdiff
            z += zdiff
            yangle += angle_diff
        # Translate the points
        cosvert = np.cos(yangle)
        sinvert = np.sin(yangle)
        rot_matrix = np.array([[cosvert, 0, sinvert],
                               [0, 1, 0],
                               [-sinvert, 0, cosvert]])

        rotated_points = (rot_matrix @ (points.T)).T
        # Move the x and z points.
        translated_points = rotated_points + [x, 0, z]
        return translated_points

    def __call__(self, scan):
        translated_points = self.translate_points(scan.points)
        scan.overwrite_points(translated_points)
        return scan


def get_scan_weights(root, sequences, class_weights):
    """ Calculate the weights used in the weighted sampling."""
    root = os.path.join(root, "sequences")
    all_labels_files = []
    for seq in sequences:
        # to string
        seq = '{0:02d}'.format(int(seq))
        print("creating weights for seq {}".format(seq))

        # get paths for each
        scan_path = os.path.join(root, seq, "velodyne")
        label_path = os.path.join(root, seq, "labels")

        # get files
        scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
        label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]
        assert(len(scan_files) == len(label_files))
        # extend list
        all_labels_files.extend(label_files)
    all_labels_files.sort()
    scan_weights =[]
    for i in range(0, len(all_labels_files)):
        filename = all_labels_files[i]
        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not filename.endswith(".label"):
            raise RuntimeError("Filename extension is not valid label file.")

        # if all goes well, open label
        labels = np.fromfile(filename, dtype=np.int32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF
        label_types, nums = np.unique(labels, return_counts=True)
        weight = 0
        for label_type, num in zip(label_types, nums):
            if class_weights[label_type] != 0:
                weight += num * 1/class_weights[label_type]
        weight /= labels.shape[0]
        scan_weights.append(np.round(weight)**2)
    return scan_weights


class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               max_points=150000,   # max number of points present in dataset
               gt=True,             # send ground truth?
               transform=None):
    # save deats
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt
    self.transform = transform

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "velodyne")
      label_path = os.path.join(self.root, seq, "labels")

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]

      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # extend list
      self.scan_files.extend(scan_files)
      self.label_files.extend(label_files)

    # sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()

    print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                    self.sequences))

  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.scan_files[index]
    if self.gt:
      label_file = self.label_files[index]

    # open a semantic laserscan
    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

    # open and obtain scan
    scan.open_scan(scan_file)
    if self.transform:
        scan = self.transform(scan)
    if self.gt:
      scan.open_label(label_file)
      # map unused classes to used classes (also for projection)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    # make a tensor of the uncompressed data (with the max num points)
    unproj_n_points = scan.points.shape[0]
    unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
    unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
    unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
    if self.gt:
      unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
    else:
      unproj_labels = []

    # get points and labels
    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)
    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []
    proj_x = torch.full([self.max_points], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
    proj_y = torch.full([self.max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
    proj = (proj - self.sensor_img_means[:, None, None]
            ) / self.sensor_img_stds[:, None, None]
    proj = proj * proj_mask.float()

    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")
    # print("path_norm: ", path_norm)
    # print("path_seq", path_seq)
    # print("path_name", path_name)

    # return
    return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

  def __len__(self):
    return len(self.scan_files)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True, # shuffle training set?
               weighted_data=False,
               label_weights=None,
               data_augmentation=False,
               augmentation_params=None):
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)

    # Data loading code
    transformations = []
    if data_augmentation:
        if augmentation_params["height_change"]["use"]:
            # Add height/altitude change data augmentation
            height_params = augmentation_params["height_change"]["params"]
            height_diff = height_params["max_height_diff"]
            angle_diff = height_params["max_angle_diff"]
            augment_prob = height_params["probability"]
            print("Using Data Augmentatation with height diff: {height}, "
                  "angle diff: {angle} and augm prog: {prob}".format(height=height_diff, angle=angle_diff,
                                                                     prob=augment_prob))
            height_transform = HeightChange((height_diff, angle_diff, augment_prob, self.color_map))
            transformations.append(height_transform)
        if augmentation_params["flip"]["use"]:
            # Add flip data augmentation
            flip_prob = augmentation_params["flip"]["params"]["probability"]
            flip_transform = Flip((flip_prob, self.color_map))
            transformations.append(flip_transform)
            print("Using Flip Augmentatation with probability: {prob}".format(prob=flip_prob))
        if augmentation_params["translate"]["use"]:
            # Add translate data augmentation
            # TODO: create a more general translate data augmentation (merging translate and height change)
            transl_params = augmentation_params["translate"]["params"]
            random_augment =  transl_params["augment"]
            if random_augment:
                x_diff =  transl_params["x"]
                z_diff = transl_params["z"]
                angle_diff = transl_params["angle"]
            else:
                x_diff =  0
                z_diff = 0
                angle_diff = 0
            translation = Translate((random_augment, x_diff, z_diff, angle_diff))
            transformations.append(translation)
            print("Using Translate Augmentatation with: random_aug: {aug}, x_diff: {x}, z_diff: {z},"
                  " angle_diff: {angle}".format(aug=random_augment, x=x_diff,z=z_diff, angle=angle_diff))
        if len(transformations) == 0:
            print("Data augmentation is True, but no data augmentation type is given.")
            train_transformations = None
        else:
            train_transformations =  transform=transforms.Compose(transformations)
    else:
        train_transformations = None
    self.train_dataset = SemanticKitti(root=self.root,
                                       sequences=self.train_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       transform=train_transformations,
                                       gt=self.gt)
    if weighted_data:
        scan_weights = get_scan_weights(root, train_sequences, label_weights)
        label_types, nums = np.unique(scan_weights, return_counts=True)
        scan_weights = torch.tensor(scan_weights)
        weights_summary =  {label_type: num for label_type, num in zip(label_types, nums)}
        print("Using weighted sampling with: Scan weights:\n {}".format(weights_summary))
        sampler = WeightedRandomSampler(scan_weights, len(scan_weights), replacement=True)
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=self.batch_size,
                                                       sampler=sampler,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=True)
    else:
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=self.shuffle_train,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=True)
    assert len(self.trainloader) > 0
    self.trainiter = iter(self.trainloader)
    if augmentation_params is not None and augmentation_params["translate"]["use"]:
        valid_transformations = transforms.Compose([Translate((False, 0, 0, 0))])
    else:
        valid_transformations = None
    self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       transform=valid_transformations,
                                       gt=self.gt)
    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)
    if self.test_sequences:
      self.test_dataset = SemanticKitti(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        sensor=self.sensor,
                                        max_points=max_points,
                                        gt=False)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)


  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)
