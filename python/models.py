import os
import math
import numpy as np
import pandas
import training_data as td
from scipy.stats import truncnorm
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

NANO_TO_SEC = 1e09
VP_ANGLE_DISTRIBUTIONS_SIGMA = np.deg2rad(5.0)
IMU_COLUMNS = ['gyro_x', 'gyro_y', 'gyro_z',
               'linacce_x', 'linacce_y', 'linacce_z',
               'grav_x', 'grav_y', 'grav_z']
POSITION_COLUMNS = ['pos_x', 'pos_y', 'pos_z']
ORIENTATION_COLUMNS = ['ori_w', 'ori_x', 'ori_y', 'ori_z']
GRAVITY_COLUMNS = ['grav_x', 'grav_y', 'grav_z']
TIME_COLUMN = 'time'


class BaseModel:
    def __init__(self):
        self.softmax = False
        self.training_features = None
        self.training_targets = None
        self.test_targets = None
        self.test_features = None
        self.feature_smooth_sigma = 0
        self.target_smooth_sigma = 0

    def input_dim(self):
        return self.training_features[0].shape[1]

    def output_dim(self):
        return self.training_targets[0].shape[1]

    def _process_feature(self, data_all):
        return None

    def _process_target(self, data_all):
        return None, None

    def load_data(self, path):
        features_all = []
        targets_all = []

        def load(file):
            data_all = pandas.read_csv(file)
            feature = self._process_feature(data_all)
            target, valid = self._process_target(data_all)
            if valid is not None:
                feature = feature[valid == 1]
                target = target[valid == 1]
            if self.feature_smooth_sigma > 0:
                feature = gaussian_filter1d(feature, sigma=self.feature_smooth_sigma, axis=0)
            if self.target_smooth_sigma > 0:
                target = gaussian_filter1d(target, sigma=self.target_smooth_sigma, axis=0)
            features_all.append(np.array(feature))
            targets_all.append(np.array(target))

        if os.path.isdir(path):
            load(os.path.join(path, 'processed', 'data.csv'))
            return features_all, targets_all
        root_dir = os.path.dirname(path)
        with open(path) as f:
            datasets = f.readlines()
        for data in datasets:
            if data[0] == '#':
                continue
            [data_name, _] = [x.strip() for x in data.split(',')]
            load(os.path.join(root_dir, data_name, 'processed', 'data.csv'))
        return features_all, targets_all

    def normalize_input(self):
        # first compute the variable of all channels
        targets_concat = np.concatenate(self.training_targets, axis=0)

        target_mean = np.mean(targets_concat, axis=0)
        target_variance = np.var(targets_concat, axis=0)

        # normalize the input
        for i in range(len(self.training_targets)):
            self.training_targets[i] = np.divide(self.training_targets[i] - target_mean, target_variance)
        for i in range(len(self.test_targets)):
            self.test_targets[i] = np.divide(self.test_targets[i] - target_mean, target_variance)

    def training_data(self):
        return zip(self.training_features, self.training_targets)

    def validation_data(self):
        return zip(self.test_features, self.training_targets)


class VelocityModel(BaseModel):
    def __init__(self, training_list, validation_list, feature_smooth_sigma=0, target_smooth_sigma=0):
        super().__init__()
        self.training_features, self.training_targets = self.load_data(training_list)
        self.test_features, self.test_targets = self.load_data(validation_list)

    def _process_feature(self, data_all):
        return data_all[IMU_COLUMNS].values

    def _process_target(self, data_all):
        ts = data_all[TIME_COLUMN].values / NANO_TO_SEC
        position = data_all[POSITION_COLUMNS].values
        orientation = data_all[ORIENTATION_COLUMNS].values
        gravity = data_all[GRAVITY_COLUMNS]
        return td.compute_local_speed_with_gravity(ts, position, orientation, gravity)


class PositionModel(BaseModel):
    def __init__(self, training_list, validation_list):
        super().__init__()
        self.training_features, self.training_targets = self.load_data(training_list)
        self.test_features, self.test_targets = self.load_data(validation_list)

    def _process_feature(self, data_all):
        feature_vectors = data_all[IMU_COLUMNS]
        position = data_all[POSITION_COLUMNS].values
        position -= position[0]
        return np.concatenate((feature_vectors[:-1], position[:-1]), axis=1)

    def _process_target(self, data_all):
        position = data_all[POSITION_COLUMNS].values
        position -= position[0]
        return position[1:]


class AngleModel(BaseModel):
    def __init__(self, training_list, validation_list=None):
        super().__init__()
        self.softmax = True
        self.pdf_size = 360
        self.pdf_min = -1
        self.pdf_max = 1
        self.standard_derivation = 0.001
        self.batch_size = 50
        self.training_features, self.training_targets = self.load_data(training_list)
        if validation_list is not None:
            self.test_features, self.test_targets = self.load_data(validation_list)

    def output_dim(self):
        return self.pdf_size

    def _process_feature(self, data_all):
        raw_data = data_all[IMU_COLUMNS].values
        features = []
        for i in range(len(raw_data) // self.batch_size):
            features.append(raw_data[i*self.batch_size:(i+1)*self.batch_size].flatten())
        return np.array(features)

    def _process_target(self, data_all):
        ts = data_all[TIME_COLUMN].values / NANO_TO_SEC
        position = data_all[POSITION_COLUMNS].values
        targes, valid = td.compute_angular_velocity(ts, position, np.arange(0, len(position), self.batch_size),
                                                    self.batch_size)
        return targes[:-1], valid[:-1]

    def angle_to_bin(self, angles):
        result = []
        angles = np.clip(angles, self.pdf_min, self.pdf_max)
        # plt.hist(angles, bins=self.pdf_size)
        # plt.show()
        for angle in angles:
            values = np.linspace(self.pdf_min, self.pdf_max, self.pdf_size)
            values = truncnorm.pdf(values, (self.pdf_min - angle) / self.standard_derivation,
                                   (self.pdf_max - angle) / self.standard_derivation,
                                   loc=angle,
                                   scale=self.standard_derivation)
            values /= sum(values)
            result.append(values)
        return np.array(result)

    def training_data(self):
        return zip(self.training_features, map(self.angle_to_bin, self.training_targets))

    def validation_data(self):
        return zip(self.test_features, map(self.angle_to_bin, self.test_targets))

    def trajectory_from_prediction(self, prediction):
        trajectory = [np.array([0, 0, 0])]
        v = np.array([0, 0, 1])
        values = np.linspace(self.pdf_min, self.pdf_max, self.pdf_size)
        result = []
        for bin in prediction:
            theta = sum(bin * values)
            result.append(theta)
            trajectory.append(trajectory[-1] + v)
            r = np.array([
                math.cos(theta), 0, -math.sin(theta),
                0, 1, 0,
                math.sin(theta), 0, math.cos(theta)
            ]).reshape(3, 3)
            v = np.matmul(r, v)
            v /= np.linalg.norm(v)
        return trajectory

