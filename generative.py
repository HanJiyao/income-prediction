import numpy as np
import sys
import csv
from numpy.linalg import inv


class DataManager:
    def __init__(self):
        self.data = {}
        self.mean = 0.
        self.std = 0.
        self.theta = 0.

    def read(self, name, path):
        with open(path, newline='') as csvfile:
            rows = np.array(list(csv.reader(csvfile))[1:], dtype=float)
            self.mean = np.mean(rows, axis=0).reshape(1, -1)
            self.std = np.std(rows, axis=0).reshape(1, -1)
            self.theta = np.ones((rows.shape[1]+1, 1), dtype=float)
            for i in range(rows.shape[0]):
                rows[i, :] = (rows[i, :] - self.mean) / self.std
            self.data[name] = rows

    def find_theta(self):
        class_0_id = []
        class_1_id = []
        for i in range(self.data['Y_train'].shape[0]):
            if self.data['Y_train'][i][0] == 0:
                class_0_id.append(i)
            else:
                class_1_id.append(i)
        class_0 = self.data['X_train'][class_0_id]
        class_1 = self.data['X_train'][class_1_id]
        mean_0 = np.mean(class_0, axis=0)
        mean_1 = np.mean(class_1, axis=0)

        n = class_0.shape[1]
        cov_0 = np.zeros((n, n))
        cov_1 = np.zeros((n, n))

        for i in range(class_0.shape[0]):
            cov_0 += np.dot(np.transpose([class_0[i] - mean_0]), [(class_0[i] - mean_0)]) / class_0.shape[0]
