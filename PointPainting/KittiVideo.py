import os
import cv2
import numpy as np
from KittiCalibration import KittiCalibration
from BiSeNetv2.utils.label import labels, id2label
# from numba import jit
output_gt = True
def parse_ply(path):
    ply_data = []
    with open(path, "r") as f:
        l_num = 0
        for line in f.readlines():
            l_num += 1
            if l_num > 8:
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                ply_data.append(line.split(' '))
    return np.array(ply_data).astype(np.float32)

# @jit(forceobj=True)
def parse_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape((-1, 4))

class KittiVideo:
    """ Load data for KITTI videos """

    def __init__(self, video_root, calib_root):
        self.video_root = video_root
        self.calib_root = calib_root

        self.calib = KittiCalibration(calib_path=calib_root, from_video=False)
        self.images_dir = os.path.join(self.video_root, 'image_2')
        self.lidar_dir = os.path.join(self.video_root, 'velodyne')
        self.semantic_dir = os.path.join(self.video_root, 'semantic')

        self.images_filenames = sorted(
            [os.path.join(self.images_dir, filename) for filename in os.listdir(self.images_dir)]
        )

        self.lidar_filenames = sorted(
            [os.path.join(self.lidar_dir, filename) for filename in os.listdir(self.lidar_dir)]
        )
        if output_gt:
            self.semantic_filenames = sorted(
                [os.path.join(self.semantic_dir, filename) for filename in os.listdir(self.semantic_dir)]
            )

        self.num_samples = min(len(self.images_filenames), len(self.lidar_filenames))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = self.__get_image(index)
        # print(index)
        if output_gt:
            semantic = self.__get_semantic(index)
            return image, self.__get_lidar(index), self.__get_calibration(), semantic
        return image, self.__get_lidar(index), self.__get_calibration()

    def __get_image(self, index):
        assert index < self.num_samples
        path = self.images_filenames[index]
        return cv2.imread(path)

    def __get_semantic(self, index):
        assert index < self.num_samples
        path = self.semantic_filenames[index]
        # print(path)
        sem = cv2.imread(path, cv2.IMREAD_COLOR)
        semantic = np.asarray(sem)
        semantic = semantic[:, :, 2]
        # print(sem.shape)
        new_shape = (1024, 512)
        semantic = cv2.resize(semantic, new_shape,  interpolation=cv2.INTER_NEAREST)
        # print(semantic.shape)
        # print(semantic)
        semantic = self.remove_ignore_index_labels(semantic)
        # print(semantic.shape)
        # print(semantic)
        # print(' ')
        return semantic

    def remove_ignore_index_labels(self, semantic):
        for id in id2label:
            label = id2label[id]
            trainId = label.trainId
            semantic[semantic == id] = trainId
        # print(np.unique(semantic))
        return semantic

    # @jit
    def __get_lidar(self, index):
        assert index < self.num_samples
        path = self.lidar_filenames[index]
        # print(path)
        lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
        # print(lidar)
        # lidar = lidar[:, [2, 0, 1, 3]]
        # return parse_bin(path)
        return lidar
        # lidar = parse_ply(path)
        # lidar = lidar[:,[1,2,0,3]]
        # lidar[:,0] *= -1
        # # print(lidar.shape)
        # return lidar

    def __get_calibration(self):
        return self.calib