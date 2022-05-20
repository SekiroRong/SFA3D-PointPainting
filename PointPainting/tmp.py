# -*- coding = utf-8 -*-
# @Time : 2022/1/22 20:43
# @Author : 戎昱
# @File : tmp.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import numpy as np

path = r"G:\KITTI\training\velodyne\000000.bin"
lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
print(lidar)