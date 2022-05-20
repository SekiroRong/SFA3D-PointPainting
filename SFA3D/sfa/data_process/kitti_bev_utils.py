"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
"""

import math
import os
import sys

import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf

from utils.config import PP,pure,Ultimate


# def __semantics_to_colors(self, semantics):
#     # default color is black to hide outscreen points
#     colors = np.zeros((semantics.shape[0], 3))
#
#     for id in trainId2label:
#         label = trainId2label[id]
#         if id == 255 or id == -1:
#             continue
#
#         color = label.color
#         indices = semantics == id
#         colors[indices] = (color[0] / 255, color[1] / 255, color[2] / 255)
#
#     return colors

def pause():
    while 1:
        a = 0

def makeBEVMap(PointCloud_, boundary, mode='val'):
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)

    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    # print(PointCloud.shape)
    # print(PointCloud[:, 0:2].shape)
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    # print(sum(unique_counts))

    # if mode == 'test':
    #     unique_counts = np.array(unique_counts)
    #     crowd = np.where(unique_counts>5)
    #     crowd = np.array(crowd).squeeze(0)
    #     for i in crowd:
    #         uc = unique_counts[i]
    #         ui = unique_indices[i]
    #         sem = PointCloud[ui:ui+uc][:, 4]
    #         sem = np.array(sem, dtype='int16')
    #         counts = np.bincount(sem)
    #         # 返回众数
    #         argmax = np.argmax(counts)
    #         PointCloud[ui][4] = argmax

    PointCloud_top = PointCloud[unique_indices]
    # PointCloud_top = PointCloud
    # print(PointCloud_top.shape)
    # print(PointCloud_top)
    if not pure:
        Pedestrian = np.where(PointCloud_top[:, 4] == 2)
        # print(Pedestrian)
        Vehicles = np.where(PointCloud_top[:, 4] == 8)

        if mode == 'train':
            p_scale = np.random.uniform(0.2, 0.6)
            v_scale = np.random.uniform(0.5, 0.8)
            Pedestrian = np.array(Pedestrian).squeeze(0)
            # print(Pedestrian)
            Pedestrian = np.random.choice(Pedestrian, int(p_scale*Pedestrian.shape[0]),replace=False)

            Vehicles = np.array(Vehicles).squeeze(0)
            # print(Pedestrian)
            Vehicles = np.random.choice(Vehicles, int(v_scale * Vehicles.shape[0]), replace=False)
            # print('Pedestrian')
            # print(Pedestrian)
    # print(Vehicles)

        semantic_cloud = np.zeros((PointCloud_top.shape[0],2))
        semantic_cloud[Pedestrian, 0] = 1
        semantic_cloud[Vehicles, 1] = 1
        # Pedestrian = np.where(semantic_cloud[:, 0] == 1)
        # # print(Pedestrian)
        # Vehicles = np.where(semantic_cloud[:, 1] == 1)
    # print(Vehicles)

    # print(PointCloud_top)
    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))
    semanticMapP = np.zeros((Height, Width))
    semanticMapV = np.zeros((Height, Width))
    semanticMap = np.zeros((Height, Width))
    # print(semanticMap.shape)
    # print(semanticMap[0].shape)
    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    if not pure:
        semanticMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 4]
        semanticMapP[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = semantic_cloud[:, 0]
        semanticMapV[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = semantic_cloud[:, 1]
    # intensityMap 确实是trainID
    # print(semanticMap.shape)
    # print(intensityMap)
    # Pedestrian = np.where(intensityMap == 2)
    # print(Pedestrian)
    # semanticMap[0,Pedestrian] = 1
    # Vehicles = np.where(intensityMap == 8)
    # print(Vehicles)
    # semanticMap[1, Vehicles] = 1
    # print(semanticMap[0,300])
    # c = np.where(semanticMap[0] == 1)
    # print(c)


    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    if not pure:
        # print('pure not')
        if Ultimate:
            RGB_Map = np.zeros((5, Height - 1, Width - 1))
            RGB_Map[0, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
            RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
            RGB_Map[3, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
            RGB_Map[4, :, :] = semanticMapP[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
            RGB_Map[2, :, :] = semanticMapV[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
            return RGB_Map
        RGB_Map = np.zeros((4, Height - 1, Width - 1))
        RGB_Map[0, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
        RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
        RGB_Map[3, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
        RGB_Map[2, :, :] = semanticMapP[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
        # RGB_Map[2, :, :] = semanticMapV[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
        # RGB_Map[3, :, :] = semanticMapP[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
        return RGB_Map
    else:
        # print('pure')
        RGB_Map = np.zeros((3, Height - 1, Width - 1))
        RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
        RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
        # RGB_Map[2, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]*30  # b_map
        # RGB_Map[1, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]*20  # b_map
        # RGB_Map[2, :, :] = semanticMapP[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map
        # RGB_Map[1, :, :] = semanticMapV[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map
        RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map
        # print(intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH])
        # print(RGB_Map)
        # print(RGB_Map.shape)
        return RGB_Map


# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners

import copy
def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    # if Ultimate:
    #     img = np.array(img)
    #     new_img = copy.deepcopy(img[:,:,:4])
    #
    #     # print(new_img.shape)
    #     cv2.polylines(new_img, [corners_int], True, color, 1)
    #     corners_int = bev_corners.reshape(-1, 2)
    #     # print(corners_int)
    #     cv2.line(new_img, (int(round(corners_int[0, 0])), int(round(corners_int[0, 1]))), (int(round(corners_int[3, 0])), int(round(corners_int[3, 1]))), (255, 255, 0), 1)
    # else:
    cv2.polylines(img, [corners_int], True, color, 1)
    corners_int = bev_corners.reshape(-1, 2)
    # print(corners_int)
    cv2.line(img, (int(round(corners_int[0, 0])), int(round(corners_int[0, 1]))),
             (int(round(corners_int[3, 0])), int(round(corners_int[3, 1]))), (255, 255, 0), 1)
