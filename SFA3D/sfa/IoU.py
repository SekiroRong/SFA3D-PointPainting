# -*- coding = utf-8 -*-
# @Time : 2022/2/10 19:15
# @Author : 戎昱
# @File : IoU.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import cv2
import numpy as np
import math
import time

convert = 180/math.pi
"""
计算旋转面积
boxes1,boxes2格式为x,y,w,h,theta(360°制)
"""
def iou_rotate_calculate(boxes1, boxes2):
    # print("####boxes2:", boxes1.shape)
    # print("####boxes2:", boxes2.shape)

    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    # print(area1,area2)
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4]*convert)
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4]*convert+90)
    # print(r1, r2)
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        # 计算出iou
        ious = int_area * 1.0 / (area1 + area2 - int_area)
    else:
        ious=0

    # print(' ')
    #
    # print(boxes2[2])
    # if boxes2[2] < 0.5: # pedestrain
    #     print(ious)
    # print(r1, r2, ious)
    return ious

# bbox1 = np.array([1.0, 1.0, 2, 4, 0])
# bbox2 = np.array([1.0, 1.0, 2, 4, 1.5])

# t0 = time.time()
# print(iou_rotate_calculate(bbox1, bbox2))
# t1 = time.time()
# print(t1-t1)

