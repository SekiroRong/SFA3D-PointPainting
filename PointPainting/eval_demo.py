# -*- coding = utf-8 -*-
# @Time : 2022/4/3 15:36
# @Author : 戎昱
# @File : eval_demo.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import cv2
from BiSeNetv2.utils.label import labels, id2label
__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""
def remove_ignore_index_labels(semantic):
    for id in id2label:
        label = id2label[id]
        trainId = label.trainId
        # print(trainId)
        semantic[semantic == id] = trainId
    return semantic

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def get_eval(l_path, p_path):
    imgPredict = cv2.imread(p_path, cv2.IMREAD_COLOR)
    imgPredict = cv2.resize(imgPredict, (1280, 720))
    imgLabel = cv2.imread(l_path, cv2.IMREAD_COLOR)
    imgLabel = remove_ignore_index_labels(imgLabel)
    imgLabel = np.array(imgLabel)
    imgPredict = np.array(imgPredict)
    # cv2.imshow("imgLabel", imgLabel)
    # cv2.imshow("imgPredict", imgPredict)
    # cv2.waitKey(0)
    # print(imgLabel[imgLabel == 255])
    # print('----')
    # imgPredict[imgLabel == 255] = imgLabel[imgLabel == 255]
    # print(imgPredict[imgLabel == 255])
    # while 1:
    #     a = 1
    metric = SegmentationMetric(16)  # 3表示有3个分类，有几个分类就填几
    metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    # print('pa is : %f' % pa)
    # print('cpa is :')  # 列表
    # print(cpa)
    cpa = np.nan_to_num(cpa)
    Ppa, Vpa = cpa[2],cpa[8]
    # print(Ppa)
    # if Ppa == np.nan:
    #     print(Ppa)
    # print(Ppa, Vpa, cpa[15])
    # print('mpa is : %f' % mpa)
    # print('mIoU is : %f' % mIoU)
    return pa, Ppa, Vpa, mpa, mIoU

import os
from tqdm import tqdm
labels_path = r'G:\PP\carla\testing\semantic'
pred_path = r'G:\PP\carla\testing\semantic_result'

labels_filenames = sorted(
    [os.path.join(labels_path,filename) for filename in os.listdir(labels_path)]
)
pred_filenames = (
    [os.path.join(pred_path,filename) for filename in os.listdir(pred_path)]
)
assert len(labels_filenames) == len(pred_filenames)
print('There are ' + str(len(labels_filenames)) + ' predictions')

if __name__ == '__main__':
    pa_sum, mpa_sum, mIoU_sum, Ppa_sum, Vpa_sum = 0, 0, 0, 0, 0
    P_skip, V_skip = 0, 0
    for i in tqdm(range(len(labels_filenames))):
        # print(labels_filenames[i], pred_filenames[i])
        pa, Ppa, Vpa, mpa, mIoU = get_eval(labels_filenames[i], pred_filenames[i])
        pa_sum += pa
        mpa_sum += mpa
        mIoU_sum += mIoU
        Ppa_sum += Ppa
        Vpa_sum += Vpa
        if Ppa < 0.0001:
            P_skip += 1
        if Vpa < 0.0001:
            V_skip += 1
        # print(Ppa_sum, Vpa_sum)

    pa_sum /= len(labels_filenames)
    mpa_sum /= len(labels_filenames)
    mIoU_sum /= len(labels_filenames)
    Ppa_sum /= (len(labels_filenames)-P_skip)
    Vpa_sum /= (len(labels_filenames)-V_skip)
    print(P_skip, V_skip)
    print('pa is : %f' % pa_sum)
    print('Ppa is : %f' % Ppa_sum)
    print('Vpa is : %f' % Vpa_sum)
    # print('cpa is :')  # 列表
    # print(cpa)
    print('mpa is : %f' % mpa_sum)
    print('mIoU is : %f' % mIoU_sum)