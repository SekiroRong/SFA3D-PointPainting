# -*- coding = utf-8 -*-
# @Time : 2022/4/7 19:24
# @Author : 戎昱
# @File : demoPlot.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
from matplotlib import pyplot as plt
Ppa = [0.109905,0.119212,0.137328,0.149663,0.17425,0.199377,0.205914,0.206477]
mpa = [0.544187,0.548442,0.558211,0.559547,0.567749,0.577594,0.577998,0.57996,0.6]

mAP = [79.56,79.38,80.18,80.09,81.11,81.67,82.26,82.28,86.63]
plt.ylabel("mAP of SFA-3D-PointPainting")
plt.xlabel("mPA of BiseNetV2")
plt.plot(mpa,mAP)
plt.plot(mpa,mAP, 'o')
# plt.show()
plt.savefig('output.jpg')