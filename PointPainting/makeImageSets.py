# -*- coding = utf-8 -*-
# @Time : 2022/1/29 13:01
# @Author : 戎昱
# @File : makeImageSets.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import os
video_path = r'G:\PP\carla\testing'

images_dir = os.path.join(video_path, 'image_2')
# txt_dir = r'G:\PP\carla\ImageSets\train.txt'
txt_dir2 = r'G:\PP\carla\ImageSets\test.txt'
# txt_dir3 = r'G:\PP\carla\ImageSets\trainval.txt'
# txt_dir4 = r'G:\PP\carla\ImageSets\val.txt'

images_filenames = sorted(
    [os.path.join(images_dir, filename) for filename in os.listdir(images_dir)]
)
with open(txt_dir2,'w') as f:
    for img in images_filenames:
        img_name = img[28:].strip('.jpg')
        print(img_name)
        f.write(img_name)
        f.write('\n')

# with open(txt_dir2,'w') as f:
#     for img in images_filenames:
#         img_name = img[29:].strip('.jpg')
#         print(img_name)
#         f.write(img_name)
#         f.write('\n')
#
# with open(txt_dir3,'w') as f:
#     for img in images_filenames:
#         img_name = img[29:].strip('.jpg')
#         print(img_name)
#         f.write(img_name)
#         f.write('\n')
#
# with open(txt_dir4,'w') as f:
#     for img in images_filenames:
#         img_name = img[29:].strip('.jpg')
#         print(img_name)
#         f.write(img_name)
#         f.write('\n')

