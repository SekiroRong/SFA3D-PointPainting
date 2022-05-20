import os
import cv2
import numpy as np
import argparse
import time
import torch

from KittiCalibration import KittiCalibration
from KittiVideo import KittiVideo, output_gt
from visualizer import Visualizer
from BiSeNetv2.model.BiseNetv2 import BiSeNetV2
from BiSeNetv2.utils.utils import preprocessing_kitti, postprocessing
from pointpainting import PointPainter

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

savePP = False

makeVideo = True

root_dir = r'G:\PP\carla\testing'
pp_Path = os.path.join(root_dir,'paintedVelodyne')

def pause():
    while 1:
        a = 1
def savePaintedPoint(Points,path):
    lidar = np.array(Points).astype(np.float32)
    # print(lidar[0])
    lidar = lidar[:, [2, 0, 1, 3, 4]]
    lidar.tofile(path)

def main(args):
    # Semantic Segmentation
    bisenetv2 = BiSeNetV2(n_classes=19)
    checkpoint = torch.load(args.weights_path, map_location=dev)
    bisenetv2.load_state_dict(checkpoint['bisenetv2'], strict=False)
    bisenetv2.eval()
    bisenetv2.to(device)

    # Fusion
    painter = PointPainter()

    video = KittiVideo(
        video_root=args.video_path,
        calib_root=args.calib_path
    )

    images_dir = os.path.join(args.video_path, 'image_2')

    images_filenames = sorted(
        [os.path.join(images_dir, filename) for filename in os.listdir(images_dir)]
    )

    visualizer = Visualizer(args.mode)

    frames = []
    if args.mode == '3d':
        frame_shape = (1280, 720)
    else:
        frame_shape = (750, 900)
    avg_time = 0

    # start = 4752
    for i in range(len(video)):
        # j = i + start
        t1 = time.time()
        if output_gt:
            image, pointcloud, calib, semantic = video[i]
        else:
            image, pointcloud, calib = video[i]
        t2 = time.time()
        # print(image.shape, pointcloud.shape)

        if not output_gt:
            input_image = preprocessing_kitti(image)
            t3 = time.time()
            semantic = bisenetv2(input_image)
            t4 = time.time()
            # print(semantic.shape)

            semantic = postprocessing(semantic)
            img_path = r'G:\PP\carla\testing\semantic_result'
            img_path = os.path.join(img_path, str(i).rjust(4,'0') + '.jpg')
            cv2.imwrite(img_path, semantic)
        # print(semantic)
        t5 = time.time()

        painted_pointcloud = painter.paint(pointcloud, semantic, calib)
        # print(painted_pointcloud)
        t6 = time.time()

        # print('get_data', t2 - t1)
        # print('preprocessing_kitti', t3 - t2)
        # print('bisenetv2', t4 - t3)
        # print('postprocessing', t5 - t4)
        # print('paint', t6 - t5)
        # print(painted_pointcloud)

        # print(painted_pointcloud.shape)
        if savePP:
            if root_dir == r'G:\PP\carla\testing':
                start = 28
            else:
                start = 29
            save_PPpath = pp_Path + '/' + images_filenames[i][start:].replace('jpg','bin')
            # print(save_PPpath)
            # pause()
            savePaintedPoint(painted_pointcloud,save_PPpath)


        if makeVideo:
            if args.mode == '3d':
                screenshot = visualizer.visuallize_pointcloud(painted_pointcloud, blocking=False)
                print(screenshot.shape)
                frames.append(screenshot)
            else:
                color_image = visualizer.get_colored_image(image, semantic)
                if args.mode == 'img':
                    frames.append(color_image)
                    # cv2.imshow('color_image', color_image)
                elif args.mode == '2d':
                    scene_2D = visualizer.get_scene_2D(color_image, painted_pointcloud, calib)
                    frames.append(scene_2D)
                    # cv2.imshow('scene', scene_2D)
                # if cv2.waitKey(0) == 27:
                #     cv2.destroyAllWindows()
                #     break
            # if i == 20:
            #     break

        avg_time += (time.time()-t1)
        print(f'{i} sample')

    # Time & FPS
    avg_time /= len(video)
    FPS = 1 / avg_time
    print("Average Time",round(avg_time*1000,2), "ms  FPS", round(FPS,2))
    if makeVideo:
        # Save Video
        save_path = os.path.join(args.save_path, args.mode + '_demo.mp4')
        out_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, frame_shape)
        for idx,frame in enumerate(frames):
            frame = cv2.resize(frame, frame_shape)
            # img_path = r'G:\PP\carla\testing\semantic_result'
            # img_path = os.path.join(img_path, str(idx) + '.jpg')
            # cv2.imwrite(img_path,frame)
            out_video.write(frame)
        print(f'Saved Video @ {save_path}')


def boundary_3d_modify():
    from bev_utils import boundary
    boundary['minY'] = -30
    boundary['maxY'] = 30
    boundary['maxX'] = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str,
    default=root_dir,)
    # default='Videos/1/2011_09_26/2011_09_26_drive_0048_sync',)
    parser.add_argument('--calib_path', type=str,
    default=r'G:\PP\carla\training\modified_calib_file.txt',)
    # default='Videos/1/2011_09_26_calib/2011_09_26',)
    parser.add_argument('--weights_path', type=str, default=r'G:\PP\pth\weights_epoch_70_tl_1.208_vl_0.114.pth',)
    parser.add_argument('--save_path', type=str, default=r'G:\PP\val_videos',)
    parser.add_argument('--mode', type=str, default='img', choices=['img', '2d', '3d'],
    help='visualization mode .. img is semantic image .. 2d is semantic + bev .. 3d is colored pointcloud')
    args = parser.parse_args()
    if args.mode == '3d':
        boundary_3d_modify()

    main(args)



