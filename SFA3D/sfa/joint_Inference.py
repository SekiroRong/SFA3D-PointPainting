# -*- coding = utf-8 -*-
# @Time : 2022/2/3 15:26
# @Author : 戎昱
# @File : joint_Inference.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import argparse
import copy
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration
from data_process.kitti_bev_utils import drawRotatedBox
from data_process import transformation

from utils.config import PP, step, project_semantic

pointpainting_dir = 'L:\sekiro\PointPainting-master'
sys.path.append(pointpainting_dir)

# print(sys.path)
from KittiCalibration import KittiCalibration
from KittiVideo import KittiVideo,output_gt
from visualizer import Visualizer
from BiSeNetv2.model.BiseNetv2 import BiSeNetV2
from BiSeNetv2.utils.utils import preprocessing_kitti, postprocessing
from pointpainting import PointPainter

from data_process.kitti_dataset import get_filtered_lidar, makeBEVMap

save_result = True # 保存检测结果

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default=r'G:\PP\carla\checkpoints\Ultimate/Model_Ultimate_epoch_34_0.693_0.753.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='video', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    # if PP:
    #     configs.num_classes = 2
    # else:
    #     configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    if PP:
        configs.root_dir = 'G:\PP\carla'
        # configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')
        configs.dataset_dir = 'G:\PP\carla'
    else:
        configs.root_dir = 'G:\kitti'
        configs.dataset_dir = 'G:\kitti'

    if True:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    # Initialize

    # -----------------------------------------------------------------------
    # SFA3D Part
    configs = parse_test_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')

    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cuda'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # dev = "cpu"
    print(dev)
    device = torch.device(dev)

    configs.device = device
    print('model to device')
    # print('device:' + )
    model = model.to(device=configs.device)

    out_cap = None

    print('model.eval()')

    model.eval()

    print('Create Dataloader')

    test_dataloader = create_test_dataloader(configs)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # PointPainting Part
    # Semantic Segmentation
    bisenetv2 = BiSeNetV2(n_classes=19)
    checkpoint = torch.load(r'G:\PP\pth\weights_epoch_295_tl_0.93_vl_0.087.pth', map_location=dev)
    bisenetv2.load_state_dict(checkpoint['bisenetv2'], strict=False)
    bisenetv2.eval()
    bisenetv2.to(device)

    # Fusion
    painter = PointPainter()

    video = KittiVideo(
        video_root=r'G:\PP\carla\testing',
        calib_root=r'G:\PP\carla\training\modified_calib_file.txt'
    )

    visualizer = Visualizer('img')

    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Inference
    avg_time = 0

    label_dir = os.path.join(r'G:\PP\carla\testing', 'label_2')

    label_filenames = sorted(
        [filename for filename in os.listdir(label_dir)]
    )

    # print(label_filenames)

    for i in range(len(video)):
        t0 = time.time()
        t1 = time.time()

        # -----------------------------------------------------------------------
        # PointPainting Part
        print('output_gt',output_gt)
        if output_gt:
            image, pointcloud, calib, semantic = video[i]
        else:
            image, pointcloud, calib = video[i]

        t8 = time.time()
        # print("get_data cost ", (t8 - t1))
        image2 = copy.deepcopy(image)
        # print(image2.shape)
        # t8 = time.time()
        if not output_gt:
            input_image = preprocessing_kitti(image)

            t4 = time.time()
            # print("preprocessing_kitti cost ", (t4 - t8))

            semantic = bisenetv2(input_image)

            t5 = time.time()
            # print("bisenetv2 cost ", (t5 - t4))

            semantic = postprocessing(semantic)

        t6 = time.time()
        # print("postprocessing cost ", (t6 - t5))

        painted_pointcloud = painter.paint(pointcloud, semantic, calib)
        # print(painted_pointcloud)
        t7 = time.time()
        # print("painted_pointcloud cost ", (t7 - t6))

        lidar = painted_pointcloud[:, [2, 0, 1, 3, 4]]
        # print(lidar)

        t2 = time.time()
        pp_time = t2 - t1
        # print("PointPainting cost ", (pp_time))


        if project_semantic:
            t9 = time.time()
            image2 = visualizer.get_colored_image(image, semantic)
            image2 = cv2.resize(image2, (1280,720))
            # print(image2.shape)
            print("project_semantic cost {:.2f} s".format(t9 - t2))
        # -----------------------------------------------------------------------

        # -----------------------------------------------------------------------
        # SFA3D Part
        with torch.no_grad():
            img_rgb = image2
            # img_rgb = semantic
            lidarData = get_filtered_lidar(lidar, cnf.boundary)
            bev_map = makeBEVMap(lidarData, cnf.boundary, mode='test')
            # print(bev_map.shape)
            bev_maps = bev_map[None, :, :, :]
            # print(bev_maps.shape)
            bev_maps = torch.from_numpy(bev_map)

            input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
            input_bev_maps = input_bev_maps.unsqueeze(0)
            # print(input_bev_maps.size())

            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            sfa_time = t2 - t1
            print("SFA3D cost {:.2f} s".format(sfa_time))

            detections = detections[0]  # only first batch
            # Draw prediction in the image
            bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))

            bev_map = copy.deepcopy(bev_map[:, :, :4])

            result_filename = os.path.join(r'G:\PP\carla\testing', 'result', label_filenames[i])
            # print(result_filename)
            bev_map = draw_predictions(bev_map, detections.copy(), configs.num_classes, save_result=save_result, path=result_filename)

            label_filename = os.path.join(r'G:\PP\carla\testing', 'label_2', label_filenames[i])

            labels = []
            for line in open(label_filename, 'r'):
                line = line.strip('\n')
                line_parts = line.split(' ')
                cat_id = int(line_parts[0])
                x, y, z = float(line_parts[1]), float(line_parts[2]), float(line_parts[3])
                h, w, l = float(line_parts[4]), float(line_parts[5]), float(line_parts[6])
                ry = -float(line_parts[7])

                object_label = [cat_id, x, -y, z, h, w, l, ry]
                labels.append(object_label)

            if len(labels) == 0:
                labels = np.zeros((1, 8), dtype=np.float32)
                has_labels = False
            else:
                labels = np.array(labels, dtype=np.float32)
                has_labels = True

            calib = Calibration(r'G:\PP\carla\training\modified_calib_file.txt')
            if has_labels:
                labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)

            for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels):
                # Draw rotated box
                yaw = -yaw
                y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
                x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
                # x1 = int((cnf.boundary['maxY'] - y) / cnf.DISCRETIZATION)
                w1 = int(w / cnf.DISCRETIZATION)
                l1 = int(l / cnf.DISCRETIZATION)

                # print(x1, y1, w1, l1, yaw)
                # drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, [255,255,255]) # GT

            # print(detections)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            # print(img_rgbs.shape)
            img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
            # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            calib = Calibration(r'G:\PP\carla\training\modified_calib_file.txt')

            kitti_dets = convert_det_to_real_values(detections)
            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                img_rgb = show_rgb_image_with_boxes(img_rgb, kitti_dets, calib)

            # print(img_bgr.shape)
            out_img = merge_rgb_to_bev(img_rgb, bev_map[:,:,:3], output_width=configs.output_width)

            t3 = time.time()

            print("Total cost {:.2f} s\n".format(t2-t8))

            # cv2.putText(out_img, "FPS:{:.2f}".format(1/(t2-t8)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 1)

            # print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
            #                                                                                1 / (t2 - t1)))
            if out_cap is None:
                out_cap_h, out_cap_w = out_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                out_cap = cv2.VideoWriter(
                    os.path.join(configs.results_dir, 'Joint_Inference.mp4'),
                    fourcc, 10, (out_cap_w, out_cap_h))

            out_cap.write(out_img)


            if step:
                cv2.imshow('test-img', out_img)
                print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                if cv2.waitKey(0) & 0xFF == 27:
                    break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()