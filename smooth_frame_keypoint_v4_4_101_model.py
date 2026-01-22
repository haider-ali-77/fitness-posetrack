#!/home/ubuntu/anaconda3/bin/python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################


"""

Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.

"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
import subprocess
import matplotlib.pyplot as plt
from datetime import timedelta

from tqdm import tqdm

import keypoints_post_processing
import scipy
from scipy.interpolate import splprep,splev
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
from collections import defaultdict
import argparse
import cv2 as cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import traceback
import logging
import os
import sys
import time
import json
import math
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from math import hypot
from caffe2.python import workspace
from kalman import KalmanFilter as kf
# from pose_track import Posetracker as pt
from datetime import datetime
from scipy.signal import savgol_filter

##

import pickle
from densepose.data.structures import DensePoseResult
from sklearn import preprocessing


# preprocess data
scaler_file = 'min_max_scaler_5classes.sav'
min_max_scaler = pickle.load(open(scaler_file, 'rb'))
# min_max_scaler = preprocessing.MinMaxScaler()

# loading pickle model and predicting foot on ground
pkl_filename = "pickle_model_series_5classes.pkl"
with open(pkl_filename, 'rb') as file:
    rf = pickle.load(file)
##

def function(img_i, imgu, imgv, foot=''):
    img_u = imgu.copy()
    img_v = imgv.copy()

    if (foot == 'left'):  # left foot
        # right pose
        # i = 5
        # u = 71
        # v = 40

        # left pose
        # i = 5
        # u = 21
        # v = 222

        # back pose
        i = 5
        u = 84#53#56
        v = 96#103#139

        # front pose
        # i = 5
        # u = 203
        # v = 201
    elif (foot == 'right'):  # right foot
        # right pose
        # i = 6
        # u = 236
        # v = 235

        # left pose
        # i = 6
        # u = 164
        # v = 71

        # back pose
        i = 6
        u = 161#237#235#209
        v = 128#168#170#95

        # front pose
        # i = 6
        # u = 46
        # v = 193
    else:
        print('no foot')

    #     print (i,u,v)

    map_ = (img_i == i)

    img_u[~map_] = 0
    img_v[~map_] = 0
    img_u = img_u.astype(np.int32)
    img_v = img_v.astype(np.int32)
    dist_img = (u - img_u) ** 2 + (v - img_v) ** 2

    min_ = np.min(dist_img)
    # print(min_)
    min_indices = np.where(dist_img == min_)
    return map_, dist_img, min_indices


# dist_img_left = function(img2_i, img2_u, img2_v, foot = 'left')
# dist_img_right = function(img2_i, img2_u, img2_v, foot = 'right')

def iuv(img_name):

    dirs = os.listdir('./results')

    # f = open('./results/'+dirs[0], 'rb')
    f = open('./results/'+img_name+'.pkl', 'rb')
    data = pickle.load(f)
    # print(len(data))

    # for i in range(0, len(data)):
    # print ('data = ',data)
    i = 0
    img_id = i  # Look at the first image and the first detected instance
    areas = []
    for j in range(0, len(data[i]['pred_densepose'])):
        bbox_xyxy = data[img_id]['pred_boxes_XYXY'][j]
        # print ('len data[i] ',len(data[i]['pred_densepose']))
        # print ('j = ',j, bbox_xyxy)
        area = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
        areas.append(area)

        # print('area ', area)
        # print('j =', bbox_xyxy.numpy()), bbox_xyxy[0], bbox_xyxy[1]
    instance_id = np.argmax(areas)
    # print('areas, instance_id = ', areas, instance_id)
    bbox_xyxy = data[img_id]['pred_boxes_XYXY'][instance_id]
    result_encoded = data[img_id]['pred_densepose'].results[instance_id]

    # result_encoded[0] = ()
    # print ('result_encoded = ',result_encoded[0])
    iuv_arr = DensePoseResult.decode_png_data(*result_encoded)

    # img_id, instance_id = 0, 0  # Look at the first image and the first detected instance
    # bbox_xyxy = data[img_id]['pred_boxes_XYXY'][instance_id]
    # result_encoded = data[img_id]['pred_densepose'].results[instance_id]
    # iuv_arr = DensePoseResult.decode_png_data(*result_encoded)

    img_i = iuv_arr[0,:,:]
    img_u = iuv_arr[1,:,:]
    img_v = iuv_arr[2,:,:]

    map_l, dist_img_left, min_indcs_l = function(img_i, img_u, img_v, foot='left')
    map_r, dist_img_right, min_indcs_r = function(img_i, img_u, img_v, foot='right')

    return map_l, dist_img_left, min_indcs_l, map_r, dist_img_right, min_indcs_r, bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]
    # min_indcs_l[0][0]][min_indcs_l[1][0]

    ##

# from detectron2.core.config import assert_and_infer_cfg
# from detectron2.core.config import cfg
# from detectron2.core.config import merge_cfg_from_file
# from detectron2.utils.io import cache_url
# from detectron2.utils.logging import setup_logging
# from detectron2.utils.timer import Timer
# import detectron2.core.test_engine as infer_engine
# import detectron2.datasets.dummy_datasets as dummy_datasets
# import detectron2.utils.c2 as c2_utils
# import detectron2.utils.vis as vis_utils

# c2_utils.import_detectron_ops()

# # OpenCL may be enabled by default in OpenCV3; disable it because it's not
# # thread safe and causes unwanted GPU memory allocations.
# cv2.ocl.setUseOpenCL(False)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


# import some common libraries
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--output-option',
        dest='output_option',
        help='output directoty structure format',
        default=None,
        type=str
    )
    parser.add_argument(
        '--distance',
        dest='distance_cam',
        help='Distance from camera to person',
        default=None,
        type=int
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: png)',
        default='png',
        type=str
    )

    parser.add_argument(
        '--input_type',
        dest='input_type',
        help='check whether video available or image',
        default='image',
        type=str
    )

    parser.add_argument(
        '--input_fps',
        dest='input_fps',
        help='frame rate of video',
        default='60',
        type=str
    )

    parser.add_argument(
        '--video_type',
        dest='video_type',
        help='type of video on which you want to run inference',
        default='mov',
        type=str
    )

    parser.add_argument(
        '--video_path',
        dest='video_path',
        help='full path for video',
        default='0',
        type=str
    )

    parser.add_argument(
        '--raw_video',
        dest='raw_video',
        help='set raw_video = raw for better quality of video and raw_video = normal for compact video',
        default='normal',
        type=str
    )

    parser.add_argument(
        '--cam-distance',
        dest='cam_dist',
        help='Distance of personin feet from camera',
        default='7',
        type=str
    )

    parser.add_argument(
        '--max-frame',
        dest='max_frame',
        help='Max frame to process',
        default='2000',
        type=str
    )

    parser.add_argument(
        '--pelvis',
        dest='pelvis',
        help='Set Pelvis on or off',
        default='off',
        type=str
    )

    parser.add_argument(
        '--pose_detection',
        dest='pose_detection',
        help='Set pose_detection on or off',
        default='off',
        type=str
    )
    parser.add_argument('--line_thickness', dest='line_thickness', help='set line thickness',
                        default=2, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

# functions to generate angles


def calc_angle(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return np.degrees(math.pi / 2)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# Function to verify varous angle are correct or not
def verify_trunck_knee_femur_hipAnkleDist(right_shoulder, right_hip, right_knee, right_ankle, left_shoulder, left_hip, left_knee, left_ankle, max_trunk_ang, max_knee_ang, max_femur_ang, hipAnkle_dist_param, view):
    left_knee_ang = 180 - \
        calc_angle(np.array(left_hip), np.array(
            left_knee), np.array(left_ankle))
    right_knee_ang = 180 - calc_angle(np.array(right_hip),
                                      np.array(right_knee), np.array(right_ankle))

    hip_mean = []
    hip_mean.append((right_hip[0] + left_hip[0]) / 2)
    hip_mean.append((right_hip[1] + left_hip[1]) / 2)
    shoulder_mean = []
    shoulder_mean.append((right_shoulder[0] + left_shoulder[0]) / 2)
    shoulder_mean.append((right_shoulder[1] + left_shoulder[1]) / 2)
    vertical_coord = []
    vertical_coord.append(hip_mean[0])
    vertical_coord.append(shoulder_mean[1])
    trunk_ang = calc_angle(np.array(shoulder_mean), np.array(
        hip_mean), np.array(vertical_coord))

    left_knee_vertical = []
    left_knee_vertical.append(left_hip[0])
    left_knee_vertical.append(left_knee[1])
    left_femur_ang = calc_angle(np.array(left_knee), np.array(left_hip),
                                np.array(left_knee_vertical))

    right_knee_vertical = []
    right_knee_vertical.append(right_hip[0])
    right_knee_vertical.append(right_knee[1])
    right_femur_ang = calc_angle(np.array(right_knee), np.array(
        right_hip), np.array(right_knee_vertical))

    left_ankle_hip_dist = hip_mean[0] - left_ankle[0]
    right_ankle_hip_dist = hip_mean[0] - right_ankle[0]
    left_hip_mean_dist = (
        abs(left_ankle[0] - hip_mean[0])) * hipAnkle_dist_param
    right_hip_mean_dist = (
        abs(right_ankle[0] - hip_mean[0])) * hipAnkle_dist_param

    if left_knee_ang < 50 and right_knee_ang < 50 and trunk_ang < 30 and left_femur_ang < 17 and right_femur_ang < 17:
        if view == 'back_side':
            if left_ankle_hip_dist > 0 and right_ankle_hip_dist < 0:
                return "true"
            elif left_ankle_hip_dist <= 0 and right_ankle_hip_dist >= 0 and abs(left_ankle_hip_dist) < right_hip_mean_dist and abs(right_ankle_hip_dist) < left_hip_mean_dist:
                return "true"
            else:
                return "false"

        elif view == 'front_side':
            if left_ankle_hip_dist < 0 and right_ankle_hip_dist > 0:
                return "true"
            elif left_ankle_hip_dist >= 0 and right_ankle_hip_dist <= 0 and abs(left_ankle_hip_dist) < right_hip_mean_dist and abs(right_ankle_hip_dist) < left_hip_mean_dist:
                return "true"
            else:
                return "false"
        else:
            return "false"
    else:
        return "false"


# Determine whether knee is on left side or right side.
def knee_point_dir_hip_ankle_line(hip, knee, ankle):
    dist_knee = (knee[0] - hip[0]) * (ankle[1] - hip[1]) - \
        (knee[1] - hip[1]) * (ankle[0] - hip[0])
    left_ptx = ankle[0] - 1
    left_pty = ankle[1]
    dist_left_pt = (left_ptx - hip[0]) * (ankle[1] - hip[1]) - \
        (left_pty - hip[1]) * (ankle[0] - hip[0])
    if dist_left_pt * dist_knee >= 0:
        return -1
    else:
        return 1


# Calculate the eqclidean distance between two point
def calculate_euclidean_distance(pointA, pointB):
    dist_in_x = pointA[0] - pointB[0]
    dist_in_y = pointA[1] - pointB[1]
    distance = math.sqrt(dist_in_x * dist_in_x + dist_in_y * dist_in_y)
    return distance


def calculate_distance_list(listA, listB, dist_dir):
    distance_listA_listB = list()
    for i in range(0, len(listA)):
        if dist_dir == 'x':
            distance_listA_listB.append(abs(listA[i][0] - listB[i][0]))
        elif dist_dir == 'y':
            distance_listA_listB.append(abs(listA[i][1] - listB[i][1]))
        else:  # dist_dir == 'y':
            distance_listA_listB.append(
                calculate_euclidean_distance(listA[i], listB[i]))
    return distance_listA_listB


def check_valid_ankle_point(knee, ankle, frame5_dist):

    mean_5frame_distance = (frame5_dist[0] + frame5_dist[1]
                            + frame5_dist[2] + frame5_dist[3] + frame5_dist[4]) / 5
    knee_ankle_distance = calculate_euclidean_distance(knee, ankle)
    if knee_ankle_distance >= 0.5 * mean_5frame_distance and knee_ankle_distance <= 1.5 * mean_5frame_distance:
        return 'valid'
    else:
        return 'notvalid'


def save_optimal_frame(image_path, org_img_name, rename_img):
    input_image = os.path.join(image_path, org_img_name)
    output_image = os.path.join(image_path, rename_img + org_img_name)
    copy_image = 'cp ' + input_image + ' ' + output_image
    os.system(copy_image)


def front_back_optimal_keypoints(right_knee_leg_angle, left_knee_leg_angle, right_knee_hip_vertical, left_knee_hip_vertical, left_hip_right_hip, trunk_lean, optimal_point, img_name, optimal_image_name, img_ext, pelvis_status, alignment, cross_over_sign='', angle=-1):

    # print(optimal_point, img_name)
    seq_info = {}
    keypoint_info = {}
    keypoint_info['right_knee_varus_valgus'] = round(
        right_knee_leg_angle[optimal_point], 1)

    keypoint_info['left_knee_varus_valgus'] = round(
        left_knee_leg_angle[optimal_point], 1)

    if cross_over_sign == 'left_knee':
        if left_knee_leg_angle[optimal_point] <=-5:
            keypoint_info['left_knee_varus_valgus'] = 2
        elif left_knee_leg_angle[optimal_point] <-10:
            keypoint_info['left_knee_varus_valgus'] = 1
        else:
            keypoint_info['left_knee_varus_valgus'] = 0

    if cross_over_sign == 'right_knee':
        if right_knee_leg_angle[optimal_point] <= -5:
            keypoint_info['right_knee_varus_valgus'] = 2
        elif right_knee_leg_angle[optimal_point] < -10:
            keypoint_info['right_knee_varus_valgus'] = 1
        else:
            keypoint_info['right_knee_varus_valgus'] = 0

    keypoint_info['right_femur_angle'] = round(
        right_knee_hip_vertical[optimal_point], 1)

    if cross_over_sign == 'right_femour':
        if right_knee_hip_vertical[optimal_point] <=-5:
            keypoint_info['right_femur_angle'] = 2
        elif right_knee_hip_vertical[optimal_point] <-10:
            keypoint_info['right_femur_angle'] = 1
        else:
            keypoint_info['right_femur_angle'] = 0

    keypoint_info['left_femur_angle'] = round(
        left_knee_hip_vertical[optimal_point], 1)

    if cross_over_sign == 'left_femour':
        if left_knee_hip_vertical[optimal_point] <=-5:
            keypoint_info['left_femur_angle'] = 2
        elif left_knee_hip_vertical[optimal_point] <-10:
            keypoint_info['left_femur_angle'] = 1
        else:
            keypoint_info['left_femur_angle'] = 0

    keypoint_info['trunk_lateral_lean'] = round(trunk_lean[optimal_point], 1)
    if cross_over_sign == 'trunk':
        if trunk_lean[optimal_point] <= -5:
            keypoint_info['trunk_lateral_lean'] = 2
        elif trunk_lean[optimal_point] <= -10:
            keypoint_info['trunk_lateral_lean'] = 1
        else:
            keypoint_info['trunk_lateral_lean'] = 0

    if pelvis_status == 'on':
        keypoint_info['pelvis_angle'] = round(
            left_hip_right_hip[optimal_point], 1)


    seq_info['frame_number'] = optimal_image_name + '_' + str(img_name)
    #seq_info['frame_number'] =  str(optimal_point) + '_' + optimal_image_name
    seq_info['keyangle'] = keypoint_info
    seq_info['pose'] = alignment
    if cross_over_sign == 'Yes':
        seq_info['Cross-over-sign'] = cross_over_sign
    if cross_over_sign == 'No':
        seq_info['Cross-over-sign'] = cross_over_sign
    if angle >= 0:
        seq_info['Pelvis-angle'] = angle
    return seq_info

def back_chest(name,distance_chest_keypoints):
    seq_info = {}
    if len(distance_chest_keypoints) !=0:
        if distance_chest_keypoints[0] > 0:
            seq_info['Type'] = 'Max y-axis difference in chest keypoints'
            seq_info['chest_min_yaxis_frame'] = name+'min_value_'+distance_chest_keypoints[1]
            seq_info['chest_max_yaxis_frame'] = name+'max_value_'+distance_chest_keypoints[2]
            seq_info['distance'] = distance_chest_keypoints[0]
    return seq_info


def left_right_optimal_keypoints(left_hip_ankle_dist, left_knee_leg_angle, left_knee_ankle_vertical_angle, left_knee_hip_vertical, right_hip_ankle_dist, right_knee_leg_angle, right_knee_ankle_vertical_angle, right_knee_hip_vertical, trunk_lean, optimal_point, img_name, optimal_image_name, img_ext, alignment,sign=''):
    # print("angle is :", angle)
    seq_info = {}
    keypoint_info = {}
    keypoint_info['left_hip_ankle_dist'] = round(
        left_hip_ankle_dist[optimal_point], 1)
    keypoint_info['left_knee_angle'] = round(
        left_knee_leg_angle[optimal_point], 1)
    keypoint_info['left_tibial_angle'] = round(
        left_knee_ankle_vertical_angle[optimal_point], 1)
    keypoint_info['left_hip_angle'] = round(
        left_knee_hip_vertical[optimal_point], 1)
    keypoint_info['right_hip_ankle_dist'] = round(
        right_hip_ankle_dist[optimal_point], 1)
    keypoint_info['right_knee_angle'] = round(
        right_knee_leg_angle[optimal_point], 1)
    keypoint_info['right_tibial_angle'] = round(
        right_knee_ankle_vertical_angle[optimal_point], 1)
    keypoint_info['right_hip_angle'] = round(
        right_knee_hip_vertical[optimal_point], 1)
    keypoint_info['trunk_forward_lean'] = round(trunk_lean[optimal_point], 1)

    seq_info['frame_number'] = optimal_image_name + '_' + str(img_name)
    seq_info['keyangle'] = keypoint_info
    seq_info['pose'] = alignment
    if sign == 'Yes':
        seq_info['Knee-forward-of-toe'] = sign
    if sign == 'No':
        seq_info['Knee-forward-of-toe'] = sign
    # if angle != 0:
    #     seq_info['Side-pelvis-tilt'] = angle

    return seq_info


# Maximum angle between hip knee and ankle when foot is on ground
def optimal_frame_max_knee_flexion(all_hip_coord, all_knee_coord, all_ankle_coord, all_knee_leg_angle, all_knee_coord_anotherleg, all_ankle_coord_anotherleg, image_name_list, all_shoulder_coord, all_hip_ankle_distance_y):

    np_knee_another_knee_dist_x = np.array(calculate_distance_list(
        all_knee_coord, all_knee_coord_anotherleg, 'x'))
    sorted_ind_knee_another_knee_dist_x = np.argsort(
        np_knee_another_knee_dist_x)

    np_hip_ankle_distance_y = np.array(all_hip_ankle_distance_y)
    sorted_hip_ankle_dist_y = np.sort(np_hip_ankle_distance_y)
    sorted_hip_ankle_dist_y = sorted_hip_ankle_dist_y[::-1]

    np_ankle_knee_anotherleg_y = np.array(calculate_distance_list(
        all_knee_coord_anotherleg, all_ankle_coord_anotherleg, 'y'))
    sorted_np_ankle_knee_anotherleg_y = np.sort(np_ankle_knee_anotherleg_y)
    max_ankle_knee_anotherleg_y = max(sorted_np_ankle_knee_anotherleg_y)

    np_ankle_hip_x = np.array(calculate_distance_list(
        all_hip_coord, all_ankle_coord, 'x'))
    sorted_np_ankle_hip_x = np.sort(np_ankle_hip_x)

    optimal_point = 0
    optimal_image = ''
    ind = 0
    while optimal_image == '':
        frame_valid_knee_leg_angle = []
        valid_index = []
        for index in sorted_ind_knee_another_knee_dist_x:
            if all_hip_coord[index][1] < all_knee_coord[index][1] and all_knee_coord[index][1] < all_ankle_coord[index][1] and all_shoulder_coord[index][1] < all_hip_coord[index][1] and all_hip_coord[index][1] < all_knee_coord_anotherleg[index][1] and all_hip_coord[index][1] < all_ankle_coord_anotherleg[index][1] and abs(all_ankle_coord_anotherleg[index][1] - all_knee_coord_anotherleg[index][1]) <= sorted_np_ankle_knee_anotherleg_y[ind] + 40 and abs(all_hip_coord[index][0] - all_ankle_coord[index][0]) <= sorted_np_ankle_hip_x[ind] + 20 and abs(all_hip_coord[index][1] - all_ankle_coord[index][1]) >= sorted_hip_ankle_dist_y[ind] - 55 and abs(all_knee_coord[index][0] - all_knee_coord_anotherleg[index][0]) < np_knee_another_knee_dist_x[sorted_ind_knee_another_knee_dist_x[ind]] + 10:

                #optimal_point = index
                #optimal_image = os.path.basename(image_name_list[index])
                valid_index.append(index)
                frame_valid_knee_leg_angle.append(all_knee_leg_angle[index])
                # return optimal_point, optimal_image
        ind = ind + 1
        if ind == len(sorted_np_ankle_hip_x) - 1 or len(frame_valid_knee_leg_angle) > 0:
            optimal_image = 'not'
    if optimal_image == 'not':
        optimal_image = ''
    np_frame_valid_knee_leg_angle = np.array(frame_valid_knee_leg_angle)
    sorted_ind_valid_knee_leg_angle = np.argsort(np_frame_valid_knee_leg_angle)
    sorted_ind_valid_knee_leg_angle = sorted_ind_valid_knee_leg_angle[::-1]

    optimal_point = valid_index[sorted_ind_valid_knee_leg_angle[0]]
    optimal_image = os.path.basename(image_name_list[optimal_point])
    return optimal_point, optimal_image


# Optimal frame when foot is touching the ground
def optimal_frame_foot_touching_ground(hip_ankle_distance_y, hip_ankle_distance_x, direction, image_name_list, all_hip_coord, all_shoulder_coord, all_knee_leg_angle, all_ankle_coor, all_ankle_another_leg_coor):

    valid_knee_angle = []
    valid_index = []
    optimal_point = 0
    optimal_image = ''

    for index in range(0, len(all_knee_leg_angle)):
        if direction == 'positive' and hip_ankle_distance_x[index] > 0 and hip_ankle_distance_y[index] > 0 and all_shoulder_coord[index][1] < all_hip_coord[index][1] and all_ankle_coor[index][0] > all_ankle_another_leg_coor[index][0] and all_ankle_another_leg_coor[index][0] < all_hip_coord[index][0]:
            valid_knee_angle.append(all_knee_leg_angle[index])
            valid_index.append(index)
        if direction == 'negative' and hip_ankle_distance_x[index] < 0 and hip_ankle_distance_y[index] > 0 and all_shoulder_coord[index][1] < all_hip_coord[index][1] and all_ankle_coor[index][0] < all_ankle_another_leg_coor[index][0] and all_ankle_another_leg_coor[index][0] >= all_hip_coord[index][0]:
            valid_knee_angle.append(all_knee_leg_angle[index])
            valid_index.append(index)

    if len(valid_knee_angle) <= 0:
        print(" No Valid_ Optimal Frame for foot touching ground is found")
        return optimal_point, optimal_image

    np_valid_knee_leg_angle = np.array(valid_knee_angle)
    sorted_valid_index_knee_angle = np.argsort(np_valid_knee_leg_angle)
    optimal_point = 0
    optimal_image = ''

    ind = 0
    while optimal_image == '':
        frame_valid_hip_ankle_disty = []
        optimal_index = []
        for index in sorted_valid_index_knee_angle:
            if np_valid_knee_leg_angle[index] <= np_valid_knee_leg_angle[sorted_valid_index_knee_angle[ind]] + 10:

                frame_valid_hip_ankle_disty.append(
                    hip_ankle_distance_y[valid_index[index]])
                optimal_index.append(valid_index[index])

        ind = ind + 1
        if len(frame_valid_hip_ankle_disty) > 0 or ind == len(sorted_valid_index_knee_angle) - 1:
            optimal_image = 'not'

    optimal_image = ''
    if len(frame_valid_hip_ankle_disty) > 0:
        np_valid_hip_ankle_disty = np.array(frame_valid_hip_ankle_disty)
        sorted_index_valid_hip_ankle_disty = np.argsort(
            np_valid_hip_ankle_disty)
        sorted_index_valid_hip_ankle_disty = sorted_index_valid_hip_ankle_disty[::-1]

        optimal_point = optimal_index[sorted_index_valid_hip_ankle_disty[0]]
        optimal_image = os.path.basename(image_name_list[optimal_point])

    return optimal_point, optimal_image


# Optimal frame when foot is leaving the ground
def optimal_frame_foot_leaving_ground(hip_ankle_distance_y, hip_ankle_distance_x, direction, image_name_list, all_hip_coord, all_shoulder_coord, all_knee_leg_angle, all_ankle_coor, all_ankle_another_leg_coor):

    valid_knee_angle = []
    valid_index = []
    optimal_point = 0
    optimal_image = ''

    for index in range(0, len(all_knee_leg_angle)):
        if direction == 'positive' and hip_ankle_distance_x[index] > 0 and hip_ankle_distance_y[index] > 0 and all_shoulder_coord[index][1] < all_hip_coord[index][1] and all_ankle_coor[index][0] > all_ankle_another_leg_coor[index][0] and all_ankle_another_leg_coor[index][0] < all_hip_coord[index][0]:
            valid_knee_angle.append(all_knee_leg_angle[index])
            valid_index.append(index)
        if direction == 'negative' and hip_ankle_distance_x[index] < 0 and hip_ankle_distance_y[index] > 0 and all_shoulder_coord[index][1] < all_hip_coord[index][1] and all_ankle_coor[index][0] < all_ankle_another_leg_coor[index][0] and all_ankle_another_leg_coor[index][0] >= all_hip_coord[index][0]:
            valid_knee_angle.append(all_knee_leg_angle[index])
            valid_index.append(index)

    if len(valid_knee_angle) <= 0:
        print(" No Valid_ Optimal Frame for foot leaving ground is found")
        return optimal_point, optimal_image

    np_valid_knee_leg_angle = np.array(valid_knee_angle)
    sorted_index_valid_knee_angle = np.argsort(np_valid_knee_leg_angle)

    ind = 0

    while optimal_image == '':
        frame_valid_hip_ankle_distx = []
        optimal_index = []
        for index in sorted_index_valid_knee_angle:
            if valid_knee_angle[index] <= valid_knee_angle[sorted_index_valid_knee_angle[ind]] + 5:
                current_index = valid_index[index]
                frame_valid_hip_ankle_distx.append(
                    abs(hip_ankle_distance_x[current_index]))
                optimal_index.append(current_index)

        ind = ind + 1
        if len(frame_valid_hip_ankle_distx) > 0 or ind == len(sorted_index_valid_knee_angle) - 1:
            optimal_image = 'not'

    optimal_image = ''
    np_valid_hip_ankle_distx = np.array(frame_valid_hip_ankle_distx)
    if len(frame_valid_hip_ankle_distx) > 0:
        sorted_index_valid_hip_ankle_distx = np.argsort(
            np_valid_hip_ankle_distx)
        sorted_index_valid_hip_ankle_distx = sorted_index_valid_hip_ankle_distx[::-1]

        optimal_point = optimal_index[sorted_index_valid_hip_ankle_distx[0]]
        optimal_image = os.path.basename(image_name_list[optimal_point])

    return optimal_point, optimal_image


# Optimal frame when foot is on the ground for back and front view
def optimal_frame_foot_on_ground_back_front(all_hip_coord, all_knee_coord, all_ankle_coord, all_knee_coord_anotherleg, all_ankle_coord_anotherleg, image_name_list, all_shoulder_coord):
    # print(all_hip_coord, all_knee_coord, all_ankle_coord, all_knee_coord_anotherleg, all_ankle_coord_anotherleg, image_name_list, all_shoulder_coord)

    np_another_knee_ankle_disty = np.array(calculate_distance_list(
        all_knee_coord_anotherleg, all_ankle_coord_anotherleg, 'y'))
    np_knee_another_knee_disty = np.array(calculate_distance_list(
        all_knee_coord, all_knee_coord_anotherleg, 'y'))

    sorted_ind_knee_another_knee_y = np.argsort(np_knee_another_knee_disty)

    frame_valid_another_knee_ankle_disty = []
    valid_index = []

    optimal_point = 0
    optimal_image = ''
    for index in sorted_ind_knee_another_knee_y:
        if all_shoulder_coord[index][1] < all_hip_coord[index][1] and all_hip_coord[index][1] < all_knee_coord[index][1] and abs(all_knee_coord[index][1] - all_knee_coord_anotherleg[index][1]) <= np_knee_another_knee_disty[sorted_ind_knee_another_knee_y[0]] + 10:
            valid_index.append(index)
            frame_valid_another_knee_ankle_disty.append(
                np_another_knee_ankle_disty[index])

    np_valid_another_knee_ankle_disty = np.array(
        frame_valid_another_knee_ankle_disty)
    sorted_index_valid_another_knee_ankle_disty = np.argsort(
        np_valid_another_knee_ankle_disty)

    # print(valid_index,sorted_index_valid_another_knee_ankle_disty)
    optimal_point = valid_index[sorted_index_valid_another_knee_ankle_disty[0]]
    optimal_image = os.path.basename(image_name_list[optimal_point])
    # print(optimal_point,optimal_image,all_ankle_coord[optimal_point])
    # optimal_image_list = []
    # image_list = []
    # for index in sorted_index_valid_another_knee_ankle_disty:
    #     optimal_image_list.append(valid_index[index])

    # for index in optimal_image_list:
    #     image_list.append(os.path.basename(image_name_list[index]))

    # print(optimal_image_list,image_list)

    return optimal_point, optimal_image

# Optimal frame when foot is on the ground for back and front view with max angles
# def optimal_frame_foot_on_ground_back_front_max_angles(all_hip_coord, all_knee_coord, all_ankle_coord, all_knee_coord_anotherleg, all_ankle_coord_anotherleg, image_name_list, all_shoulder_coord, angle_list, type=''):
#
#     np_another_knee_ankle_disty = np.array(calculate_distance_list(
#         all_knee_coord_anotherleg, all_ankle_coord_anotherleg, 'y'))
#     np_knee_another_knee_disty = np.array(calculate_distance_list(
#         all_knee_coord, all_knee_coord_anotherleg, 'y'))
#
#     sorted_ind_knee_another_knee_y = np.argsort(np_knee_another_knee_disty)
#
#     frame_valid_another_knee_ankle_disty = []
#     valid_index = []
#
#     optimal_point = 0
#     optimal_image = ''
#     optimal_image_list = []
#     for index in sorted_ind_knee_another_knee_y:
#         if all_shoulder_coord[index][1] < all_hip_coord[index][1] and all_hip_coord[index][1] < all_knee_coord[index][1] and abs(all_knee_coord[index][1] - all_knee_coord_anotherleg[index][1]) <= np_knee_another_knee_disty[sorted_ind_knee_another_knee_y[0]] + 10:
#             valid_index.append(index)
#             frame_valid_another_knee_ankle_disty.append(
#                 np_another_knee_ankle_disty[index])
#
#     np_valid_another_knee_ankle_disty = np.array(
#         frame_valid_another_knee_ankle_disty)
#     sorted_index_valid_another_knee_ankle_disty = np.argsort(
#         np_valid_another_knee_ankle_disty)
#
#     # print(sorted_index_valid_another_knee_ankle_disty, len(sorted_index_valid_another_knee_ankle_disty))
#
#     for index in sorted_index_valid_another_knee_ankle_disty:
#         optimal_image_list.append(valid_index[index])
#
#     # x_image = []
#     # for x in optimal_image_list:
#     #     x_image.append(os.path.basename(image_name_list[x]))
#     # print(x_image)
#
#
#     if type == 'non_trunk':
#         # print('non_trunk')
#         max_negative_angle = 0
#         for image in optimal_image_list:
#             angle = angle_list[image]
#             # print("Actual angle is {}".format(angle))
#             if angle < 0:
#                 # print("angle < 0 {}".format(angle))
#                 if angle < max_negative_angle:
#                     # print("selected angle is {}".format(angle))
#                     max_negative_angle = angle
#                     optimal_point = image
#         # print("Final :",max_negative_angle)
#         optimal_image = os.path.basename(image_name_list[optimal_point])
#
#     if type == "trunk":
#         # print('trunk')
#         # print(optimal_image_list)
#         max_angle = 0
#         for image in optimal_image_list:
#             angle = angle_list[image]
#             # print("Actual angle is {}".format(angle))
#             if abs(angle) > abs(max_angle):
#                 # print(angle,max_angle)
#                 # print("Trunk angle is {}".format(angle))
#                 max_angle = angle
#                 optimal_point = image
#         # print("Selected angle :",max_angle)
#         # optimal_image = os.path.basename(image_name_list[optimal_point])
#         optimal_image = os.path.basename(image_name_list[optimal_point])
#     # print(optimal_point, optimal_image)
#
#     return optimal_point, optimal_image

def optimal_frame_foot_on_ground_back_front_max_angles(all_hip_coord, all_knee_coord, all_ankle_coord, all_knee_coord_anotherleg, all_ankle_coord_anotherleg, image_name_list, all_shoulder_coord, angle_list, vid_keypoints, flag, type=''):

    optimal_point = 0
    optimal_image = ''
    optimal_image_list = []
    # print(sorted_index_valid_another_knee_ankle_disty, len(sorted_index_valid_another_knee_ankle_disty))
    #####
    # print('np.shape = ', np.shape(vid_keypoints))
    vid_keypoints = np.array(vid_keypoints)
    print('arr_keypoints_old = ', np.shape(vid_keypoints))
    arr_keypoints = vid_keypoints[:, 1:, :]
    print('arr_keypoints = ', np.shape(arr_keypoints))
    arr_keypoints = arr_keypoints.reshape(arr_keypoints.shape[0], arr_keypoints.shape[1] * arr_keypoints.shape[2])
    # print('np.shape = ', np.shape(arr_keypoints))
    # # normalizing keypoints
    X_list = []
    w_size = 17
    t = math.floor(w_size / 2)
    i_range = list(np.arange(0 - t, len(arr_keypoints) + t))
    zeros = list(np.zeros(len(arr_keypoints[0])).astype(int))
    print ('arr_keypoints = ',np.shape(arr_keypoints))

    for i in range(len(arr_keypoints)):
        list1 = []
        for w in range(w_size):
            if (i_range[i + w] < 0):
                list1 = np.concatenate((list1, zeros))
            elif (i_range[i + w] >= len(arr_keypoints)):
                list1 = np.concatenate((list1, zeros))
            else:
                list1 = np.concatenate((list1, arr_keypoints[i + w - t]))
        # print(list1)
        X_list.append(list1)
    arr_keypoints = X_list
    print("@"*12,np.shape(arr_keypoints))
    arr_keypoints = min_max_scaler.transform(arr_keypoints)
    labels = rf.predict(arr_keypoints)
    # print('labels shape = ', np.shape(labels))
    # print ('labels = ',labels)
    #
    # appending predicted labels
    idx = np.where(labels == flag)
    #####
    optimal_image_list = idx[0]
    if (flag == 2):
        angle_list_copy = np.array(angle_list)
        image_name_list_copy = np.array(image_name_list)
        print ('-----------')
        print ('type = ',type)
        print('frames of left foot on ground = ', image_name_list_copy[idx])
        print ('indexes of left foot on ground = ',idx)
        print ('angles = ',angle_list_copy[idx])
        print('max neg angles = ', np.min(angle_list_copy[idx]))
        minIdx = np.argmin(angle_list_copy[idx])
        img_lsit_left = image_name_list_copy[idx]
        print ('frame no of min angle = ',img_lsit_left[minIdx])
    # print (type(angle_list))
    # x_image = []
    # for x in optimal_image_list:
    #     x_image.append(os.path.basename(image_name_list[x]))
    # print(x_image)
    print (len(labels),len(angle_list))

    if type == 'non_trunk':
        # print('non_trunk')
        max_negative_angle = 0
        for image in optimal_image_list:
            # print ('image = ',image)
            angle = angle_list[image]
            # print("Actual angle is {}".format(angle))
            if angle < 0:
                # print("angle < 0 {}".format(angle))
                if angle < max_negative_angle:
                    # print("selected angle is {}".format(angle))
                    max_negative_angle = angle
                    optimal_point = image
        # print("Final :",max_negative_angle)
        optimal_image = os.path.basename(image_name_list[optimal_point])
    if type == "trunk":
        # print('trunk')
        # print(optimal_image_list)
        max_angle = 0
        for image in optimal_image_list:
            # print('image = ', image)
            angle = angle_list[image]
            # print ('angles on predicted left foot on fround = ',angle)
            # print("Actual angle is {}".format(angle))
            if abs(angle) > abs(max_angle):
                # print(angle,max_angle)
                # print("Trunk angle is {}".format(angle))
                max_angle = angle
                optimal_point = image
        # print("Selected angle :",max_angle)
        # optimal_image = os.path.basename(image_name_list[optimal_point])
        optimal_image = os.path.basename(image_name_list[optimal_point])
    # print(optimal_point, optimal_image)
    return optimal_point, optimal_image

################################################################

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    vA = [(lineA[0]-lineA[2]), (lineA[1]-lineA[3])]
    vB = [(lineB[0]-lineB[2]), (lineB[1]-lineB[3])]
    dot_prod = dot(vA, vB)
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    cos_ = dot_prod/magA/magB
    angle = math.acos(dot_prod/magB/magA)
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        return 360 - ang_deg
    else:

        return ang_deg

def middle_line_image(width,height):
    points = [0,height/2,width,height/2]
    points = np.array(points).astype(int)
    return points

def line_length(line):
    length = (line[2] - line[0])
    return length

def middle_point_coordinates(line):
    points = ([(line[0] + line[2])/2,(line[1] + line[3])/2])
    points = np.array(points).astype(int)
    return points

def image_points(point_array):
    x1 = int(point_array[0])
    y1 = int(point_array[1])
    x2 = int(point_array[2])
    y2 = int(point_array[3])
    return x1,y1,x2,y2

# hip angle maut
def image_processing(img_path,points):
#     read image
    try:
        # print ('img_path = ',img_path)
        im = cv2.imread(img_path)
        # print('im = ', np.shape(im))
        #     image points
        x1,y1,x2,y2 =image_points(points)
        # print ('x1,y1,x2,y2 = ',x1,y1,x2,y2)
        # x1,y1,x2,y2 = np.array([x1,y1,x2,y2],dtype=np.int)
        #     crop image
        # print(x1,y1,x2,y2,img_path)
        img = im[y1:y2, x1:x2]
        # print('saving tmp img',x1,y1,x2,y2)
        # print(img.shape)
        # cv2.imwrite('/opt/detectron2/tools/101/tmp_imgs/'+str(np.random.randint(0,9999))+'.png',img)
        # cv2.imw
        # # #     crop image height_width
        height, width, _ = img.shape
        # print(img.shape)
        # # #     image draw canny edge detector
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,600,apertureSize = 3)
        # # #     hough line
        lines = cv2.HoughLinesP(edges, 2, math.pi/180, 15, 100, 30, 7)
        lines = np.squeeze(lines)

        # # #     mid draw line on image mid
        mid_line = middle_line_image(width,height)
        min_line = (line_length(mid_line))/2
        middle_point = middle_point_coordinates(mid_line)
        # # #     color = (0, 255, 0)
        # # #     thickness = 1
        line_distance = 1000
        final_angle = 100
        final_line = 0
        final_distance = 0
        for line in lines:
            angle = ang(mid_line,line)
            line_length_= int(line_length(line))
            line_mid_points = middle_point_coordinates(line)

            # print(mid_line,line,min_line,line_length_,line_mid_points,angle)
            # and line_length_ > (min_line - 6)
            if angle < 15 and line_length_ > (min_line - 10):
                distance = middle_point[1] -  line_mid_points[1]
                if distance < line_distance:
                    line_distance= distance
                    # print("angle : {}, line length : {}, distance : {}".format(angle,line_length_,distance))
                    if angle != 0:
                        if angle < final_angle:
                            final_angle = round(angle,1)
                            final_line = line
                            mid = mid_line
                            final_distance = distance
        #final_angle min angle num
        #mid mid line coords [x,y,x,y]
        #final_line angle line
        #final_distance euc(mid,final_line(mid)) num
        return final_angle,mid,final_line,final_distance
    except:
        # print('image processing error')
        pass


def image_processing_right(img_path,points,right_side_path,img_name):
#     read image
    try:
        # print("inner try")
        # print(img_path,points,right_side_path)
        im = cv2.imread(img_path)
    #     # print(im)
    # #     image points
        x1,y1,x2,y2 =image_points(points)
    # #     crop image
        img = im[y1:y2, x1:x2]



    #     # plt.imshow(img)
        # crop image height_width
        height, width, _ = img.shape
    #     # print(height, width)

    # # #     image draw canny edge detector
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,100,600)
    # # #     hough line
        lines = cv2.HoughLinesP(edges, 1, math.pi/180, 1, 5, 11, 7)
        lines = np.squeeze(lines)
        # print(len(lines))

    # # #     mid draw line on image mid
        mid_line = middle_line_image(width,height)
        min_line = line_length(mid_line)
        middle_point = middle_point_coordinates(mid_line)
        color = (0, 255, 255)
        thickness = args.line_thickness
        # image = cv2.line(img, (mid_line[0],mid_line[1]), (mid_line[2],mid_line[3]), color, thickness)
        line_distance = 1000
        angle_list = []
        line_select = 0
        for ind,line in enumerate(lines):
            angle = ang(mid_line,line)
            line_length_= int(line_length(line))
            line_mid_points = middle_point_coordinates(line)
            if angle < 45 and angle != 0:
                distance = middle_point[1] -  line_mid_points[1]
                if abs(distance) < line_distance:
                    line_distance= abs(distance)
                    angle_list.append(round(angle,1))
                    line_select = line
        new_mid_points = [points[0],int((points[1]+points[3])/2),points[2],int((points[1]+points[3])/2)]
        # print(mid_line,line_select,points,new_mid_points)
        newLine_difference =[int(line_select[0]-mid_line[0]),int(line_select[1]-mid_line[1]),int(line_select[2]-mid_line[0]),int(line_select[3]-mid_line[1])]
        newLine = [(new_mid_points[0]+newLine_difference[0]),(new_mid_points[1]+newLine_difference[1]),(new_mid_points[0]+newLine_difference[2]),(new_mid_points[1]+newLine_difference[3])]
        # print(mid_line,line_select,points,new_mid_points,newLine_difference,newLine)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        fontColor = (0, 0, 255)
        lineType = 2

        image_original = cv2.imread(right_side_path+''+img_name)
        # print(right_side_path+''+img_name,image_original)
        image_original = cv2.line(image_original, (new_mid_points[0],new_mid_points[1]), (new_mid_points[2],new_mid_points[3]), color, thickness)
        image_original = cv2.line(image_original, (newLine[0],newLine[1]), (newLine[2],newLine[3]), color, thickness)
        image_original = cv2.putText(image_original, str((round(angle_list[0], 1))),
                        (int(newLine[2]+10), int(newLine[3])), font, fontScale, fontColor, lineType)
        cv2.imwrite(right_side_path+''+img_name,image_original)
        final_angle = angle_list[0]
        print(final_angle)
        return final_angle
    except:
        # print("inner pass")
        pass

def line_segment(type, distance):
    values = []
    if type == 'right':
        m_x = 3
        c_x = -50
        m_y = -11
        c_y = 138
        value_x = m_x * distance + c_x
        value_y = m_y * distance + c_y
        values.append(value_x)
        values.append(value_y)

    if type == 'left':
        m_x = -4
        c_x = 59
        m_y = -11
        c_y = 138
        value_x = m_x * distance + c_x
        value_y = m_y * distance + c_y
        values.append(value_x)
        values.append(value_y)

    return values

def feet_inner_edge(optimal_point,all_back_hip_difference,distance,type):
    ratio = all_back_hip_difference[optimal_point][0] / all_back_hip_difference[0][0]
    # print(type,ratio)
    x_point = 0
    y_point = 0
    if type == 'right':
        x = line_segment(type,distance)
        x_point = ratio * x[0]
        y_point = ratio * x[1]

    if type == 'left':
        x = line_segment(type,distance)
        x_point = ratio * x[0]
        y_point = ratio * x[1]
    return x_point,y_point

# def optimal_frame_foot_on_ground_back_front_all_points(all_hip_coord, all_knee_coord, all_ankle_coord, all_knee_coord_anotherleg, all_ankle_coord_anotherleg, image_name_list, all_shoulder_coord, angle_list,path = ''):

#     np_another_knee_ankle_disty = np.array(calculate_distance_list(
#         all_knee_coord_anotherleg, all_ankle_coord_anotherleg, 'y'))
#     np_knee_another_knee_disty = np.array(calculate_distance_list(
#         all_knee_coord, all_knee_coord_anotherleg, 'y'))

#     sorted_ind_knee_another_knee_y = np.argsort(np_knee_another_knee_disty)

#     frame_valid_another_knee_ankle_disty = []
#     valid_index = []

#     optimal_point = 0
#     optimal_image = ''
#     optimal_image_list = []
#     for index in sorted_ind_knee_another_knee_y:
#         if all_shoulder_coord[index][1] < all_hip_coord[index][1] and all_hip_coord[index][1] < all_knee_coord[index][1] and abs(all_knee_coord[index][1] - all_knee_coord_anotherleg[index][1]) <= np_knee_another_knee_disty[sorted_ind_knee_another_knee_y[0]] + 10:
#             valid_index.append(index)
#             frame_valid_another_knee_ankle_disty.append(
#                 np_another_knee_ankle_disty[index])

#     np_valid_another_knee_ankle_disty = np.array(
#         frame_valid_another_knee_ankle_disty)
#     sorted_index_valid_another_knee_ankle_disty = np.argsort(
#         np_valid_another_knee_ankle_disty)

#     # print(valid_index,sorted_index_valid_another_knee_ankle_disty)
#     optimal_point = valid_index[sorted_index_valid_another_knee_ankle_disty[0]]
#     optimal_image = os.path.basename(image_name_list[optimal_point])

#     # index_first = image_name_list.index('1.PNG')
#     print(optimal_point,optimal_image,all_ankle_coord[optimal_point])

    # print(angle_list)

    # print(sorted_index_valid_another_knee_ankle_disty, len(sorted_index_valid_another_knee_ankle_disty))

    # for index in sorted_index_valid_another_knee_ankle_disty:
    #     optimal_image_list.append(valid_index[index])
    # print(optimal_image_list)

    # x_image = []
    # for image in optimal_image_list:
    #     name = os.path.basename(image_name_list[image])
    #     print(all_back_right_feet[image],name)

    # print(all_hip_coord)


    # print(image_name_list)
    # print(optimal_image_list)
    # min_angle = 100
    # for image in optimal_image_list:
    #     image_points = angle_list[image]
    #     image_name_base = os.path.basename(image_name_list[image])
    #     image_path = path+'/'+image_name_base
    #     print(image_points,image_path)

    #     try:
    #         angle = image_processing(image_path,image_points)
    #         # print(angle)
    #         if angle < min_angle:
    #             optimal_point = image
    #             min_angle = angle
    #         if min_angle == 100:
    #             min_angle = 0

    #     except:
    #         continue
    # # # print(min_angle)

    # optimal_image = os.path.basename(image_name_list[optimal_point])

    # return optimal_point, optimal_image,min_angle

def optimal_frame_foot_on_ground_back_front_all_points(all_hip_coord, all_knee_coord, all_ankle_coord, all_knee_coord_anotherleg, all_ankle_coord_anotherleg, image_name_list, all_shoulder_coord, angle_list,path):

    np_another_knee_ankle_disty = np.array(calculate_distance_list(
        all_knee_coord_anotherleg, all_ankle_coord_anotherleg, 'y'))
    np_knee_another_knee_disty = np.array(calculate_distance_list(
        all_knee_coord, all_knee_coord_anotherleg, 'y'))

    sorted_ind_knee_another_knee_y = np.argsort(np_knee_another_knee_disty)

    frame_valid_another_knee_ankle_disty = []
    valid_index = []

    optimal_point = 0
    optimal_image = ''
    optimal_image_list = []
    for index in sorted_ind_knee_another_knee_y:
        if all_shoulder_coord[index][1] < all_hip_coord[index][1] and all_hip_coord[index][1] < all_knee_coord[index][1] and abs(all_knee_coord[index][1] - all_knee_coord_anotherleg[index][1]) <= np_knee_another_knee_disty[sorted_ind_knee_another_knee_y[0]] + 10:
            valid_index.append(index)
            frame_valid_another_knee_ankle_disty.append(
                np_another_knee_ankle_disty[index])

    np_valid_another_knee_ankle_disty = np.array(
        frame_valid_another_knee_ankle_disty)
    sorted_index_valid_another_knee_ankle_disty = np.argsort(
        np_valid_another_knee_ankle_disty)

    # print(angle_list)

    # print(sorted_index_valid_another_knee_ankle_disty, len(sorted_index_valid_another_knee_ankle_disty))

    for index in sorted_index_valid_another_knee_ankle_disty:
        optimal_image_list.append(valid_index[index])
    # # print(optimal_image_list, len(optimal_image_list))
    # print(optimal_image_list)


    min_angle = 100

    image_point_final = None
    mid_points_final = 0
    line_final = 0
    path_final = ''
    distance = 0


    # try:
    i = 0
    for image in optimal_image_list:
        i = i + 1
        image_points = angle_list[image]
        # print(image_points)
        image_name_base = os.path.basename(image_name_list[image])
        image_path = os.path.join(path,image_name_base)
        # print(image_points,image_name_base)
        # mid_points = 0
        # path_new = ''
        # new_lines = 0
        try:
            # print ('image_path,image_points = ', image_path,image_points)
            angle,mid,final_line,final_distance = image_processing(image_path,image_points)
            # print('angle, min_angle = ',angle, min_angle)
            if angle < min_angle:
                optimal_point = image
                min_angle = angle
                mid_points_final = mid
                line_final = final_line
                path_final = image_path
                distance = final_distance
                image_point_final = image_points

            if min_angle == 100:
                min_angle = 0


            #     img = cv2.imread(path_new)
            #     color = (0, 255, 0)
            #     thickness = 1
                # image = cv2.line(img, (mid_points[0],mid_points[1]), (mid_points[2],mid_points[3]), color, thickness)
                # image = cv2.line(image, (new_lines[0],new_lines[1]), (new_lines[2],new_lines[3]), color, thickness)
                # print(image_points,path_new,mid_points,new_lines)
        except:
            continue
    if min_angle != 0:
        # print("angle")
        img = cv2.imread(path_final)
        # print ('path_final', path_final)
        # print ('img = ',np.shape(img))
        color = (0, 255, 255)
        thickness = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        fontColor = (0, 0, 255)
        lineType = 2
        if image_point_final is not None:
            print ('angle2')
            new_mid_points = [image_point_final[0],int((image_point_final[1]+image_point_final[3])/2),image_point_final[2],int((image_point_final[1]+image_point_final[3])/2)]
            newLine_difference =[int(line_final[0]-mid_points_final[0]),int(line_final[1]-mid_points_final[1]),int(line_final[2]-mid_points_final[0]),int(line_final[3]-mid_points_final[1])]
            newLine = [(new_mid_points[0]+newLine_difference[0]),(new_mid_points[1]+newLine_difference[1]),(new_mid_points[0]+newLine_difference[2]),(new_mid_points[1]+newLine_difference[3])]
            # print(new_mid_points,mid_points_final,line_final,newLine_difference,newLine)
            image = cv2.line(img, (new_mid_points[0],new_mid_points[1]), (new_mid_points[2],new_mid_points[3]), color, thickness)
            img = cv2.line(img, (newLine[0],newLine[1]), (newLine[2],newLine[3]), color, thickness)
            img = cv2.putText(img, str((round(min_angle, 1))),
                            (int(newLine[2]+10), int(newLine[3])), font, fontScale, fontColor, lineType)


        # print ('shape image = ',np.shape(img))
        # print('type image = ', type(img))
        try:
            cv2.imwrite(path_final,img)
        except:
            pass
            # print ('np.shape(image) = ',np.shape(img))

    # except:
        # continue
    # # print(min_angle)

    optimal_image = os.path.basename(image_name_list[optimal_point])

    return optimal_point, optimal_image,min_angle

################################################################
video_frame_path = ''
# points_keypoints = []

def main(args):
    check = 0
    # print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
    # print(" the value of args is ", args.input_type)
    json_list = {}
    if args.input_type == 'video':
        video_name, _ = os.path.splitext(os.path.basename(args.video_path))
        creation_date = creation_date_fun(args.video_path)
        creation_date = creation_date.strftime('%B%d_%Y_%I.%M%p')
        print("Creation Date: ", creation_date)

    if args.input_type == 'image':
        video_name = args.video_path.split("/")[:-1]
        video_name = str(video_name[len(video_name) - 1])
        print(" image path name ----", video_name)
    json_list['video_name'] = video_name
    grant_permission = 'sudo chmod -R 777 ' + str(args.output_dir)
    args.output_dir = args.output_dir + video_name + '_' + creation_date + '/'
    if os.path.exists(str(args.output_dir)):
        #grant_permission='sudo chmod -R 777 '+ str(args.output_dir)
        os.system(grant_permission)
        remove_image_dir = 'rm -rf ' + str(args.output_dir)
        os.system(remove_image_dir)

    create_dir = 'sudo mkdir ' + str(args.output_dir)
    os.system(create_dir)
    os.system(grant_permission)
    if args.input_type == 'video':
        check = 1
        print(" the path is of video path", args.video_path)
        image_path_video = args.video_path.split("/")[:-1]
        image_path_video = '/'.join(image_path_video) + "/"
        print("image path video is", image_path_video)
        video_frame_path = os.path.join(str(args.output_dir), 'images')
        if os.path.exists(video_frame_path):
            remove_image_dir = 'rm -rf ' + str(args.output_dir) + 'images'
            os.system(remove_image_dir)

        create_dir = 'mkdir ' + video_frame_path
        os.system(create_dir)
        grant_permission = 'sudo chmod -R 777 ' + video_frame_path
        os.system(grant_permission)
        # generate_frames = 'ffmpeg -i ' + str(args.video_path) + ' -r ' + str(
        #     args.input_fps) + ' -qscale:v 2 ' + video_frame_path + '/%d.' + args.image_ext

        # new generate_frame
        generate_frames = 'ffmpeg -i ' + str(args.video_path) + ' -qscale:v 2 ' + video_frame_path + '/%d.' + args.image_ext
        #
        #generate_frames='ffmpeg -i '+str(args.video_path)+' -r '+str(args.input_fps) +' -f image2 -vcodec png '+video_frame_path +'/%d.'+ args.image_ext
        print("generate frames is", generate_frames)
        os.system(generate_frames)

    logger = logging.getLogger(__name__)
    '''merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()'''

    if check == 0:
        im_list = glob.iglob(args.video_path + '/*.' + args.image_ext)
        im_list = list(im_list)
    if check == 1:
        im_list = glob.iglob(video_frame_path + '/*.' + args.image_ext)
        im_list = list(im_list)
        # sorted the frame number for video
        im_list = sorted(im_list, key=lambda s: int(
            str(s).split("/")[-1][:-4]))
    # print(im_list)
    # exit(0)
    json_output = []
    all_image_no = []
    keypoint_no = 0

    image_no = 0
    img_ind = 0
    all_right_knee_leg_angle = []
    write_image_info = []

    # back side lists
    all_back_trunk_lean = []
    all_back_trunk_lean_label = []
    write_image_back_trunk_lean = []
    all_back_right_hip_left_hip = []
    write_image_back_right_hip_left_hip = []
    write_image_back_left_hip_right_hip = []
    all_back_left_hip_right_hip = []
    all_back_left_knee_hip_vertical = []
    all_back_left_knee_hip_vertical_label = []
    write_image_back_left_knee_hip_vertical = []
    all_back_right_knee_hip_vertical = []
    all_back_right_knee_hip_vertical_label = []
    write_image_back_right_knee_hip_vertical = []
    write_image_back_left_hip_knee_ankle = []
    all_back_left_knee_leg_angle = []
    all_back_left_knee_leg_angle_label = []
    all_back_right_knee_leg_angle = []
    all_back_right_knee_leg_angle_lable = []
    write_image_back_right_hip_knee_ankle = []
    all_back_left_knee_coor = []
    all_back_left_ankle_coor = []
    all_back_right_knee_coor = []
    all_back_right_ankle_coor = []
    all_back_hip_coor = []
    all_back_shoulder_coor = []

    all_backview_hip_leftankle_distance_y = []
    all_backview_hip_rightankle_distance_y = []

    # check index
    all_back_chest_yaxis_values = []

    # all hip coordinates for pelvis image segmentation
    all_back_hip_coor_left_right = []

    # all back feet left and right for crop feet
    all_back_left_feet = []
    all_back_right_feet = []

    # all back hip distace on x_axis,yaxis
    all_back_hip_difference = []

    # front side lists
    all_front_trunk_lean = []
    all_front_trunk_lean_label = []
    write_image_front_trunk_lean = []
    all_front_right_hip_left_hip = []
    write_image_front_right_hip_left_hip = []
    write_image_front_left_hip_right_hip = []
    all_front_left_hip_right_hip = []
    all_front_left_knee_hip_vertical = []
    all_front_left_knee_hip_vertical_label = []
    write_image_front_left_knee_hip_vertical = []
    all_front_right_knee_hip_vertical = []
    all_front_right_knee_hip_vertical_label = []
    write_image_front_right_knee_hip_vertical = []
    write_image_front_left_hip_knee_ankle = []
    all_front_left_knee_leg_angle = []
    all_front_left_knee_leg_angle_label = []
    all_front_right_knee_leg_angle = []
    all_front_right_knee_leg_angle_label = []
    write_image_front_right_hip_knee_ankle = []
    all_front_left_knee_coor = []
    all_front_left_ankle_coor = []
    all_front_right_knee_coor = []
    all_front_right_ankle_coor = []
    all_front_hip_coor = []
    all_front_shoulder_coor = []

    all_frontview_hip_leftankle_distance_y = []
    all_frontview_hip_rightankle_distance_y = []

    # all hip coordinates for pelvis image segmentation
    all_front_hip_coor_left_right = []

    # left_side_view
    # left side lists
    all_leftview_trunk_lean = []
    write_image_leftview_trunk_lean = []
    all_leftview_right_hip_left_hip = []
    write_image_leftview_right_hip_left_hip = []
    write_image_leftview_left_hip_right_hip = []
    all_leftview_left_hip_right_hip = []
    all_leftview_left_knee_hip_vertical = []
    write_image_leftview_left_knee_hip_vertical = []
    all_leftview_right_knee_hip_vertical = []
    write_image_leftview_right_knee_hip_vertical = []
    write_image_leftview_left_hip_knee_ankle = []
    all_leftview_left_knee_leg_angle = []
    all_leftview_right_knee_leg_angle = []
    write_image_leftview_right_hip_knee_ankle = []
    all_leftview_right_knee_ankle_vertical_angle = []
    write_image_leftview_left_knee_ankle_vertical = []
    all_leftview_left_knee_ankle_vertical_angle = []
    write_image_leftview_right_knee_ankle_vertical = []

    all_leftview_hip_leftankle_distance_xy = []
    all_leftview_hip_leftankle_distance_x = []
    all_leftview_hip_leftankle_distance_y = []
    all_leftview_hip_rightankle_distance_xy = []
    all_leftview_hip_rightankle_distance_x = []
    all_leftview_hip_rightankle_distance_y = []

    all_leftview_left_hip_coord = []
    all_leftview_left_shoulder_coord = []
    all_leftview_left_knee_coord = []
    all_leftview_left_ankle_coord = []
    all_leftview_right_knee_coord = []
    all_leftview_right_ankle_coord = []

    all_leftview_right_hip_ankle_dist = []
    all_leftview_left_hip_ankle_dist = []

    all_leftview_right_hip_knee_ankle = []
    all_leftview_left_hip_knee_ankle = []
    leftview_5frame_leftknee_ankle_distance = [0, 0, 0, 0, 0]
    leftview_5frame_rightknee_ankle_distance = [0, 0, 0, 0, 0]

    # rightview side
    all_rightview_trunk_lean = []
    write_image_rightview_trunk_lean = []
    all_rightview_right_hip_left_hip = []
    write_image_rightview_right_hip_left_hip = []
    write_image_rightview_left_hip_right_hip = []
    all_rightview_left_hip_right_hip = []
    all_rightview_left_knee_hip_vertical = []
    write_image_rightview_left_knee_hip_vertical = []
    all_rightview_right_knee_hip_vertical = []
    write_image_rightview_right_knee_hip_vertical = []
    write_image_rightview_left_hip_knee_ankle = []
    all_rightview_left_knee_leg_angle = []
    all_rightview_right_knee_leg_angle = []
    write_image_rightview_right_hip_knee_ankle = []
    all_rightview_right_knee_ankle_vertical_angle = []
    write_image_rightview_left_knee_ankle_vertical = []
    all_rightview_left_knee_ankle_vertical_angle = []
    write_image_rightview_right_knee_ankle_vertical = []

    all_rightview_hip_leftankle_distance_xy = []
    all_rightview_hip_leftankle_distance_x = []
    all_rightview_hip_leftankle_distance_y = []
    all_rightview_hip_rightankle_distance_xy = []
    all_rightview_hip_rightankle_distance_x = []
    all_rightview_hip_rightankle_distance_y = []

    all_rightview_right_hip_ankle_dist = []
    all_rightview_left_hip_ankle_dist = []

    all_rightview_right_hip_coord = []
    all_rightview_right_shoulder_coord = []
    all_rightview_right_knee_coord = []
    all_rightview_right_ankle_coord = []
    all_rightview_left_knee_coord = []
    all_rightview_left_ankle_coord = []

    all_rightview_right_hip_knee_ankle = []
    all_rightview_left_hip_knee_ankle = []

    # right view hip line
    all_rightview_right_pelvis_tilt = []

    list_of_all_images = []
    list_of_all_angles = []
    list_of_output_dir = []
    rightview_5frame_leftknee_ankle_distance = [0, 0, 0, 0, 0]
    rightview_5frame_rightknee_ankle_distance = [0, 0, 0, 0, 0]

    # distance between knee and ankle keypoint for front view calculation
    right_hip_knee_distance = []
    right_knee_ankle_distance = []
    left_hip_knee_distance = []
    left_knee_ankle_distance = []

    dist_right_hka_all = []
    dist_left_hka_all = []

    left_knee_ymax = []
    right_knee_ymax = []
    #number_half_image = int(len(im_list)/2)
    back_folder_created = False
    front_folder_created = False
    left_folder_created = False
    right_folder_created = False
    alignment = ''

    time_list = []
    total_time = []
    post_time01 = []
    post_time1 = []
    post_time2 = []
    post_time3 = []
    seq_list = []
    #json_dict = {}

    # funcs
    def draw_line(a, b, image):
        x1 = int(a[0])
        y1 = int(a[1])
        x2 = int(b[0])
        y2 = int(b[1])  # x3=c[0];y3=c[1]
        cv2.line(image, (x1, y1), (x2, y2),
                 (0, 255, 255), args.line_thickness)
        cv2.circle(image, (x1, y1), 5, (0, 255, 255), args.line_thickness)
        cv2.circle(image, (x2, y2), 5, (0, 255, 255), args.line_thickness)

    # config for writing text on images
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.75
    fontColor = (0, 0, 255)
    lineType = 2

    def draw_front_back_line(lt_shoulder, rt_shoulder, lt_hip, rt_hip, lt_knee, rt_knee, lt_ankle, rt_ankle, lt_image,
                             rt_image, pelvis_on='off'):
        hip_mean = []
        hip_mean.append((lt_hip[0] + rt_hip[0]) / 2)
        hip_mean.append((lt_hip[1] + rt_hip[1]) / 2)
        shoulder_mean = []
        shoulder_mean.append((lt_shoulder[0] + rt_shoulder[0]) / 2)
        shoulder_mean.append((lt_shoulder[1] + rt_shoulder[1]) / 2)

        draw_line(shoulder_mean, hip_mean, lt_image)
        draw_line(lt_hip, lt_knee, lt_image)
        draw_line(lt_knee, lt_ankle, lt_image)

        draw_line(shoulder_mean, hip_mean, rt_image)
        draw_line(rt_hip, rt_knee, rt_image)
        draw_line(rt_knee, rt_ankle, rt_image)

        if pelvis_on == 'on':
            draw_line(lt_hip, rt_hip, lt_image)
            draw_line(lt_hip, rt_hip, rt_image)

    def draw_left_right_line(lt_shoulder, rt_shoulder, lt_hip, rt_hip, lt_knee, rt_knee, lt_ankle, rt_ankle, lt_image,
                             rt_image, align):

        # draw_line(lt_shoulder, lt_hip, lt_image)
        # draw_line(rt_shoulder, rt_hip, rt_image)
        draw_line(rt_hip, rt_knee, rt_image)
        draw_line(lt_hip, lt_knee, lt_image)

        if align == 'left':
            draw_line(lt_shoulder, lt_hip, lt_image)
            draw_line(lt_shoulder, rt_hip, rt_image)
        if align == 'right':
            draw_line(rt_shoulder, lt_hip, lt_image)
            draw_line(rt_shoulder, rt_hip, rt_image)
        # draw_line(lt_hip, lt_knee, lt_image)
        draw_line(lt_knee, lt_ankle, lt_image)

        # draw_line(rt_hip, rt_knee, rt_image)
        draw_line(rt_knee, rt_ankle, rt_image)

    def return_angles_write_images(value1, value2, value3, output_dir, angle, img_opencv, ang_name, img_name):
        # image_actual_name=str(im_list[i]).split("/")[-1][:-4]
        # draw_line(value1,value2,img_opencv)
        write_image = str(output_dir) + str(img_name) + \
            '.' + args.image_ext
        # json_file=str(output_dir)+str(img_name)+".json"
        # write the vlue in json with same name as of image name
        # with open(json_file, 'w') as outfile:
        #	json.dump(str(angle), outfile)
        return write_image, img_opencv

    def calculate_hip_knee_ankle_front_back(hip, knee, ankle, output_dir, img_opencv, ang_name, img_name, ang_dir,
                                            txt_pos):
        angle = calc_angle(np.array(hip), np.array(knee), np.array(ankle))
        angle = (180 - angle) * ang_dir
        write_image, out_img = return_angles_write_images(
            hip, knee, ankle, output_dir, angle, img_opencv, ang_name, img_name)
        # draw_line(knee, ankle, out_img)
        # ang_xmin, ang_ymin = verify_and_return_boundary(int(knee[0]-150), int(knee[1]), 0, 0, out_img.rows, out_img.cols)
        # txt_xmin, txt_ymin = verify_and_return_boundary(int(knee[0]-400), int(knee[1]), 0, 0, out_img.rows, out_img.cols)
        if txt_pos == 'right':
            cv2.putText(out_img, str((round(angle, 1))),
                        (int(knee[0] + 20), int(knee[1])), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                knee[0] + 100), int(knee[1])), font, fontScale, fontColor, lineType)
        if txt_pos == 'left':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                knee[0] - 200), int(knee[1])), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                knee[0] - 130), int(knee[1])), font, fontScale, fontColor, lineType)
        return write_image, angle, out_img

    def calculate_knee_hip_vertical_front_back(knee, hip, output_dir, img_opencv, ang_name, img_name, ang_dir, txt_pos):
        vertical_coord = []
        vertical_coord.append(hip[0])
        vertical_coord.append(knee[1])
        angle = calc_angle(np.array(knee), np.array(
            hip), np.array(vertical_coord)) * ang_dir
        write_image, out_img = return_angles_write_images(
            knee, hip, vertical_coord, output_dir, angle, img_opencv, ang_name, img_name)
        # ang_xmin, ang_ymin = verify_and_return_boundary(int(knee[0]-150), int(knee[1]), 0, 0, out_img.rows, out_img.cols)
        # txt_xmin, txt_ymin = verify_and_return_boundary(int(knee[0]-400), int(knee[1]), 0, 0, out_img.rows, out_img.cols)
        if txt_pos == 'right':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                hip[0] + 10), int(hip[1] + 50)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                hip[0] + 100), int(hip[1] + 50)), font, fontScale, fontColor, lineType)
        if txt_pos == 'left':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                hip[0] - 230), int(hip[1] + 50)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                hip[0] - 160), int(hip[1] + 50)), font, fontScale, fontColor, lineType)
        return write_image, angle, out_img

    def calculate_hip_pelvis_front_back(hip1, hip2, output_dir, img_opencv, ang_name, img_name, ang_dir, txt_pos):
        horiz_coord = []
        horiz_coord.append(hip1[0])
        horiz_coord.append(hip2[1])
        angle = calc_angle(np.array(hip1), np.array(
            hip2), np.array(horiz_coord)) * ang_dir
        write_image, out_img = return_angles_write_images(
            hip1, hip2, horiz_coord, output_dir, angle, img_opencv, ang_name, img_name)
        # ang_xmin, ang_ymin = verify_and_return_boundary(int(knee[0]-150), int(knee[1]), 0, 0, out_img.rows, out_img.cols)
        # txt_xmin, txt_ymin = verify_and_return_boundary(int(knee[0]-400), int(knee[1]), 0, 0, out_img.rows, out_img.cols)
        if txt_pos == 'right':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                hip2[0] + 20), int(hip2[1] - 30)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                hip2[0] + 80), int(hip2[1] - 30)), font, fontScale, fontColor, lineType)
        if txt_pos == 'left':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                hip2[0] - 230), int(hip2[1] - 30)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                hip2[0] - 150), int(hip2[1] - 30)), font, fontScale, fontColor, lineType)

        return write_image, angle, out_img

    def calculate_trunk_lean_front_back(hip1, hip2, shoulder1, shoulder2, output_dir, img_opencv, ang_name, img_name,
                                        ang_dir, txt_pos):
        hip_mean = []
        hip_mean.append((hip1[0] + hip2[0]) / 2)
        hip_mean.append((hip1[1] + hip2[1]) / 2)
        shoulder_mean = []
        shoulder_mean.append((shoulder1[0] + shoulder2[0]) / 2)
        shoulder_mean.append((shoulder1[1] + shoulder2[1]) / 2)
        vertical_coord = []
        vertical_coord.append(hip_mean[0])
        vertical_coord.append(shoulder_mean[1])

        angle = calc_angle(np.array(shoulder_mean), np.array(
            hip_mean), np.array(vertical_coord)) * ang_dir
        write_image, out_img = return_angles_write_images(
            shoulder_mean, hip_mean, vertical_coord, output_dir, angle, img_opencv, ang_name, img_name)
        if txt_pos == 'right':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                shoulder_mean[0] - 120), int(shoulder_mean[1] - 50)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                shoulder_mean[0] - 40), int(shoulder_mean[1] - 50)), font, fontScale, fontColor, lineType)
        if txt_pos == 'left':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                shoulder_mean[0] - 120), int(shoulder_mean[1] - 50)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                shoulder_mean[0] - 40), int(shoulder_mean[1] - 50)), font, fontScale, fontColor, lineType)
        return write_image, angle, out_img

    def calculate_trunk_lean_left_right(hip, shoulder, output_dir, img_opencv, ang_name, img_name, ang_dir, txt_pos):
        vertical_coord = []
        vertical_coord.append(hip[0])
        vertical_coord.append(shoulder[1])
        angle = calc_angle(np.array(shoulder), np.array(hip),
                           np.array(vertical_coord)) * ang_dir
        write_image, out_img = return_angles_write_images(
            shoulder, hip, vertical_coord, output_dir, angle, img_opencv, ang_name, img_name)
        if txt_pos == 'right':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                shoulder[0] - 90), int(shoulder[1] - 30)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                shoulder[0] - 10), int(shoulder[1] - 30)), font, fontScale, fontColor, lineType)
        if txt_pos == 'left':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                shoulder[0] - 90), int(shoulder[1] - 30)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                shoulder[0] - 10), int(shoulder[1] - 30)), font, fontScale, fontColor, lineType)
        return write_image, angle, out_img

    def calculate_knee_ankle_vertical(knee, ankle, output_dir, img_opencv, ang_name, img_name, ang_dir, txt_pos):
        vertical_coord = []
        vertical_coord.append(ankle[0])
        vertical_coord.append(knee[1])
        angle = calc_angle(np.array(knee), np.array(
            ankle), np.array(vertical_coord)) * ang_dir
        write_image, out_img = return_angles_write_images(
            knee, ankle, vertical_coord, output_dir, angle, img_opencv, ang_name, img_name)
        if txt_pos == 'right':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                ankle[0] + 20), int(ankle[1] - 50)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                ankle[0] + 110), int(ankle[1] - 50)), font, fontScale, fontColor, lineType)
        if txt_pos == 'left':
            cv2.putText(out_img, str((round(angle, 1))), (int(
                ankle[0] - 250), int(ankle[1] - 50)), font, fontScale, fontColor, lineType)
            cv2.putText(out_img, str(ang_name), (int(
                ankle[0] - 170), int(ankle[1] - 50)), font, fontScale, fontColor, lineType)
        return write_image, angle, out_img

    def calculate_hip_ankle_distance_left_right(lefthip, righthip, ankle, output_dir, img_opencv, ang_name, img_name,
                                                dist_dir, txt_pos):
        centre_hip_x = (lefthip[0] + righthip[0]) / 2
        distance_in_px = ankle[0] - centre_hip_x
        # distance_in_px7 = distance_in_px*
        distance_in_cm = distance_in_px * \
            (2.54 / 25) * (int(args.cam_dist) / 7) * dist_dir
        if txt_pos == 'right':
            cv2.putText(img_opencv, str((round(distance_in_cm, 1))), (int(
                ankle[0] - 90), int(ankle[1] + 40)), font, fontScale, fontColor, lineType)
            cv2.putText(img_opencv, str(ang_name), (int(
                ankle[0] - 10), int(ankle[1] + 40)), font, fontScale, fontColor, lineType)
        if txt_pos == 'left':
            cv2.putText(img_opencv, str((round(distance_in_cm, 1))), (int(
                ankle[0] - 90), int(ankle[1] + 40)), font, fontScale, fontColor, lineType)
            cv2.putText(img_opencv, str(ang_name), (int(
                ankle[0] - 10), int(ankle[1] + 40)), font, fontScale, fontColor, lineType)

        # image_actual_name=str(im_list[i]).split("/")[-1][:-4]
        write_image = str(output_dir) + str(img_name) + \
            '.' + args.image_ext
        # json_file=str(output_dir)+str(img_name)+".json"
        # with open(json_file, 'w') as outfile:
        #	json.dump(str(distance_in_cm), outfile)
        return write_image, distance_in_cm, img_opencv

    def calculate_hip_ankle_distance_pixel(hip, ankle):
        dist_in_x = ankle[0] - hip[0]
        dist_in_y = ankle[1] - hip[1]

        distance_in_pixel = math.sqrt(
            dist_in_x * dist_in_x + dist_in_y * dist_in_y)

        return dist_in_x, dist_in_y, distance_in_pixel
    # now im_list have all the image file name. It will run for each image present in list
    process_starttime = time.time()

    skip_frames = 0
    start_tracking = 20
    num_keypoints = 17
    #tracker = pt(tracker="KCF")
    kalman = []
    vid_keypoints = []
    seq_dict_list=[]
    keyangle_dict_list = []
    vid_images_list = []
    image_actual_name_list = []
    for i in range(num_keypoints):
        kalman.append(kf())
    u = []
    umax = 0
    os.system('mkdir knee_points_outputs')
    os.system('mkdir results')
    for i, im_name in enumerate(im_list):
        inprocess_start = time.time()

        seq_dict = {}
        keyangle_dict = {}

        image_no = image_no + 1
        image_actual_name = str(im_name).split("/")[-1][:-4]
        umax = i

        if image_no >= int(args.max_frame):

            break

        logger.info('Processing {} -> {}'.format(im_name, os.path.join(args.output_dir,
                                                                       (str(image_actual_name) + '.' + args.image_ext))))
        im = cv2.imread(im_name)
        # it read the image
        # timers = defaultdict(Timer)
        # t = time.time()
        # with c2_utils.NamedCudaScope(0):
        #     cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
        #         model, im, None, timers=timers
        #     )  # in this line, image is passed to detectron for detecting the key point
        # logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        # for k, v in timers.items():
        #     logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        # if i == 0:
        #     logger.info(
        #         ' \ Note: inference on the first image will be slower than the '
        #         'rest (caches and auto-tuning need to warm up)'
        #     )

        # boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
        # #     cls_boxes, cls_segms, cls_keyps)
        # inprocess_endtime = time.time() #processing time ending here
        # totaltimeused = inprocess_endtime - inprocess_start
        # total_time.append(totaltimeused)
        # times = {1: [], 2: [], 3: [], 4: [], 5:[]}
 # processing_time = process_endtime - process_starttime
        # time_list.append(processing_time)
        # logger.info('processing time: {:.3f}s'.format(process_endtime - process_starttime))
        # print("////////////////////////////////////////////////////////////////////////////////////////////")
        # print("////////////////////////////////////////////////////////////////////////////////////////////")
        # print('before processing time: {:.3f}s'.format(inprocess_endtime - inprocess_start))
        # print("////////////////////////////////////////////////////////////////////////////////////////////")
        # print("////////////////////////////////////////////////////////////////////////////////////////////")
        # for i in range(1, 5):
        #     new_shape=(int(im.shape[0]/i), int(im.shape[1]/i))
        #     im = cv2.resize(im, new_shape)
        #     start_time = time.time()

        if (i != 0 and i % (skip_frames + 1) != 0 and skip_frames != 0 ):
            # tracker_output = tracker.track(im)
            # num_classes = tracker_output["num_classes"]
            # relevant_keypoint = tracker_output["pred_keypoints"]
            # relevant_keypoint = tracker.tracker_update(im)
            # relevant_keypoint = np.array(relevant_keypoint).astype(int)
            # tik = time.time()
            # for j in range(num_keypoints):
            #     predicted = kalman[j].update(relevant_keypoint[j][0], relevant_keypoint[j][1])
            #     relevant_keypoint[j][0] = predicted[0] - predicted[2]
            #     relevant_keypoint[j][1] = predicted[1] - predicted[3]
            # tok = time.time()
            # print("Kalman Time: {}ms".format((tok-tik) * 1000))
            # #relevant_keypoint = relevant_keypoint.numpy().squeeze().astype(int)
            pass
        else:
            print(i)
            u.append(i)

            tik = time.time()
            model_output = model(im)
            #
            # im_name
            # print('python apply_net.py dump configs/densepose_rcnn_R_101_FPN_s1x.yaml model_final_c6ab63.pkl {} --output results/img_{}.pkl -v'.format(path_img, img_name))
            img_name = im_name.split('/')[-1]
            img_name = img_name.split('.')[-2]
            print ('img_name = ',img_name)



            subprocess.call(
                '/home/ubuntu/anaconda3/envs/densepose/bin/python' + ' apply_net.py' + ' dump' + ' configs/densepose_rcnn_R_101_FPN_s1x.yaml' + ' model_final_c6ab63.pkl ' + im_name + ' --output' + ' results/' + img_name + '.pkl' + ' -v',
                shell=True)

            map_l, dist_img_left, min_indcs_l, map_r, dist_img_right, min_indcs_r, x, y, x2, y2 = iuv(img_name)

            print ('min_indcs_l = ',min_indcs_l)
            print ('min_indcs_r = ',min_indcs_r)
            print ('x y = ',x, y)
            print ('img shape =',np.shape(im))

            im = cv2.rectangle(im, (x, y), (x2, y2), (0, 0, 255), 3)

            im = cv2.circle(im, (x + min_indcs_l[1][0], y + min_indcs_l[0][0]), radius=5, color=(0, 0, 255),
                             thickness=-1)
            im = cv2.circle(im, (x + min_indcs_r[1][0], y + min_indcs_r[0][0]), radius=5, color=(0, 0, 255),
                             thickness=-1)

            path = './knee_points_outputs/'+img_name+'.' + args.image_ext
            print ('path = ',path)
            cv2.imwrite(path,im)

            #
            tok = time.time()
            print("Model Time: {}ms".format((tok-tik) * 1000))
            postprocess1_start = time.time()
            keypoints = model_output['instances'].pred_keypoints.cpu(
            ).numpy().squeeze().astype(int)
            boxes = model_output['instances'].pred_boxes
            area = boxes.area().cpu().numpy()
            print("printing image location")
            # print(str(im_list[i]))


            #tracker.add_frame(im, model_output)

            #tracker.init_tracker(im, model_output)


            relevant_person = np.argmax(area)
            # print(list(boxes)[0].cpu().numpy())
            print("relevant person")

            num_classes = model_output['instances'].pred_classes

            postprocess1_end = time.time()
            postprocess2_start = time.time()

            if (len(num_classes) == 1):
                keypoints = [keypoints]

            relevant_keypoint = keypoints[relevant_person]

            num_keypoints = relevant_keypoint.shape[0] # 17x2
            #apply smoothing filter mautisim
            # for j in range(len(relevant_keypoint)):
            #     kalman[j].update(relevant_keypoint[j][0],relevant_keypoint[j][1])
            all_image_no.append(im_list[i])
        # vid_images_list.append(im)
            vid_keypoints.append(relevant_keypoint)
        # if i ==9:
        #     cv2.imwrite('img.png',im)
        # end_time = time.time()
        # print("////////////////////////////%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%////////""////////////////////")
        # print('model time: {:.3f}s'.format(end_time - start_time))
        # print("///////////////////////////%%%%%%%%%%%%%%%%%%%%%%%%%%//////////////////////////////////")
        # totaltime01 = end_time - start_time
        # times[i].append(totaltime01)

        # logger.info('Inference time: {:.3f}s'.format(end_time - start_time))
        # print('Inference time: {:.3f}s'.format(end_time - start_time))
        # boxes return number




        # all_areas = []


        # if len(boxes) <= 0:
        #    continue
        # for num_key in range(len(boxes)):
        #     x1, y1, x2, y2 = boxes[num_key]
        #     area = (x2 - x1) * (y2 - y1)
        #     #print("---------------------area of the boxes is-----------------------------------------------------")
        #     # print(area)
        #     all_areas.append(area)

        # relevant_person = all_areas.index(max(all_areas))

        # relevant_person = np.argmax(area)
        # # print(list(boxes)[0].cpu().numpy())
        # print("relevant person")

        # num_classes = model_output['instances'].pred_classes

        # postprocess1_end = time.time()
        # postprocess2_start = time.time()

        # if (len(num_classes) == 1):
        #     keypoints = [keypoints]

        # relevant_keypoint = keypoints[relevant_person]


        # x1, y1, x2, y2 = list(boxes[relevant_person])[0].cpu().numpy()
        # print("the relevant keypoints are ", x1, y1, x2, y2)
        #print("-------------relevant keypoint----------------------------------")
        # print(relevant_keypoint)
    image_no = 0

    umax+=1
    # print('Keypoints Detection Complete')
    # print(len(im_list),len(vid_keypoints))
    vid_keypoints = np.array(vid_keypoints)
    raw_keypoints = vid_keypoints.copy().astype(np.int)[:,:,:2]
    # np.save('kpts.npy',vid_keypoints)
    total_keypoints = len(vid_keypoints)

    tik = time.time()
    # proc_vid_keypoints = vid_keypoints[:,:,:2].reshape(-1,34)
    # window_length = 7
    # poly_order = 3
    # if  len(proc_vid_keypoints)<window_length:
    #     window_length=len(proc_vid_keypoints) -1
    # if window_length%2 ==0:
    #     window_length= window_length -1
    #
    # if poly_order >= window_length:
    #     poly_order = window_length -1
    #
    # # proc_vid_keypoints = savgol_filter(proc_vid_keypoints,polyorder=poly_order,window_length=window_length,axis=0)
    #
    #
    # # print(proc_vid_keypoints.shape)
    # # proc_vid_keypoints = proc_vid_keypoints.transpose()
    # # print(proc_vid_keypoints.shape)
    # proc_vid_keypoints = proc_vid_keypoints.reshape(-1,17,2)
    # hip_keypoints = proc_vid_keypoints[:,[11,12],:]
    # hip_filter_window_len = 21
    # if  len(proc_vid_keypoints)<hip_filter_window_len:
    #     hip_filter_window_len=len(proc_vid_keypoints) -1
    # if hip_filter_window_len%2 == 0:
    #     hip_filter_window_len-=1
    #
    # import matplotlib.pyplot as plt
    # plt.plot(hip_keypoints[:,0,1])
    # plt.plot(hip_keypoints[:,1,1])
    #
    # plt.title('small window savgol hipkpts')
    # plt.show()
    #
    # hip_keypoints[:,:,0] = savgol_filter(hip_keypoints[:,:,0],window_length= hip_filter_window_len,polyorder = 3,axis = 0)
    # hip_keypoints[:,:,1] = savgol_filter(hip_keypoints[:,:,1],window_length=hip_filter_window_len,polyorder =3,axis = 0)
    # # plt.plot(hip_keypoints[:,0,1])
    # # plt.plot(hip_keypoints[:,1,1])
    # # plt.title('savgol hipkpts')
    # # plt.show()
    # # sos = scipy.signal.butter(N=2,Wn=0.7,output='sos',btype = 'lowpass',fs =60)
    # # hip_keypoints[:,:,1] =scipy.signal.sosfilt(sos,hip_keypoints[:,:,1],axis=0)
    # # plt.plot(hip_keypoints[:,0,1])
    # # plt.plot(hip_keypoints[:,1,1])
    # # plt.title('butterworth')
    # # plt.show()
    # hip_ptsv_savgol  = hip_keypoints.copy()
    # kernel_size = 10
    # avg_kernel = np.ones(kernel_size)/kernel_size
    # hip_keypoints[:,0,1] = np.convolve(hip_keypoints[:,0,1],avg_kernel,'same')
    # hip_keypoints[:,1,1] = np.convolve(hip_keypoints[:,1,1],avg_kernel,'same')
    # hip_keypoints[:kernel_size] = hip_ptsv_savgol[:kernel_size]
    # hip_keypoints[-kernel_size:] = hip_ptsv_savgol[-kernel_size:]
    # plt.plot(hip_keypoints[:,0,1])
    # plt.plot(hip_keypoints[:,1,1])
    # plt.title('averaging')
    # plt.show()
    # proc_vid_keypoints[:,11:13,:] = hip_keypoints
    #
    #
    # proc_vid_keypoints = proc_vid_keypoints.reshape(-1,34)



# print(umax)
    # print(u)
    # u = np.array(u)
    plot_hip_graphs = False
    if plot_hip_graphs:
        plt.plot(vid_keypoints[:,11,1])
        plt.plot(vid_keypoints[:,12,1])
        plt.title('raw_hip_y')
        plt.show()

        plt.plot(vid_keypoints[:,15,0])
        plt.plot(vid_keypoints[:,16,0])
        plt.title('raw_ankle_x')
        plt.show()

    tmp_cap = cv2.VideoCapture(args.video_path)
    filter_input_fps = tmp_cap.get(cv2.CAP_PROP_FPS)
    print('FPS: ',filter_input_fps)
    tmp_cap.release()
    proc_vid_keypoints = keypoints_post_processing.smooth_hip_points(vid_keypoints,filter_input_fps)


    proc_vid_keypoints[:,13:,:] = savgol_filter(proc_vid_keypoints[:,13:,:],polyorder=1,window_length=3,axis=0)
    for _ in range(0,int(filter_input_fps/15)+1):
        proc_vid_keypoints[:,13:,:] = savgol_filter(proc_vid_keypoints[:,13:,:],polyorder=1,window_length=3,axis=0)



    if plot_hip_graphs:
        plt.plot(proc_vid_keypoints[:,11,1])
        plt.plot(proc_vid_keypoints[:,12,1])
        plt.title('processed_hip_y')
        plt.show()
        plt.plot(proc_vid_keypoints[:,15,0])
        plt.plot(proc_vid_keypoints[:,16,0])
        plt.title('processed_ankle_x')
        plt.show()

    if skip_frames > 0:
        interpolated_keypoints = np.zeros((umax,34))
        for i in range(34):
            tck,u = splprep(x=proc_vid_keypoints[:,i][np.newaxis],u=u)
            kpts_intrp = splev(x=np.arange(0,umax),tck = tck)[0]
            interpolated_keypoints[:,i]=kpts_intrp
        interpolated_keypoints_final = np.zeros((umax,17,3))
        interpolated_keypoints_final[:,:,2] = 1 #the value isn't used anywhere in code
        interpolated_keypoints_final[:,:,:2] = interpolated_keypoints.reshape(-1,17,2)
        interpolated_keypoints_final = interpolated_keypoints_final.astype(np.int)
    else:
        interpolated_keypoints_final =proc_vid_keypoints.reshape(-1,17,2)
    # interpolated_keypoints_final[:,:,:2] = vid_keypoints[:,:,:2]
    # stop_loop = True

    np.save('kpts',interpolated_keypoints_final)
    save_videos = False
    if save_videos:
        import kpts_video_maker
        print('saving debug videos...')
        kpts_video_maker.to_video(args.video_path,raw_keypoints,'debug_vids/'+video_name[:-3]+'_raw.avi')
        kpts_video_maker.to_video(args.video_path,interpolated_keypoints_final.astype(np.int),'debug_vids/'+video_name[:-3]+'_processed.avi')
        print('done')
        return


    # print(proc_vid_keypoints)
    # print('transposed_shape: ',proc_vid_keypoints.shape)
    # splprep()
    # proc_vid_keypoints = proc_vid_keypoints.reshape(-1,17,2)
    # vid_keypoints[:,:,:2] = proc_vid_keypoints.astype(np.int)

    tok = time.time()
    print("PolySmoothingIntrp Average Per Frame Time: {:.2f}us".format((tok-tik) * 1000000/umax))

    print("PolySmoothingIntrp Total Time: {:.2f}ms".format((tok-tik) * 1000))
    print('Performing  Calculations on all keypoints...')

    waistline_color_cluster = None


    # print(vid_keypoints.shape)
    for i, im_name in tqdm(enumerate(im_list)):

        image_no = image_no + 1

        if image_no >= int(args.max_frame):
            break

        image_actual_name = str(im_name).split("/")[-1][:-4]
        relevant_keypoint =     interpolated_keypoints_final [i]


        seq_dict = {}
        keyangle_dict = {}
        nose = []
        left_eye = []
        right_eye = []
        left_ear = []
        right_ear = []
        left_shoulder = []
        right_shoulder = []
        left_elbow = []
        right_elbow = []
        left_wrist = []
        right_wrist = []
        left_hip = []
        right_hip = []
        left_knee = []
        right_knee = []
        left_ankle = []
        right_ankle = []

        nose.append(relevant_keypoint[0][0])
        nose.append(relevant_keypoint[0][1])

        left_eye.append(relevant_keypoint[1][0])
        left_eye.append(relevant_keypoint[1][1])

        right_eye.append(relevant_keypoint[2][0])
        right_eye.append(relevant_keypoint[2][1])

        left_ear.append(relevant_keypoint[3][0])
        left_ear.append(relevant_keypoint[3][1])

        right_ear.append(relevant_keypoint[4][0])
        right_ear.append(relevant_keypoint[4][1])

        left_shoulder.append(relevant_keypoint[5][0])
        left_shoulder.append(relevant_keypoint[5][1])

        right_shoulder.append(relevant_keypoint[6][0])
        right_shoulder.append(relevant_keypoint[6][1])

        left_elbow.append(relevant_keypoint[7][0])
        left_elbow.append(relevant_keypoint[7][1])

        right_elbow.append(relevant_keypoint[8][0])
        right_elbow.append(relevant_keypoint[8][1])

        left_wrist.append(relevant_keypoint[9][0])
        left_wrist.append(relevant_keypoint[9][1])

        right_wrist.append(relevant_keypoint[10][0])
        right_wrist.append(relevant_keypoint[10][1])

        left_hip.append(relevant_keypoint[11][0])
        left_hip.append(relevant_keypoint[11][1])

        right_hip.append(relevant_keypoint[12][0])
        right_hip.append(relevant_keypoint[12][1])

        left_knee.append(relevant_keypoint[13][0])
        left_knee.append(relevant_keypoint[13][1])

        right_knee.append(relevant_keypoint[14][0])
        right_knee.append(relevant_keypoint[14][1])

        left_ankle.append(relevant_keypoint[15][0])
        left_ankle.append(relevant_keypoint[15][1])

        right_ankle.append(relevant_keypoint[16][0])
        right_ankle.append(relevant_keypoint[16][1])

        # point_new = []
        # point_new.append(relevant_keypoint[0][0])
        # point_new.append(relevant_keypoint[0][1])

        # point_new.append(relevant_keypoint[1][0])
        # point_new.append(relevant_keypoint[1][1])

        # point_new.append(relevant_keypoint[2][0])
        # point_new.append(relevant_keypoint[2][1])

        # point_new.append(relevant_keypoint[3][0])
        # point_new.append(relevant_keypoint[3][1])

        # point_new.append(relevant_keypoint[4][0])
        # point_new.append(relevant_keypoint[4][1])

        # point_new.append(relevant_keypoint[5][0])
        # point_new.append(relevant_keypoint[5][1])

        # point_new.append(relevant_keypoint[6][0])
        # point_new.append(relevant_keypoint[6][1])

        # point_new.append(relevant_keypoint[7][0])
        # point_new.append(relevant_keypoint[7][1])

        # point_new.append(relevant_keypoint[8][0])
        # point_new.append(relevant_keypoint[8][1])

        # point_new.append(relevant_keypoint[9][0])
        # point_new.append(relevant_keypoint[9][1])

        # point_new.append(relevant_keypoint[10][0])
        # point_new.append(relevant_keypoint[10][1])

        # point_new.append(relevant_keypoint[11][0])
        # point_new.append(relevant_keypoint[11][1])

        # point_new.append(relevant_keypoint[12][0])
        # point_new.append(relevant_keypoint[12][1])

        # point_new.append(relevant_keypoint[13][0])
        # point_new.append(relevant_keypoint[13][1])

        # point_new.append(relevant_keypoint[14][0])
        # point_new.append(relevant_keypoint[14][1])

        # point_new.append(relevant_keypoint[15][0])
        # point_new.append(relevant_keypoint[15][1])

        # point_new.append(relevant_keypoint[16][0])
        # point_new.append(relevant_keypoint[16][1])

        # point_new = np.array(point_new)

        # points_keypoints.append(point_new)

        # print(im_name,left_ankle,right_ankle)

        # distinguishing between side poses using noses
        # get shoulder x values
        shoulder_x = (right_shoulder[0] + left_shoulder[0]) / 2

        # postprocess2_end = time.time()
        # postprocess3_start = time.time()

        # check the value wrt nose
        if (image_no == 1 and args.input_type == 'video') or (args.pose_detection == 'on') or (args.input_type == 'image'):
            if (left_shoulder[0] < nose[0] and right_shoulder[0] > nose[0]):
                alignment = "back_side"
            elif (left_hip[0] > right_hip[0] and left_knee[0] > right_knee[0] and left_ankle[0] > right_ankle[0] and (left_hip[0] - right_hip[0]) > (left_knee[0] - right_knee[0]) and (left_hip[0] - right_hip[0]) > (left_ankle[0] - right_ankle[0])):
                alignment = "front_side"
            else:
                if (nose[0] - shoulder_x) > 0:
                    print("right_end")
                    alignment = "right_side"

                if (nose[0] - shoulder_x) < 0:
                    print("left_side")
                    alignment = "left_side"

        # draw line and circle on the image

        # funcs moved

        all_is = []
        # print("alignment is : ",alignment)
        # generate angles
        # if args.video_available == '1' and image_no == 1:
        #    json_list['pose'] = alignment

        if alignment == "back_side":  # or alignment=="front_side":
            # print("back_side_loop")
            # knee angle calculation
            # create 5 directories for backside
            # print(" the value of i is ", i)

            if back_folder_created == False:
                back_right_hip_knee_ankle_path = str(
                    args.output_dir) + "/backside/" + "right_hip_knee_ankle/"
                back_left_hip_knee_ankle_path = str(
                    args.output_dir) + "/backside/" + "left_hip_knee_ankle/"
                back_right_knee_hip_vertical_path = str(
                    args.output_dir) + "/backside/" + "right_knee_hip_vertical/"
                back_left_knee_hip_vertical_path = str(
                    args.output_dir) + "/backside/" + "left_knee_hip_vertical/"
                back_left_hip_right_hip_pelvis_path = str(
                    args.output_dir) + "/backside/" + "left_hip_right_hip_pelvis/"
                #back_right_hip_left_hip_pelvis_path=str(args.output_dir) +"backside/"+"right_hip_left_hip_pelvis/"
                back_trunk_lean_path = str(
                    args.output_dir) + "/backside/" + "trunk_lean/"
                back_side_left_image_path = str(args.output_dir) + \
                    "/backside/" + "left/" + "images/"
                back_side_right_image_path = str(args.output_dir) + \
                    "/backside/" + "right/" + "images/"
            # use shell script to build the paths
                # os.makedirs(back_right_hip_knee_ankle_path);os.makedirs(back_left_hip_knee_ankle_path);os.makedirs(back_right_knee_hip_vertical_path)
                # os.makedirs(back_left_knee_hip_vertical_path);os.makedirs(back_left_hip_right_hip_pelvis_path);os.makedirs(back_trunk_lean_path);
                os.makedirs(back_side_right_image_path)
                os.makedirs(back_side_left_image_path)
                back_folder_created = True

            # image_actual_name=str(im_list[i]).split("/")[-1][:-4]
            img_opencv_right = cv2.imread(str(im_list[i]))
            from waistline_modules import waistline
            _,waistline_angle =  waistline.draw_waistline_front_back(img_opencv_right, pts=relevant_keypoint, clt=waistline_color_cluster,
                                                                     angle_font_params=(font,fontScale,fontColor,lineType))

            img_opencv_left = img_opencv_right.copy()

            # image_actual_name=str(im_list[i]).split("/")[-1][:-4]
            # calc_angle(np.array(hip),np.array(knee),np.array(ankle))
            if verify_trunck_knee_femur_hipAnkleDist(right_shoulder, right_hip, right_knee, right_ankle, left_shoulder, left_hip, left_knee, left_ankle, 30, 50, 17, 0.3, alignment) == "true":
                # for back_right_hip_knee_ankle
                if len(right_hip) > 0 and len(right_knee) > 0 and len(right_ankle) > 0 and len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0:
                    draw_front_back_line(left_shoulder, right_shoulder, left_hip, right_hip, left_knee,
                                         right_knee, left_ankle, right_ankle, img_opencv_left, img_opencv_right, args.pelvis)

                if len(right_hip) > 0 and len(right_knee) > 0 and len(right_ankle) > 0:
                    knee_point_dir = knee_point_dir_hip_ankle_line(
                        right_hip, right_knee, right_ankle)
                    if knee_point_dir >= 0:
                        write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                            right_hip, right_knee, right_ankle, back_right_hip_knee_ankle_path, img_opencv_right, "Knee Varus/Valgus", image_actual_name, 1, 'right')
                    if knee_point_dir < 0:
                        write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                            right_hip, right_knee, right_ankle, back_right_hip_knee_ankle_path, img_opencv_right, "Knee Varus/Valgus", image_actual_name, -1, 'right')

                    write_image_back_right_hip_knee_ankle.append(write_image)
                    all_back_right_knee_leg_angle.append(angle)
                    keyangle_dict['right_knee_varus_valgus'] = round(angle, 1)
                    all_back_right_knee_leg_angle_lable.append(relevant_keypoint)

                # for back_left_hip_knee_ankle
                if len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0:
                    knee_point_dir = knee_point_dir_hip_ankle_line(
                        left_hip, left_knee, left_ankle)
                    if knee_point_dir >= 0:
                        write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                            left_hip, left_knee, left_ankle, back_left_hip_knee_ankle_path, img_opencv_left, "Knee Varus/Valgus", image_actual_name, -1, 'right')

                    if knee_point_dir < 0:
                        write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                            left_hip, left_knee, left_ankle, back_left_hip_knee_ankle_path, img_opencv_left, "Knee Varus/Valgus", image_actual_name, 1, 'right')

                    write_image_back_left_hip_knee_ankle.append(write_image)
                    all_back_left_knee_leg_angle.append(angle)
                    keyangle_dict['left_knee_varus_valgus'] = round(angle, 1)
                    all_back_left_knee_leg_angle_label.append(relevant_keypoint)

                # back_right_knee_hip_vertical
                if len(right_knee) > 0 and len(right_hip) > 0:
                    if right_knee[0] <= right_hip[0]:
                        write_image, angle, img_opencv_right = calculate_knee_hip_vertical_front_back(
                            right_knee, right_hip, back_right_knee_hip_vertical_path, img_opencv_right, "Femur Angle", image_actual_name, -1, 'left')
                    if right_knee[0] > right_hip[0]:
                        write_image, angle, img_opencv_right = calculate_knee_hip_vertical_front_back(
                            right_knee, right_hip, back_right_knee_hip_vertical_path, img_opencv_right, "Femur Angle", image_actual_name, 1, 'left')
                    write_image_back_right_knee_hip_vertical.append(
                        write_image)
                    all_back_right_knee_hip_vertical.append(angle)
                    keyangle_dict['right_femur_angle'] = round(angle, 1)
                    all_back_right_knee_hip_vertical_label.append(relevant_keypoint)

                # back_left_knee_hip_vertical
                if len(left_knee) > 0 and len(left_hip) > 0:
                    if left_knee[0] >= left_hip[0]:
                        write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                            left_knee, left_hip, back_left_knee_hip_vertical_path, img_opencv_left, "Femur Angle", image_actual_name, -1, 'right')
                    if left_knee[0] < left_hip[0]:
                        write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                            left_knee, left_hip, back_left_knee_hip_vertical_path, img_opencv_left, "Femur Angle", image_actual_name, 1, 'right')
                    write_image_back_left_knee_hip_vertical.append(write_image)
                    all_back_left_knee_hip_vertical.append(angle)
                    keyangle_dict['left_femur_angle'] = round(angle, 1)
                    all_back_left_knee_hip_vertical_label.append(relevant_keypoint)

                # back_left_hip_right_hip_pelvis_path
                if len(left_hip) > 0 and len(right_hip) > 0 and args.pelvis == 'on':
                    if False:
                        if right_hip[1] >= left_hip[1]:
                            write_image, angle, img_opencv_right = calculate_hip_pelvis_front_back(
                                left_hip, right_hip, back_left_hip_right_hip_pelvis_path, img_opencv_right, "Pelvis Angle", image_actual_name, 1, 'left')
                        if right_hip[1] < left_hip[1]:
                            write_image, angle, img_opencv_right = calculate_hip_pelvis_front_back(
                                left_hip, right_hip, back_left_hip_right_hip_pelvis_path, img_opencv_right, "Pelvis Angle", image_actual_name, -1, 'left')

                        if right_hip[1] >= left_hip[1]:
                            write_image, angle, img_opencv_left = calculate_hip_pelvis_front_back(
                                left_hip, right_hip, back_left_hip_right_hip_pelvis_path, img_opencv_left, "Pelvis Angle", image_actual_name, 1, 'left')
                        if right_hip[1] < left_hip[1]:
                            write_image, angle, img_opencv_left = calculate_hip_pelvis_front_back(
                                left_hip, right_hip, back_left_hip_right_hip_pelvis_path, img_opencv_left, "Pelvis Angle", image_actual_name, -1, 'left')


                    write_image_back_left_hip_right_hip.append(img_opencv_right)
                    all_back_left_hip_right_hip.append(waistline_angle)
                    keyangle_dict['pelvis_angle'] = round(waistline_angle, 1)
                    # print(write_image_back_left_hip_right_hip)

                # calculate_trunk_lean_front_back(hip1,hip2,shoulder1,shoulder2,output_dir)
                if len(left_hip) > 0 and len(right_hip) > 0 and len(left_shoulder) > 0 and len(right_shoulder) > 0:
                    hip_meanx = (left_hip[0] + right_hip[0]) / 2
                    shoulder_meanx = (left_shoulder[0] + right_shoulder[0]) / 2
                    if shoulder_meanx >= hip_meanx:
                        write_image, angle, img_opencv_right = calculate_trunk_lean_front_back(
                            left_hip, right_hip, left_shoulder, right_shoulder, back_trunk_lean_path, img_opencv_right, "Trunk Lateral Lean", image_actual_name, 1, 'left')
                    if shoulder_meanx < hip_meanx:
                        write_image, angle, img_opencv_right = calculate_trunk_lean_front_back(
                            left_hip, right_hip, left_shoulder, right_shoulder, back_trunk_lean_path, img_opencv_right, "Trunk Lateral Lean", image_actual_name, -1, 'left')
                    if shoulder_meanx >= hip_meanx:
                        write_image, angle, img_opencv_left = calculate_trunk_lean_front_back(
                            left_hip, right_hip, left_shoulder, right_shoulder, back_trunk_lean_path, img_opencv_left, "Trunk Lateral Lean", image_actual_name, 1, 'right')
                    if shoulder_meanx < hip_meanx:
                        write_image, angle, img_opencv_left = calculate_trunk_lean_front_back(
                            left_hip, right_hip, left_shoulder, right_shoulder, back_trunk_lean_path, img_opencv_left, "Trunk Lateral Lean", image_actual_name, -1, 'right')
                    write_image_back_trunk_lean.append(write_image)
                    all_back_trunk_lean.append(angle)
                    keyangle_dict['trunk_lateral_lean'] = round(angle, 1)
                    all_back_trunk_lean_label.append(relevant_keypoint)

                if len(right_hip) > 0 and len(left_ankle) > 0 and len(right_ankle) > 0:
                    left_ankle_distx, left_ankle_disty, left_ankle_distxy = calculate_hip_ankle_distance_pixel(
                        right_hip, left_ankle)
                    right_ankle_distx, right_ankle_disty, right_ankle_distxy = calculate_hip_ankle_distance_pixel(
                        right_hip, right_ankle)
                    all_backview_hip_leftankle_distance_y.append(
                        left_ankle_disty)
                    all_backview_hip_rightankle_distance_y.append(
                        right_ankle_disty)

                hip_mean = []
                hip_mean.append(hip_meanx)
                hip_mean.append((left_hip[1] + right_hip[1]) / 2)
                # print(hip_mean, len(hip_mean))
                shoulder_mean = []
                shoulder_mean.append(shoulder_meanx)
                shoulder_mean.append(
                    (left_shoulder[1] + right_shoulder[1]) / 2)
                # print(shoulder_mean, len(shoulder_mean))

                all_back_left_knee_coor.append(left_knee)
                all_back_left_ankle_coor.append(left_ankle)
                all_back_right_knee_coor.append(right_knee)
                all_back_right_ankle_coor.append(right_ankle)
                all_back_hip_coor.append(hip_mean)
                all_back_shoulder_coor.append(shoulder_mean)

                # hip coordinates left and right
                hip_coor_left_right = []
                hip_coor_left_right.append(left_hip[0])
                hip_coor_left_right.append((left_hip[1]+left_shoulder[1])/2)
                hip_coor_left_right.append(right_hip[0])
                hip_coor_left_right.append(right_hip[1])
                all_back_hip_coor_left_right.append(hip_coor_left_right)

                # feet coordinates
                feet_coordinate_left = []
                distance_left_y = (left_knee[1] + left_ankle[1])/2
                distance_left_image = (left_ankle[1] - left_knee[1])
                feet_coordinate_left.append(left_knee[0] - (distance_left_image/2))
                feet_coordinate_left.append(distance_left_y)
                feet_coordinate_left.append(left_ankle[0] + (distance_left_image/2))
                feet_coordinate_left.append(left_ankle[1] + distance_left_image)
                all_back_left_feet.append(feet_coordinate_left)

                feet_coordinate_right = []
                # print(right_knee,right_ankle)
                distance_right_y = (right_knee[1] + right_ankle[1])/2
                distance_right_image = ( right_ankle[1] - right_knee[1])
                feet_coordinate_right.append(right_knee[0] - (distance_right_image/2))
                feet_coordinate_right.append(distance_right_y)
                feet_coordinate_right.append(right_ankle[0] + (distance_right_image/2))
                feet_coordinate_right.append(right_ankle[1] + distance_right_image)
                all_back_right_feet.append(feet_coordinate_right)
                # print(all_back_right_feet)


                # y-axis chest
                chest_y_axis = []
                chest_y_axis.append(str(image_actual_name) + "." + args.image_ext)
                chest_y_axis.append(shoulder_mean[1])
                all_back_chest_yaxis_values.append(chest_y_axis)

                # all back hip differece
                distance_hip = []
                dist_x_hip = right_hip[0] - left_hip[0]
                distance_hip.append(dist_x_hip)
                dist_y_hip = right_hip[1] - left_hip[1]
                distance_hip.append(dist_y_hip)
                # print(left_hip,right_hip)
                all_back_hip_difference.append(distance_hip)

            cv2.imwrite(str(back_side_left_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_left)
            cv2.imwrite(str(back_side_right_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_right)
            # print('image saved is ---', image_actual_name,
            #     '-- with value of i is ----', i)
            # print('full path of image saved is -------', str(back_side_left_image_path)
            #     + str(image_actual_name) + "." + args.image_ext)
            # print('full path of image saved is -------', str(back_side_right_image_path)
            #     + str(image_actual_name) + "." + args.image_ext)


            #cv2.imwrite(str(back_side_left_image_path )+str(img_ind)+"."+args.image_ext,img_opencv_left)
            #cv2.imwrite(str(back_side_right_image_path )+str(img_ind)+"."+args.image_ext,img_opencv_right)
            seq_dict['frame_number'] = str(
                image_actual_name) + "." + args.image_ext
            seq_dict['keyangle'] = keyangle_dict
            # if args.video_available == '0':
            seq_dict['pose'] = alignment
            seq_list.append(seq_dict)
            # print(seq_list)

        if alignment == "front_side":  # or alignment=="front_side":
            # print("front_side_loop")

            # knee angle calculation
            # create 5 directories for backside
            # print(" the value of i is ", i)

            if front_folder_created == False:
                front_right_hip_knee_ankle_path = str(
                    args.output_dir) + "/frontside/" + "right_hip_knee_ankle/"
                front_left_hip_knee_ankle_path = str(
                    args.output_dir) + "/frontside/" + "left_hip_knee_ankle/"
                front_right_knee_hip_vertical_path = str(
                    args.output_dir) + "/frontside/" + "right_knee_hip_vertical/"
                front_left_knee_hip_vertical_path = str(
                    args.output_dir) + "/frontside/" + "left_knee_hip_vertical/"
                front_left_hip_right_hip_pelvis_path = str(
                    args.output_dir) + "/frontside/" + "left_hip_right_hip_pelvis/"
                #front_right_hip_left_hip_pelvis_path=str(args.output_dir) +"frontside/"+"right_hip_left_hip_pelvis/"
                front_trunk_lean_path = str(
                    args.output_dir) + "/frontside/" + "trunk_lean/"
                front_left_image_path = str(
                    args.output_dir) + "/frontside/" + "left/" + "images/"
                front_right_image_path = str(
                    args.output_dir) + "/frontside/" + "right/" + "images/"

            # use shell script to build the paths
                # os.makedirs(front_right_hip_knee_ankle_path);os.makedirs(front_left_hip_knee_ankle_path);os.makedirs(front_right_knee_hip_vertical_path)
                # os.makedirs(front_left_knee_hip_vertical_path);os.makedirs(front_left_hip_right_hip_pelvis_path);os.makedirs(front_trunk_lean_path);
                os.makedirs(front_left_image_path)
                os.makedirs(front_right_image_path)
                front_folder_created = True

            img_opencv_right = cv2.imread(str(im_list[i]))
            from waistline_modules import waistline
            _,waistline_angle =  waistline.draw_waistline_front_back(img_opencv_right, pts=relevant_keypoint, clt=waistline_color_cluster,
                                                                     angle_font_params=(font,fontScale,fontColor,lineType))


            img_opencv_left = img_opencv_right.copy()

            # image_actual_name=str(im_list[i]).split("/")[-1][:-4]
            if True:
                if True:
                    if True:

                        # for back_right_hip_knee_ankle
                        if len(right_hip) > 0 and len(right_knee) > 0 and len(right_ankle) > 0 and len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0:
                            draw_front_back_line(left_shoulder, right_shoulder, left_hip, right_hip, left_knee,
                                                 right_knee, left_ankle, right_ankle, img_opencv_left, img_opencv_right, args.pelvis)

                        if len(right_hip) > 0 and len(right_knee) > 0 and len(right_ankle) > 0:
                            knee_point_dir = knee_point_dir_hip_ankle_line(
                                right_hip, right_knee, right_ankle)
                            if knee_point_dir >= 0:
                                write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                    right_hip, right_knee, right_ankle, front_right_hip_knee_ankle_path, img_opencv_right, "Knee Varus/Valgus", image_actual_name, -1, 'right')
                            if knee_point_dir < 0:
                                write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                    right_hip, right_knee, right_ankle, front_right_hip_knee_ankle_path, img_opencv_right, "Knee Varus/Valgus", image_actual_name, 1, 'right')
                            left_knee_ymax.append(left_knee[1])
                            write_image_front_right_hip_knee_ankle.append(
                                write_image)
                            all_front_right_knee_leg_angle.append(angle)
                            keyangle_dict['right_knee_varus_valgus'] = round(
                                angle, 1)
                            all_front_right_knee_leg_angle_label.append(relevant_keypoint)

                        # for back_left_hip_knee_ankle
                        if len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0:
                            knee_point_dir = knee_point_dir_hip_ankle_line(
                                left_hip, left_knee, left_ankle)
                            if knee_point_dir >= 0:
                                write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                    left_hip, left_knee, left_ankle, front_left_hip_knee_ankle_path, img_opencv_left, "Knee Varus/Valgus", image_actual_name, 1, 'right')
                            if knee_point_dir < 0:
                                write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                    left_hip, left_knee, left_ankle, front_left_hip_knee_ankle_path, img_opencv_left, "Knee Varus/Valgus", image_actual_name, -1, 'right')
                            right_knee_ymax.append(right_knee[1])
                            write_image_front_left_hip_knee_ankle.append(
                                write_image)
                            all_front_left_knee_leg_angle.append(angle)
                            keyangle_dict['left_knee_varus_valgus'] = round(
                                angle, 1)
                            all_front_left_knee_leg_angle_label.append(relevant_keypoint)

                        # back_right_knee_hip_vertical
                        if len(right_knee) > 0 and len(right_hip) > 0:
                            if right_knee[0] >= right_hip[0]:
                                write_image, angle, img_opencv_right = calculate_knee_hip_vertical_front_back(
                                    right_knee, right_hip, front_right_knee_hip_vertical_path, img_opencv_right, "Femur Angle", image_actual_name, -1, 'right')
                            if right_knee[0] < right_hip[0]:
                                write_image, angle, img_opencv_right = calculate_knee_hip_vertical_front_back(
                                    right_knee, right_hip, front_right_knee_hip_vertical_path, img_opencv_right, "Femur Angle", image_actual_name, 1, 'right')
                            write_image_front_right_knee_hip_vertical.append(
                                write_image)
                            all_front_right_knee_hip_vertical.append(angle)
                            keyangle_dict['right_femur_angle'] = round(
                                angle, 1)
                            all_front_right_knee_hip_vertical_label.append(relevant_keypoint)

                        # back_left_knee_hip_vertical
                        if len(left_knee) > 0 and len(left_hip) > 0:
                            if left_knee[0] <= left_hip[0]:
                                write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                                    left_knee, left_hip, front_left_knee_hip_vertical_path, img_opencv_left, "Femur Angle", image_actual_name, -1, 'left')
                            if left_knee[0] > left_hip[0]:
                                write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                                    left_knee, left_hip, front_left_knee_hip_vertical_path, img_opencv_left, "Femur Angle", image_actual_name, 1, 'left')
                            write_image_front_left_knee_hip_vertical.append(
                                write_image)
                            all_front_left_knee_hip_vertical.append(angle)
                            keyangle_dict['left_femur_angle'] = round(angle, 1)
                            all_front_left_knee_hip_vertical_label.append(relevant_keypoint)

                        # back_left_hip_right_hip_pelvis_path
                        if len(left_hip) > 0 and len(right_hip) > 0 and args.pelvis == 'on':
                            if False:
                                if right_hip[1] >= left_hip[1]:
                                    write_image, angle, img_opencv_right = calculate_hip_pelvis_front_back(
                                        left_hip, right_hip, front_left_hip_right_hip_pelvis_path, img_opencv_right, "Pelvis Angle", image_actual_name, 1, 'left')
                                if right_hip[1] < left_hip[1]:
                                    write_image, angle, img_opencv_right = calculate_hip_pelvis_front_back(
                                        left_hip, right_hip, front_left_hip_right_hip_pelvis_path, img_opencv_right, "Pelvis Angle", image_actual_name, -1, 'left')

                                if right_hip[1] >= left_hip[1]:
                                    write_image, angle, img_opencv_left = calculate_hip_pelvis_front_back(
                                        left_hip, right_hip, front_left_hip_right_hip_pelvis_path, img_opencv_left, "Pelvis Angle", image_actual_name, 1, 'left')
                                if right_hip[1] < left_hip[1]:
                                    write_image, angle, img_opencv_left = calculate_hip_pelvis_front_back(
                                        left_hip, right_hip, front_left_hip_right_hip_pelvis_path, img_opencv_left, "Pelvis Angle", image_actual_name, -1, 'left')
                            write_image_front_left_hip_right_hip.append(
                                img_opencv_right)
                            all_front_left_hip_right_hip.append(waistline_angle)
                            keyangle_dict['pelvis_angle'] = round(waistline_angle, 1)

                        # calculate_trunk_lean_front_back(hip1,hip2,shoulder1,shoulder2,output_dir)
                        if len(left_hip) > 0 and len(right_hip) > 0 and len(left_shoulder) > 0 and len(right_shoulder) > 0:
                            hip_meanx = (left_hip[0] + right_hip[0]) / 2
                            shoulder_meanx = (
                                left_shoulder[0] + right_shoulder[0]) / 2
                            if shoulder_meanx <= hip_meanx:
                                write_image, angle, img_opencv_right = calculate_trunk_lean_front_back(
                                    left_hip, right_hip, left_shoulder, right_shoulder, front_trunk_lean_path, img_opencv_right, "Trunk Lateral Lean", image_actual_name, 1, 'right')
                            if shoulder_meanx > hip_meanx:
                                write_image, angle, img_opencv_right = calculate_trunk_lean_front_back(
                                    left_hip, right_hip, left_shoulder, right_shoulder, front_trunk_lean_path, img_opencv_right, "Trunk Lateral Lean", image_actual_name, -1, 'right')

                            if shoulder_meanx <= hip_meanx:
                                write_image, angle, img_opencv_left = calculate_trunk_lean_front_back(
                                    left_hip, right_hip, left_shoulder, right_shoulder, front_trunk_lean_path, img_opencv_left, "Trunk Lateral Lean", image_actual_name, 1, 'right')
                            if shoulder_meanx > hip_meanx:
                                write_image, angle, img_opencv_left = calculate_trunk_lean_front_back(
                                    left_hip, right_hip, left_shoulder, right_shoulder, front_trunk_lean_path, img_opencv_left, "Trunk Lateral Lean", image_actual_name, -1, 'right')
                            write_image_front_trunk_lean.append(write_image)
                            all_front_trunk_lean.append(angle)
                            keyangle_dict['trunk_lateral_lean'] = round(
                                angle, 1)
                            all_front_trunk_lean_label.append(relevant_keypoint)

                        # Calculate hip ankle distance front side
                        if len(right_hip) > 0 and len(left_ankle) > 0 and len(right_ankle) > 0:
                            left_ankle_distx, left_ankle_disty, left_ankle_distxy = calculate_hip_ankle_distance_pixel(
                                right_hip, left_ankle)
                            right_ankle_distx, right_ankle_disty, right_ankle_distxy = calculate_hip_ankle_distance_pixel(
                                right_hip, right_ankle)
                            all_frontview_hip_leftankle_distance_y.append(
                                left_ankle_disty)
                            all_frontview_hip_rightankle_distance_y.append(
                                right_ankle_disty)

                        hip_mean = []
                        hip_mean.append(hip_meanx)
                        hip_mean.append((left_hip[1] + right_hip[1]) / 2)
                        shoulder_mean = []
                        shoulder_mean.append(shoulder_meanx)
                        shoulder_mean.append(
                            (left_shoulder[1] + right_shoulder[1]) / 2)
                        all_front_left_knee_coor.append(left_knee)
                        all_front_left_ankle_coor.append(left_ankle)
                        all_front_right_knee_coor.append(right_knee)
                        all_front_right_ankle_coor.append(right_ankle)
                        all_front_hip_coor.append(hip_mean)
                        all_front_shoulder_coor.append(shoulder_mean)

                        # hip coordinates left and right
                        hip_coor_left_right = []
                        hip_coor_left_right.append(right_hip[0])
                        hip_coor_left_right.append((right_hip[1]+right_shoulder[1])/2)
                        hip_coor_left_right.append(left_hip[0])
                        hip_coor_left_right.append(left_hip[1])
                        all_front_hip_coor_left_right.append(hip_coor_left_right)
                        # print("mamoon")
                        # print("values: ",left_knee, left_ankle,right_knee, right_ankle, hip_mean, shoulder_mean)

            cv2.imwrite(str(front_left_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_left)
            cv2.imwrite(str(front_right_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_right)
            # print('image saved is ---', image_actual_name,
            #     '-- with value of i is ----', i)
            # print('full path of image saved is -------', str(front_left_image_path)
            #     + str(image_actual_name) + "." + args.image_ext)
            # print('full path of image saved is -------', str(front_right_image_path)
            #     + str(image_actual_name) + "." + args.image_ext)

            #cv2.imwrite(str(front_left_image_path )+str(img_ind)+"."+args.image_ext,img_opencv_left)
            #cv2.imwrite(str(front_right_image_path )+str(img_ind)+"."+args.image_ext,img_opencv_right)
            seq_dict['frame_number'] = str(
                image_actual_name) + "." + args.image_ext
            seq_dict['keyangle'] = keyangle_dict
            # if args.video_available == '0':
            seq_dict['pose'] = alignment
            seq_list.append(seq_dict)

        if alignment == "left_side":  # or alignment=="front_side":
            # print(" the value of i is ", i)

            if left_folder_created == False:
                leftview_right_hip_knee_ankle_path = str(
                    args.output_dir) + "/leftviewside/" + "right_hip_knee_ankle/"
                leftview_left_hip_knee_ankle_path = str(
                    args.output_dir) + "/leftviewside/" + "left_hip_knee_ankle/"
                leftview_right_knee_hip_vertical_path = str(
                    args.output_dir) + "/leftviewside/" + "right_knee_hip_vertical/"
                leftview_left_knee_hip_vertical_path = str(
                    args.output_dir) + "/leftviewside/" + "left_knee_hip_vertical/"
                leftview_trunk_lean_path = str(
                    args.output_dir) + "/leftviewside/" + "trunk_lean/"
                leftview_left_knee_ankle_vertical_path = str(
                    args.output_dir) + "/leftviewside/" + "left_knee_ankle_vertical/"
                leftview_right_knee_ankle_vertical_path = str(
                    args.output_dir) + "/leftviewside/" + "right_knee_ankle_vertical/"
                leftview_left_ankle_hip_path = str(
                    args.output_dir) + "/leftviewside/" + "left_ankle_hip_dist/"
                leftview_right_ankle_hip_path = str(
                    args.output_dir) + "/leftviewside/" + "right_ankle_hip_dist/"
                leftview_left_image_path = str(args.output_dir) + \
                    "/leftviewside/" + "left/" + "images/"
                leftview_right_image_path = str(args.output_dir) + \
                    "/leftviewside/" + "right/" + "images/"
            # use sh
            # use shell script to build the paths
                # os.makedirs(leftview_right_hip_knee_ankle_path);os.makedirs(leftview_left_hip_knee_ankle_path);os.makedirs(leftview_right_knee_hip_vertical_path)
                # os.makedirs(leftview_left_knee_hip_vertical_path);os.makedirs(leftview_trunk_lean_path)
                # os.makedirs(leftview_left_knee_ankle_vertical_path);os.makedirs(leftview_right_knee_ankle_vertical_path)
                # os.makedirs(leftview_left_ankle_hip_path);os.makedirs(leftview_right_ankle_hip_path)
                os.makedirs(leftview_left_image_path)
                os.makedirs(leftview_right_image_path)
                left_folder_created = True

            img_opencv_left = cv2.imread(str(im_list[i]))
            from waistline_modules import waistline
            _,waistline_angle =  waistline.draw_waistline_side(img_opencv_left, pts=relevant_keypoint, clt=waistline_color_cluster,
                                                                 angle_font_params=(font,fontScale,fontColor,lineType))


            img_opencv_right = img_opencv_left.copy()
            # image_actual_name=str(im_list[i]).split("/")[-1][:-4]
            leftside_pass_flag = False
            if (False):
                leftside_pass_flag = True
            else:
                if False:
                    leftside_pass_flag = True
                else:
                    if len(right_hip) > 0 and len(right_knee) > 0 and len(right_ankle) > 0 and len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0 and left_shoulder[1] < left_hip[1] and left_hip[1] < left_ankle[1] and left_hip[1] < right_ankle[1] and left_hip[1] < right_knee[1] and left_hip[1] < left_knee[1]:
                        leftview_5frame_rightknee_ankle_distance[img_ind % 5] = calculate_euclidean_distance(
                            right_knee, right_ankle)
                        leftview_5frame_leftknee_ankle_distance[img_ind % 5] = calculate_euclidean_distance(
                            left_knee, left_ankle)
                        if img_ind >= 5 and check_valid_ankle_point(right_knee, right_anklkle, leftview_5frame_rightknee_ankle_distance) == 'notvalid' and check_valid_ankle_point(left_knee, left_ankle, leftview_5frame_leftknee_ankle_distance) == 'notvalid':
                            leftside_pass_flag = True
                        if leftside_pass_flag == True:
                            cv2.imwrite(str(leftview_left_image_path)
                                        + str(image_actual_name) + "." + args.image_ext, img_opencv_left)
                            cv2.imwrite(str(leftview_right_image_path)
                                        + str(image_actual_name) + "." + args.image_ext, img_opencv_right)
                            # print('image saved is ---', image_actual_name,
                            #         '-- with value of i is ----', i)
                            # print('full path of image saved is -------', str(leftview_left_image_path)
                            #         + str(image_actual_name) + "." + args.image_ext)
                            # print('full path of image saved is -------', str(leftview_right_image_path)
                            #         + str(image_actual_name) + "." + args.image_ext)
                            seq_dict['frame_number'] = str(
                                image_actual_name) + "." + args.image_ext
                            seq_dict['keyangle'] = keyangle_dict
                            # if args.video_available == '0':
                            seq_dict['pose'] = alignment
                            seq_list.append(seq_dict)
                            continue
                        # left_hip[0]=(left_hip[0]+right_hip[0])/2
                        # left_hip[1]=(left_hip[1]+right_hip[1])/2
                        right_hip[0] = (left_hip[0] + right_hip[0]) / 2
                        right_hip[1] = (left_hip[1] + right_hip[1]) / 2
                        # knee angle calculation
                        # create 5 directories for backside

                        # back_right_knee_hip_vertical
                        if len(right_hip) > 0 and len(right_knee) > 0 and len(right_ankle) > 0 and len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0:
                            draw_left_right_line(left_shoulder, right_shoulder, left_hip, right_hip, left_knee,
                                                 right_knee, left_ankle, right_ankle, img_opencv_left, img_opencv_right, "left")

                        if len(right_knee) > 0 and len(right_hip) > 0:
                            if right_knee[0] >= right_hip[0]:
                                write_image, angle, img_opencv_right = calculate_knee_hip_vertical_front_back(
                                    right_knee, right_hip, leftview_right_knee_hip_vertical_path, img_opencv_right, "Hip Angle", image_actual_name, -1, 'right')
                            if right_knee[0] < right_hip[0]:
                                write_image, angle, img_opencv_right = calculate_knee_hip_vertical_front_back(
                                    right_knee, right_hip, leftview_right_knee_hip_vertical_path, img_opencv_right, "Hip Angle", image_actual_name, 1, 'right')
                            write_image_leftview_right_knee_hip_vertical.append(
                                write_image)
                            all_leftview_right_knee_hip_vertical.append(angle)
                            keyangle_dict['right_hip_angle'] = round(angle, 1)

                        # back_left_knee_hip_vertical
                        if len(left_knee) > 0 and len(left_hip) > 0:
                            if left_knee[0] >= left_hip[0]:
                                write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                                    left_knee, left_hip, leftview_left_knee_hip_vertical_path, img_opencv_left, "Hip Angle", image_actual_name, -1, 'right')
                            if left_knee[0] < left_hip[0]:
                                write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                                    left_knee, left_hip, leftview_left_knee_hip_vertical_path, img_opencv_left, "Hip Angle", image_actual_name, 1, 'right')
                            write_image_leftview_left_knee_hip_vertical.append(
                                write_image)
                            all_leftview_left_knee_hip_vertical.append(angle)
                            keyangle_dict['left_hip_angle'] = round(angle, 1)

                        # calculate_trunk_lean_left_right
                        if len(left_hip) > 0 and len(left_shoulder) > 0 and len(right_hip) > 0 and len(right_shoulder) > 0:
                            if right_hip[0] >= left_shoulder[0]:
                                write_image, angle, img_opencv_right = calculate_trunk_lean_left_right(
                                    right_hip, left_shoulder, leftview_trunk_lean_path, img_opencv_right, "Trunk Forward Lean", image_actual_name, 1, 'right')
                            if right_hip[0] < left_shoulder[0]:
                                write_image, angle, img_opencv_right = calculate_trunk_lean_left_right(
                                    right_hip, left_shoulder, leftview_trunk_lean_path, img_opencv_right, "Trunk Forward Lean", image_actual_name, -1, 'right')
                            if left_hip[0] >= left_shoulder[0]:
                                write_image, angle, img_opencv_left = calculate_trunk_lean_left_right(
                                    left_hip, left_shoulder, leftview_trunk_lean_path, img_opencv_left, "Trunk Forward Lean", image_actual_name, 1, 'right')
                            if left_hip[0] < left_shoulder[0]:
                                write_image, angle, img_opencv_left = calculate_trunk_lean_left_right(
                                    left_hip, left_shoulder, leftview_trunk_lean_path, img_opencv_left, "Trunk Forward Lean", image_actual_name, -1, 'right')
                            write_image_leftview_trunk_lean.append(write_image)
                            all_leftview_trunk_lean.append(angle)
                            keyangle_dict['trunk_forward_lean'] = round(
                                angle, 1)

                        # calculate left knee ankle vertical angle

                        if len(left_knee) > 0 and len(left_ankle) > 0:
                            if left_ankle[0] >= left_knee[0]:
                                write_image, angle, img_opencv_left = calculate_knee_ankle_vertical(
                                    left_knee, left_ankle, leftview_left_knee_ankle_vertical_path, img_opencv_left, "Tibial Angle", image_actual_name, 1, 'right')
                            if left_ankle[0] < left_knee[0]:
                                write_image, angle, img_opencv_left = calculate_knee_ankle_vertical(
                                    left_knee, left_ankle, leftview_left_knee_ankle_vertical_path, img_opencv_left, "Tibial Angle", image_actual_name, -1, 'right')
                                write_image_leftview_left_knee_ankle_vertical.append(
                                    write_image)
                            all_leftview_left_knee_ankle_vertical_angle.append(
                                angle)
                            keyangle_dict['left_tibial_angle'] = round(
                                angle, 1)

                        # calculate right knee ankle vertical angle
                        if len(right_knee) > 0 and len(right_ankle) > 0:
                            if right_ankle[0] >= right_knee[0]:
                                write_image, angle, img_opencv_right = calculate_knee_ankle_vertical(
                                    right_knee, right_ankle, leftview_right_knee_ankle_vertical_path, img_opencv_right, "Tibial Angle", image_actual_name, 1, 'right')
                            if right_ankle[0] < right_knee[0]:
                                write_image, angle, img_opencv_right = calculate_knee_ankle_vertical(
                                    right_knee, right_ankle, leftview_right_knee_ankle_vertical_path, img_opencv_right, "Tibial Angle", image_actual_name, -1, 'right')
                            write_image_leftview_right_knee_ankle_vertical.append(
                                write_image)
                            all_leftview_right_knee_ankle_vertical_angle.append(
                                angle)
                            keyangle_dict['right_tibial_angle'] = round(
                                angle, 1)

                        # calculate hip to right ankle distance
                        if len(left_hip) > 0 and len(left_hip) > 0 and len(right_ankle) > 0:
                            if right_ankle[0] >= left_hip[0]:
                                write_image, angle, img_opencv_right = calculate_hip_ankle_distance_left_right(
                                    left_hip, left_hip, right_ankle, leftview_right_ankle_hip_path, img_opencv_right, "Hip Ankle dist(cm)", image_actual_name, -1, 'left')
                            if right_ankle[0] < left_hip[0]:
                                write_image, angle, img_opencv_right = calculate_hip_ankle_distance_left_right(
                                    left_hip, left_hip, right_ankle, leftview_right_ankle_hip_path, img_opencv_right, "Hip Ankle dist(cm)", image_actual_name, -1, 'right')
                            keyangle_dict['right_hip_ankle_dist'] = round(
                                angle, 1)
                            all_leftview_right_hip_ankle_dist.append(angle)

                        # calculate hip to left ankle distance
                        if len(right_hip) > 0 and len(left_hip) > 0 and len(left_ankle) > 0:
                            if left_ankle[0] >= left_hip[0]:
                                write_image, angle, img_opencv_left = calculate_hip_ankle_distance_left_right(
                                    left_hip, left_hip, left_ankle, leftview_left_ankle_hip_path, img_opencv_left, "Hip Ankle dist(cm)", image_actual_name, -1, 'right')
                            if left_ankle[0] < left_hip[0]:
                                write_image, angle, img_opencv_left = calculate_hip_ankle_distance_left_right(
                                    left_hip, left_hip, left_ankle, leftview_left_ankle_hip_path, img_opencv_left, "Hip Ankle dist(cm)", image_actual_name, -1, 'right')
                            keyangle_dict['left_hip_ankle_dist'] = round(
                                angle, 1)
                            all_leftview_left_hip_ankle_dist.append(angle)

                        # for back_right_hip_knee_ankle
                        if len(right_hip) > 0 and len(right_knee) > 0:
                            angle = 180 - calc_angle(np.array(right_hip),
                                                     np.array(right_knee), np.array(right_ankle))
                            if angle >= 0:
                                if right_knee[0] >= right_hip[0]:
                                    write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                        right_hip, right_knee, right_ankle, leftview_right_hip_knee_ankle_path, img_opencv_right, "Knee Angle", image_actual_name, 1, 'right')
                                if right_knee[0] < right_hip[0]:
                                    write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                        right_hip, right_knee, right_ankle, leftview_right_hip_knee_ankle_path, img_opencv_right, "Knee Angle", image_actual_name, 1, 'right')
                            if angle < 0:
                                if right_knee[0] >= right_hip[0]:
                                    write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                        right_hip, right_knee, right_ankle, leftview_right_hip_knee_ankle_path, img_opencv_right, "Knee Angle", image_actual_name, -1, 'right')
                                if right_knee[0] < right_hip[0]:
                                    write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                        right_hip, right_knee, right_ankle, leftview_right_hip_knee_ankle_path, img_opencv_right, "Knee Angle", image_actual_name, -1, 'right')
                            write_image_leftview_right_hip_knee_ankle.append(
                                write_image)
                            all_leftview_right_knee_leg_angle.append(angle)
                            keyangle_dict['right_knee_angle'] = round(angle, 1)

                        # for back_left_hip_knee_ankle
                        if len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0:
                            angle = 180 - calc_angle(np.array(left_hip),
                                                     np.array(left_knee), np.array(left_ankle))
                            if angle >= 0:
                                if left_knee[0] >= left_hip[0]:
                                    write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                        left_hip, left_knee, left_ankle, leftview_left_hip_knee_ankle_path, img_opencv_left, "Knee Angle", image_actual_name, 1, 'right')
                                if left_knee[0] < left_hip[0]:
                                    write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                        left_hip, left_knee, left_ankle, leftview_left_hip_knee_ankle_path, img_opencv_left, "Knee Angle", image_actual_name, 1, 'right')
                            if angle < 0:
                                if left_knee[0] >= left_hip[0]:
                                    write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                        left_hip, left_knee, left_ankle, leftview_left_hip_knee_ankle_path, img_opencv_left, "Knee Angle", image_actual_name, -1, 'right')
                                if left_knee[0] < left_hip[0]:
                                    write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                        left_hip, left_knee, left_ankle, leftview_left_hip_knee_ankle_path, img_opencv_left, "Knee Angle", image_actual_name, -1, 'right')
                            write_image_leftview_left_hip_knee_ankle.append(
                                write_image)
                            all_leftview_left_knee_leg_angle.append(angle)
                            keyangle_dict['left_knee_angle'] = round(angle, 1)

                        if len(left_hip) > 0 and len(left_ankle) > 0 and len(right_ankle) > 0:
                            left_ankle_distx, left_ankle_disty, left_ankle_distxy = calculate_hip_ankle_distance_pixel(
                                left_hip, left_ankle)
                            right_ankle_distx, right_ankle_disty, right_ankle_distxy = calculate_hip_ankle_distance_pixel(
                                left_hip, right_ankle)
                            all_leftview_hip_leftankle_distance_xy.append(
                                left_ankle_distxy)
                            all_leftview_hip_leftankle_distance_x.append(
                                left_ankle_distx)
                            all_leftview_hip_leftankle_distance_y.append(
                                left_ankle_disty)

                            all_leftview_hip_rightankle_distance_xy.append(
                                right_ankle_distxy)
                            all_leftview_hip_rightankle_distance_x.append(
                                right_ankle_distx)
                            all_leftview_hip_rightankle_distance_y.append(
                                right_ankle_disty)

                            all_leftview_left_hip_coord.append(left_hip)
                            all_leftview_left_shoulder_coord.append(
                                left_shoulder)
                            all_leftview_left_knee_coord.append(left_knee)
                            all_leftview_left_ankle_coord.append(left_ankle)
                            all_leftview_right_knee_coord.append(right_knee)
                            all_leftview_right_ankle_coord.append(right_ankle)
                            # all_leftview_right_shoulder_coord.append(right_shoulder)
                        img_ind = img_ind + 1

            cv2.imwrite(str(leftview_left_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_left)
            cv2.imwrite(str(leftview_right_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_right)
            # print('image saved is ---', image_actual_name,
            #     '-- with value of i is ----', i)
            # print('full path of image saved is -------', str(leftview_left_image_path)
            #     + str(image_actual_name) + "." + args.image_ext)
            # print('full path of image saved is -------', str(leftview_right_image_path)
            #     + str(image_actual_name) + "." + args.image_ext)

            seq_dict['frame_number'] = str(
                image_actual_name) + "." + args.image_ext
            seq_dict['keyangle'] = keyangle_dict
            # if args.video_available == '0':
            seq_dict['pose'] = alignment
            seq_list.append(seq_dict)

        if alignment == "right_side":  # or alignment=="front_side":

            if right_folder_created == False:
                rightview_right_hip_knee_ankle_path = str(
                    args.output_dir) + "/rightviewside/" + "right_hip_knee_ankle/"
                rightview_left_hip_knee_ankle_path = str(
                    args.output_dir) + "/rightviewside/" + "left_hip_knee_ankle/"
                rightview_right_knee_hip_vertical_path = str(
                    args.output_dir) + "/rightviewside/" + "right_knee_hip_vertical/"
                rightview_left_knee_hip_vertical_path = str(
                    args.output_dir) + "/rightviewside/" + "left_knee_hip_vertical/"
                rightview_trunk_lean_path = str(
                    args.output_dir) + "/rightviewside/" + "trunk_lean/"
                rightview_left_knee_ankle_vertical_path = str(
                    args.output_dir) + "/rightviewside/" + "left_knee_ankle_vertical/"
                rightview_right_knee_ankle_vertical_path = str(
                    args.output_dir) + "/rightviewside/" + "right_knee_ankle_vertical/"
                rightview_left_ankle_hip_path = str(
                    args.output_dir) + "/rightviewside/" + "left_ankle_hip_dist/"
                rightview_right_ankle_hip_path = str(
                    args.output_dir) + "/rightviewside/" + "right_ankle_hip_dist/"
                rightview_left_image_path = str(args.output_dir) + \
                    "/rightviewside/" + "left/" + "images/"
                rightview_right_image_path = str(args.output_dir) + \
                    "/rightviewside/" + "right/" + "images/"

            # use shell script to build the paths
                # os.makedirs(rightview_right_hip_knee_ankle_path);os.makedirs(rightview_left_hip_knee_ankle_path);os.makedirs(rightview_right_knee_hip_vertical_path)
                # os.makedirs(rightview_left_knee_hip_vertical_path);os.makedirs(rightview_trunk_lean_path)
                # os.makedirs(rightview_left_knee_ankle_vertical_path);os.makedirs(rightview_right_knee_ankle_vertical_path)
                # os.makedirs(rightview_left_ankle_hip_path);os.makedirs(rightview_right_ankle_hip_path)
                os.makedirs(rightview_left_image_path)
                os.makedirs(rightview_right_image_path)
                right_folder_created = True

            img_opencv_left = cv2.imread(str(im_list[i]))
            from waistline_modules import waistline
            _,waistline_angle =  waistline.draw_waistline_side(img_opencv_left, pts=relevant_keypoint, clt=waistline_color_cluster,
                                                                     angle_font_params=(font,fontScale,fontColor,lineType))


            img_opencv_right = img_opencv_left.copy()
            # image_actual_name=str(im_list[i]).split("/")[-1][:-4]
            rightside_pass_flag = False
            if (False):
                rightside_pass_flag = True
            else:
                if False:
                    rightside_pass_flag = True
                else:
                    if len(right_hip) > 0 and len(right_knee) > 0 and len(right_ankle) > 0 and len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0 and right_shoulder[1] < right_hip[1] and right_hip[1] < right_ankle[1] and right_hip[1] < left_ankle[1] and right_hip[1] < right_knee[1] and right_hip[1] < left_knee[1]:

                        rightview_5frame_rightknee_ankle_distance[img_ind % 5] = calculate_euclidean_distance(
                            right_knee, right_ankle)
                        rightview_5frame_leftknee_ankle_distance[img_ind % 5] = calculate_euclidean_distance(
                            left_knee, left_ankle)
                        if img_ind >= 5 and check_valid_ankle_point(right_knee, right_ankle, rightview_5frame_rightknee_ankle_distance) == 'notvalid' and check_valid_ankle_point(left_knee, left_ankle, rightview_5frame_leftknee_ankle_distance) == 'notvalid':
                            rightside_pass_flag = True

                        if rightside_pass_flag == True:
                            cv2.imwrite(str(rightview_left_image_path)
                                        + str(image_actual_name) + "." + args.image_ext, img_opencv_left)
                            cv2.imwrite(str(rightview_right_image_path)
                                        + str(image_actual_name) + "." + args.image_ext, img_opencv_right)
                            print('image saved is ---', image_actual_name,
                                  '-- with value of i is ----', i)
                            print('full path of image saved is -------', str(rightview_left_image_path)
                                  + str(image_actual_name) + "." + args.image_ext)
                            print('full path of image saved is -------', str(rightview_right_image_path)
                                  + str(image_actual_name) + "." + args.image_ext)
                            seq_dict['frame_number'] = str(
                                image_actual_name) + "." + args.image_ext
                            seq_dict['keyangle'] = keyangle_dict
                            # if args.video_available == '0':
                            seq_dict['pose'] = alignment
                            seq_list.append(seq_dict)
                            continue
                        # right_hip[0]=(left_hip[0]+right_hip[0])/2
                        # right_hip[1]=(left_hip[1]+right_hip[1])/2
                        left_hip[0] = (left_hip[0] + right_hip[0]) / 2
                        left_hip[1] = (left_hip[1] + right_hip[1]) / 2
                        # knee angle calculation
                        # create 5 directories for backside
                        # print(" the value of i is ", i)
                        all_is.append(i)

                        # back_right_knee_hip_vertical
                        if len(right_hip) > 0 and len(right_knee) > 0 and len(right_ankle) > 0 and len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0:
                            draw_left_right_line(left_shoulder, right_shoulder, left_hip, right_hip, left_knee,
                                                 right_knee, left_ankle, right_ankle, img_opencv_left, img_opencv_right, "right")

                        if len(right_knee) > 0 and len(right_hip) > 0:
                            if right_knee[0] >= right_hip[0]:
                                write_image, angle, img_opencv_right = calculate_knee_hip_vertical_front_back(
                                    right_knee, right_hip, rightview_right_knee_hip_vertical_path, img_opencv_right, "Hip Angle", image_actual_name, 1, 'right')
                            if right_knee[0] < right_hip[0]:
                                write_image, angle, img_opencv_right = calculate_knee_hip_vertical_front_back(
                                    right_knee, right_hip, rightview_right_knee_hip_vertical_path, img_opencv_right, "Hip Angle", image_actual_name, -1, 'right')
                            write_image_rightview_right_knee_hip_vertical.append(
                                write_image)
                            all_rightview_right_knee_hip_vertical.append(angle)
                            keyangle_dict['right_hip_angle'] = round(angle, 1)

                        # back_left_knee_hip_vertical
                        if len(left_knee) > 0 and len(left_hip) > 0:
                            if left_knee[0] >= left_hip[0]:
                                write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                                    left_knee, left_hip, rightview_left_knee_hip_vertical_path, img_opencv_left, "Hip Angle", image_actual_name, 1, 'right')
                            if left_knee[0] < left_hip[0]:
                                write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                                    left_knee, left_hip, rightview_left_knee_hip_vertical_path, img_opencv_left, "Hip Angle", image_actual_name, -1, 'right')
                            write_image_rightview_left_knee_hip_vertical.append(
                                write_image)
                            all_rightview_left_knee_hip_vertical.append(angle)
                            keyangle_dict['left_hip_angle'] = round(angle, 1)

                        # calculate_trunk_lean_left_right
                        if len(right_hip) > 0 and len(right_shoulder) > 0 and len(left_hip) > 0 and len(left_shoulder) > 0:
                            if right_hip[0] <= right_shoulder[0]:
                                write_image, angle, img_opencv_right = calculate_trunk_lean_left_right(
                                    right_hip, right_shoulder, rightview_trunk_lean_path, img_opencv_right, "Trunk Forward Lean", image_actual_name, 1, 'right')
                            if right_hip[0] > right_shoulder[0]:
                                write_image, angle, img_opencv_right = calculate_trunk_lean_left_right(
                                    right_hip, right_shoulder, rightview_trunk_lean_path, img_opencv_right, "Trunk Forward Lean", image_actual_name, -1, 'right')
                            if left_hip[0] <= right_shoulder[0]:
                                write_image, angle, img_opencv_left = calculate_trunk_lean_left_right(
                                    left_hip, right_shoulder, rightview_trunk_lean_path, img_opencv_left, "Trunk Forward Lean", image_actual_name, 1, 'right')
                            if left_hip[0] > right_shoulder[0]:
                                write_image, angle, img_opencv_left = calculate_trunk_lean_left_right(
                                    left_hip, right_shoulder, rightview_trunk_lean_path, img_opencv_left, "Trunk Forward Lean", image_actual_name, -1, 'right')
                            write_image_rightview_trunk_lean.append(
                                write_image)
                            all_rightview_trunk_lean.append(angle)
                            keyangle_dict['trunk_forward_lean'] = round(
                                angle, 1)

                        # calculate left knee ankle vertical angle
                        if len(left_knee) > 0 and len(left_ankle) > 0:
                            if left_ankle[0] >= left_knee[0]:
                                write_image, angle, img_opencv_left = calculate_knee_ankle_vertical(
                                    left_knee, left_ankle, rightview_left_knee_ankle_vertical_path, img_opencv_left, "Tibial Angle", image_actual_name, 1, 'right')
                            if left_ankle[0] < left_knee[0]:
                                write_image, angle, img_opencv_left = calculate_knee_ankle_vertical(
                                    left_knee, left_ankle, rightview_left_knee_ankle_vertical_path, img_opencv_left, "Tibial Angle", image_actual_name, -1, 'right')
                                write_image_rightview_left_knee_ankle_vertical.append(
                                    write_image)
                            all_rightview_left_knee_ankle_vertical_angle.append(
                                angle)
                            keyangle_dict['left_tibial_angle'] = round(
                                angle, 1)

                        # calculate right knee ankle vertical angle
                        if len(right_knee) > 0 and len(right_ankle) > 0:
                            if right_ankle[0] >= right_knee[0]:
                                write_image, angle, img_opencv_right = calculate_knee_ankle_vertical(
                                    right_knee, right_ankle, rightview_right_knee_ankle_vertical_path, img_opencv_right, "Tibial Angle", image_actual_name, 1, 'right')
                            if right_ankle[0] < right_knee[0]:
                                write_image, angle, img_opencv_right = calculate_knee_ankle_vertical(
                                    right_knee, right_ankle, rightview_right_knee_ankle_vertical_path, img_opencv_right, "Tibial Angle", image_actual_name, -1, 'right')
                                write_image_rightview_right_knee_ankle_vertical.append(
                                    write_image)
                            all_rightview_right_knee_ankle_vertical_angle.append(
                                angle)
                            keyangle_dict['right_tibial_angle'] = round(
                                angle, 1)

                        # calculate hip to right ankle distance
                        if len(right_hip) > 0 and len(left_hip) > 0 and len(right_ankle) > 0:
                            if right_ankle[0] >= right_hip[0]:
                                write_image, angle, img_opencv_right = calculate_hip_ankle_distance_left_right(
                                    right_hip, right_hip, right_ankle, rightview_right_ankle_hip_path, img_opencv_right, "Hip Ankle dist(cm)", image_actual_name, 1, 'right')
                            if right_ankle[0] < right_hip[0]:
                                write_image, angle, img_opencv_right = calculate_hip_ankle_distance_left_right(
                                    right_hip, right_hip, right_ankle, rightview_right_ankle_hip_path, img_opencv_right, "Hip Ankle dist(cm)", image_actual_name, 1, 'right')
                            keyangle_dict['right_hip_ankle_dist'] = round(
                                angle, 1)
                            all_rightview_right_hip_ankle_dist.append(angle)

                        # calculate hip to left ankle distance
                        if len(right_hip) > 0 and len(left_hip) > 0 and len(left_ankle) > 0:
                            if left_ankle[0] >= right_hip[0]:
                                write_image, angle, img_opencv_left = calculate_hip_ankle_distance_left_right(
                                    right_hip, right_hip, left_ankle, rightview_left_ankle_hip_path, img_opencv_left, "Hip Ankle dist(cm)", image_actual_name, 1, 'right')
                            if left_ankle[0] < right_hip[0]:
                                write_image, angle, img_opencv_left = calculate_hip_ankle_distance_left_right(
                                    right_hip, right_hip, left_ankle, rightview_left_ankle_hip_path, img_opencv_left, "Hip Ankle dist(cm)", image_actual_name, 1, 'right')
                            keyangle_dict['left_hip_ankle_dist'] = round(
                                angle, 1)
                            all_rightview_left_hip_ankle_dist.append(angle)

                        if len(right_hip) > 0 and len(right_knee) > 0 and len(right_ankle) > 0:
                            angle = 180 - calc_angle(np.array(right_hip),
                                                     np.array(right_knee), np.array(right_ankle))
                            if angle >= 0:
                                if right_knee[0] >= right_hip[0]:
                                    write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                        right_hip, right_knee, right_ankle, rightview_right_hip_knee_ankle_path, img_opencv_right, "Knee Angle", image_actual_name, 1, 'right')
                                if right_knee[0] < right_hip[0]:
                                    write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                        right_hip, right_knee, right_ankle, rightview_right_hip_knee_ankle_path, img_opencv_right, "Knee Angle", image_actual_name, 1, 'right')
                            if angle < 0:
                                if right_knee[0] >= right_hip[0]:
                                    write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                        right_hip, right_knee, right_ankle, rightview_right_hip_knee_ankle_path, img_opencv_right, "Knee Angle", image_actual_name, -1, 'right')
                                if right_knee[0] < right_hip[0]:
                                    write_image, angle, img_opencv_right = calculate_hip_knee_ankle_front_back(
                                        right_hip, right_knee, right_ankle, rightview_right_hip_knee_ankle_path, img_opencv_right, "Knee Angle", image_actual_name, -1, 'right')
                            write_image_rightview_right_hip_knee_ankle.append(
                                write_image)
                            all_rightview_right_knee_leg_angle.append(angle)
                            keyangle_dict['right_knee_angle'] = round(angle, 1)

                        # for back_left_hip_knee_ankle
                        if len(left_hip) > 0 and len(left_knee) > 0 and len(left_ankle) > 0:
                            angle = 180 - calc_angle(np.array(left_hip),
                                                     np.array(left_knee), np.array(left_ankle))
                            if angle >= 0:
                                if left_knee[0] >= left_hip[0]:
                                    write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                        left_hip, left_knee, left_ankle, rightview_left_hip_knee_ankle_path, img_opencv_left, "Knee Angle", image_actual_name, 1, 'right')
                                if left_knee[0] < left_hip[0]:
                                    write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                        left_hip, left_knee, left_ankle, rightview_left_hip_knee_ankle_path, img_opencv_left, "Knee Angle", image_actual_name, 1, 'right')
                            if angle < 0:
                                if left_knee[0] >= left_hip[0]:
                                    write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                        left_hip, left_knee, left_ankle, rightview_left_hip_knee_ankle_path, img_opencv_left, "Knee Angle", image_actual_name, -1, 'right')
                                if left_knee[0] < left_hip[0]:
                                    write_image, angle, img_opencv_left = calculate_hip_knee_ankle_front_back(
                                        left_hip, left_knee, left_ankle, rightview_left_hip_knee_ankle_path, img_opencv_left, "Knee Angle", image_actual_name, -1, 'right')
                            write_image_rightview_left_hip_knee_ankle.append(
                                write_image)
                            all_rightview_left_knee_leg_angle.append(angle)
                            keyangle_dict['left_knee_angle'] = round(angle, 1)

                        if len(right_hip) > 0 and len(left_ankle) > 0 and len(right_ankle) > 0:
                            left_ankle_distx, left_ankle_disty, left_ankle_distxy = calculate_hip_ankle_distance_pixel(
                                right_hip, left_ankle)
                            right_ankle_distx, right_ankle_disty, right_ankle_distxy = calculate_hip_ankle_distance_pixel(
                                right_hip, right_ankle)
                            all_rightview_hip_leftankle_distance_xy.append(
                                left_ankle_distxy)
                            all_rightview_hip_leftankle_distance_x.append(
                                left_ankle_distx)
                            all_rightview_hip_leftankle_distance_y.append(
                                left_ankle_disty)
                            all_rightview_hip_rightankle_distance_xy.append(
                                right_ankle_distxy)
                            all_rightview_hip_rightankle_distance_x.append(
                                right_ankle_distx)
                            all_rightview_hip_rightankle_distance_y.append(
                                right_ankle_disty)

                            all_rightview_right_hip_coord.append(right_hip)
                            # all_rightview_left_shoulder_coord.append(left_shoulder)
                            all_rightview_right_shoulder_coord.append(
                                right_shoulder)
                            all_rightview_left_knee_coord.append(left_knee)
                            all_rightview_left_ankle_coord.append(left_ankle)
                            all_rightview_right_knee_coord.append(right_knee)
                            all_rightview_right_ankle_coord.append(right_ankle)


                            # pelvis tilt
                            right_pelvis_tilt = []
                            distance = int((right_knee[0] - right_ankle[0])/2)
                            x1 = right_shoulder[0] - distance
                            y1 = (right_shoulder[1] + right_hip[1])/2
                            x2 = right_hip[0] + distance
                            y2 = right_hip[1]
                            right_pelvis_tilt.append(x1)
                            right_pelvis_tilt.append(y1)
                            right_pelvis_tilt.append(x2)
                            right_pelvis_tilt.append(y2)
                            all_rightview_right_pelvis_tilt.append(right_pelvis_tilt)

                        img_ind = img_ind + 1
            cv2.imwrite(str(rightview_left_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_left)
            cv2.imwrite(str(rightview_right_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_right)
            # print('image saved is ---', image_actual_name,
            #     '-- with value of i is ----', i)
            # print('full path of image saved is -------', str(rightview_left_image_path)
            #     + str(image_actual_name) + "." + args.image_ext)
            # print('full path of image saved is -------', str(rightview_right_image_path)
            #     + str(image_actual_name) + "." + args.image_ext)

            seq_dict['frame_number'] = str(
                image_actual_name) + "." + args.image_ext
            seq_dict['keyangle'] = keyangle_dict
            # if args.video_available == '0':
            seq_dict['pose'] = alignment
            seq_list.append(seq_dict)
            # end_time = time.time()
            # print("////////////////////////////%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%////////""////////////////////")
            # print('model time: {:.3f}s'.format(end_time - start_time))
            # print("///////////////////////////%%%%%%%%%%%%%%%%%%%%%%%%%%//////////////////////////////////")
            # totaltime01 = end_time - start_time
            # times[i].append(totaltime01)

    #     postprocess3_end = time.time()
    #     totaltimeaftermodel1 = postprocess1_end - postprocess1_start
    #     post_time1.append(totaltimeaftermodel1)
    #     totaltimeaftermodel2 = postprocess2_end - postprocess2_start
    #     post_time2.append(totaltimeaftermodel2)
    #     totaltimeaftermodel3 = postprocess3_end - postprocess3_start
    #     post_time3.append(totaltimeaftermodel3)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print('1st post processing time: {:.3f}s'.format(postprocess1_end - postprocess1_start))
    # print('post processing time: {:.3f}s'.format(postprocess2_end - postprocess2_start))
    # print('post processing time: {:.3f}s'.format(postprocess3_end - postprocess3_start))
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # process_endtime = time.time()  # processing time ending here
    # print('***************************************************** processing time: {:.3f}s'.format(process_endtime - process_starttime))
    left_video_view_side = ""
    right_video__view_side = ""
    if alignment == "right_side":
        left_video_view_side = "rightviewside_leftextremity"
        right_video__view_side = "rightviewside_rightexteremity"
        out_put_image_dir = str(args.output_dir) + "rightviewside/"
        json_folder = str(args.output_dir) + \
            "rightviewside/" + video_name + ".json"
        optimal_json_name = str(args.output_dir) + "rightviewside/" + \
            video_name + "_optimalFrame.json"
    elif alignment == "left_side":
        left_video_view_side = "leftviewside_leftextremity"
        right_video__view_side = "leftviewside_rightexteremity"
        out_put_image_dir = str(args.output_dir) + "leftviewside/"
        json_folder = str(args.output_dir) + \
            "leftviewside/" + video_name + ".json"
        optimal_json_name = str(args.output_dir) + "leftviewside/" + \
            video_name + "_optimalFrame.json"
    if alignment == "front_side":
        left_video_view_side = "frontviewside_leftextremity"
        right_video__view_side = "frontviewside_rightexteremity"
        out_put_image_dir = str(args.output_dir) + "frontside/"
        json_folder = str(args.output_dir) + "frontside/" + \
            video_name + ".json"
        optimal_json_name = str(args.output_dir) + \
            "frontside/" + video_name + "_optimalFrame.json"
    elif alignment == "back_side":
        left_video_view_side = "backviewside_leftextremity"
        right_video__view_side = "backviewside_rightexteremity"
        out_put_image_dir = str(args.output_dir) + "backside/"
        json_folder = str(args.output_dir) + "backside/" + video_name + ".json"
        optimal_json_name = str(args.output_dir) + \
            "backside/" + video_name + "_optimalFrame.json"

    now_date_time = datetime.now()
    # dt_string = now_date_time.strftime("%m/%d/%Y_%H:%M:%S")
    dt_string = now_date_time.strftime('%B%d-%Y %I:%M %p')
    print("date and time =", dt_string)
    json_list['processed_date_time'] = dt_string
    json_list['video_date_time'] = creation_date_fun(args.video_path).strftime('%B%d-%Y %I:%M %p')

    json_list['video_path'] = args.video_path
    json_list['output_directory'] = args.output_dir
    json_list['input_type'] = args.input_type
    json_list['input_fps'] = args.input_fps
    json_list['image_extension'] = args.image_ext
    json_list['camera_distance_in_feet'] = args.cam_dist
    json_list['resolution'] = args.raw_video
    json_list['maximum_amount_of_frame'] = args.max_frame
    json_list['pose_detection'] = args.pose_detection
    if alignment == "back_side" or alignment == "front_side":
        json_list['pelvis'] = args.pelvis

    if args.input_type == 'image':
        json_list['sequence'] = seq_list
        # json.dump(json_list) #dump_json
        with open(str(args.output_dir) + video_name + ".json", 'w') as f:
            json.dump(json_list, f, ensure_ascii=False, indent=4)
        exit()
    optimal_frame_dict = {}
    optimal_frame_dict['video_name'] = video_name
    optimal_frame_dict['processed_date_time'] = dt_string
    optimal_frame_dict['video_date_time'] = creation_date_fun(
        args.video_path).strftime('%B%d-%Y %I:%M %p')
    optimal_frame_dict['video_path'] = args.video_path
    optimal_frame_dict['output_directory'] = args.output_dir
    optimal_frame_dict['input_type'] = args.input_type
    optimal_frame_dict['input_fps'] = args.input_fps
    optimal_frame_dict['image_extension'] = args.image_ext
    optimal_frame_dict['camera_distance_in_feet'] = args.cam_dist
    optimal_frame_dict['resolution'] = args.raw_video
    optimal_frame_dict['maximum_amount_of_frame'] = args.max_frame
    optimal_frame_dict['pose_detection'] = args.pose_detection

    optimal_frame_seq_list = []
    create_right_movie_dir = 'mkdir ' + \
        str(out_put_image_dir) + 'right/' + 'movie'
    os.system(create_right_movie_dir)
    create_left_movie_dir = 'mkdir ' + \
        str(out_put_image_dir) + 'left/' + 'movie'
    os.system(create_left_movie_dir)

    # 'Video_'+video_name + '_' +creation_date
    # video_full_path_left=""
    # video_full_path_right=""
    # if (args.output_option == 1 or args.output_option == 3):
    #     video_full_path_left=str(out_put_image_dir) + 'left/' + 'movie' + '/Video_'+left_video_view_side + '_' + video_name + '_' +creation_date+'.mp4'
    #     video_full_path_right=str(out_put_image_dir) + 'right/' + 'movie' +  '/Video_'+ right_video__view_side + '_' +video_name + '_' +creation_date+'.mp4'
    # elif args.output_option == 2:
    #     video_full_path_left= args.output_dir + '/Video_'+ left_video_view_side + '_' +video_name + '_' +creation_date+'.mp4'
    #     video_full_path_right= args.output_dir + '/Video_'+ right_video__view_side + '_' +video_name + '_' +creation_date+'.mp4'

    if args.raw_video == 'normal':
        # left_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(out_put_image_dir) + 'left/images/' + '/%d.' + \
        #     args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
        #     str(out_put_image_dir) + 'left/' + 'movie' + '/Video_' + \
        #     left_video_view_side + '_' + video_name + '_' + creation_date + '.mp4'

        left_shell_script = 'ffmpeg -framerate ' + str(30) + ' -i ' + str(out_put_image_dir) + 'left/images/' + '/%d.' + \
            args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
            str(out_put_image_dir) + 'left/' + 'movie' + '/Video_' + \
            left_video_view_side + '_' + video_name + '_' + creation_date + '.mp4'

        right_shell_script = 'ffmpeg -framerate ' + str(30) + ' -i ' + str(out_put_image_dir) + 'right/images/' + '/%d.' + \
            args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
            str(out_put_image_dir) + 'right/' + 'movie' + '/Video_' + \
            right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'

        # right_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(out_put_image_dir) + 'right/images/' + '/%d.' + \
        #     args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
        #     str(out_put_image_dir) + 'right/' + 'movie' + '/Video_' + \
        #     right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'

        print('raw_video =normal and left_shell_script----', left_shell_script)
        print('raw_video =normal and right_shell_script----', right_shell_script)
        os.system(left_shell_script)
        os.system(right_shell_script)

    if args.raw_video == 'raw':
        #left_shell_script='ffmpeg -framerate '+str(args.input_fps) + ' -i '+str(out_put_image_dir)+ 'left/images/' + '/%d.'+args.image_ext +' -c:v mpeg4 -b:v 3000k -qscale:v 1 -qscale:a 1 '+ str(out_put_image_dir)+ 'left/' +'movie'+'/movie.mp4'
        # left_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(out_put_image_dir) + 'left/images/' + \
        #     '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
        #     str(out_put_image_dir) + 'left/' + 'movie' + '/Video_' + \
        #     left_video_view_side + '_' + video_name + '_' + creation_date + '.mp4'

        # right_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(out_put_image_dir) + 'right/images/' + \
        #     '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
        #     str(out_put_image_dir) + 'right/' + 'movie' + '/Video_' + \
        #     right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'

        left_shell_script = 'ffmpeg -framerate ' + str(30) + ' -i ' + str(out_put_image_dir) + 'left/images/' + \
            '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
            str(out_put_image_dir) + 'left/' + 'movie' + '/Video_' + \
            left_video_view_side + '_' + video_name + '_' + creation_date + '.mp4'

        right_shell_script = 'ffmpeg -framerate ' + str(30) + ' -i ' + str(out_put_image_dir) + 'right/images/' + \
            '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
            str(out_put_image_dir) + 'right/' + 'movie' + '/Video_' + \
            right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'


        #right_shell_script='ffmpeg -framerate '+str(args.input_fps) + ' -i '+str(out_put_image_dir)+ 'right/images/' + '/%d.'+args.image_ext +' -c:v mpeg4 -b:v 3000k -qscale:v 1 -qscale:a 1 '+ str(out_put_image_dir)+ 'right/' +'movie'+'/movie.mp4'
        print('raw_video =raw and left_shell_script----', left_shell_script)
        print('raw_video =raw and right_shell_script----', right_shell_script)
        os.system(left_shell_script)
        os.system(right_shell_script)

    #seq_dict = {}
    #keyangle_dict = {}

    if alignment == "left_side":  # or alignment=="right_side":

        # Find optimal frame when left foot is leaving ground
        optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(all_leftview_hip_leftankle_distance_y, all_leftview_hip_leftankle_distance_x, 'positive', write_image_leftview_right_hip_knee_ankle,
                                                                                                all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_left_knee_leg_angle, all_leftview_left_ankle_coord, all_leftview_right_ankle_coord)

        seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                      all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_leftfoot_leaving_ground, img_leftfoot_leave, 'Phase_Dectection_leftsideview_leftextremity_toe_off_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when left foot is touching ground
        optimal_leftfoot_touching_ground, img_leftfoot_touch = optimal_frame_foot_touching_ground(all_leftview_hip_leftankle_distance_y, all_leftview_hip_leftankle_distance_x, 'negative', write_image_leftview_right_hip_knee_ankle,
                                                                                                  all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_left_knee_leg_angle, all_leftview_left_ankle_coord, all_leftview_right_ankle_coord)

        seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                      all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_leftfoot_touching_ground, img_leftfoot_touch, 'Phase_Dectection_leftsideview_leftextremity_initial_contact_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when right foot is leaving ground
        optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(all_leftview_hip_rightankle_distance_y, all_leftview_hip_rightankle_distance_x, 'positive', write_image_leftview_right_hip_knee_ankle,
                                                                                                  all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_right_knee_leg_angle, all_leftview_right_ankle_coord, all_leftview_left_ankle_coord)

        seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                      all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_rightfoot_leaving_ground, img_rightfoot_leave, 'Phase_Dectection_leftsideview_rightextremity_toe_off_phase', args.image_ext, alignment)


        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when right foot is touching ground
        optimal_rightfoot_touching_ground, img_rightfoot_touch = optimal_frame_foot_touching_ground(all_leftview_hip_rightankle_distance_y, all_leftview_hip_rightankle_distance_x, 'negative',
                                                                                                    write_image_leftview_right_hip_knee_ankle, all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_right_knee_leg_angle, all_leftview_right_ankle_coord, all_leftview_left_ankle_coord)

        seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                      all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_rightfoot_touching_ground, img_rightfoot_touch, 'Phase_Dectection_leftsideview_rightextremity_initial_contact_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when right foot is on ground
        optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground = optimal_frame_max_knee_flexion(all_leftview_left_hip_coord, all_leftview_right_knee_coord, all_leftview_right_ankle_coord, all_leftview_right_knee_leg_angle,
                                                                                                     all_leftview_left_knee_coord, all_leftview_left_ankle_coord, write_image_leftview_right_hip_knee_ankle, all_leftview_left_shoulder_coord, all_leftview_hip_rightankle_distance_y)

        seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                      all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground, 'Phase_Dectection_leftsideview_rightextremity_midstance_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when left foot is on ground
        optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground = optimal_frame_max_knee_flexion(all_leftview_left_hip_coord, all_leftview_left_knee_coord, all_leftview_left_ankle_coord, all_leftview_left_knee_leg_angle,
                                                                                                   all_leftview_right_knee_coord, all_leftview_right_ankle_coord, write_image_leftview_right_hip_knee_ankle, all_leftview_left_shoulder_coord, all_leftview_hip_leftankle_distance_y)

        knee = all_leftview_left_knee_coord[optimal_knee_flexion_leftfoot_ground][0]
        ankle = all_leftview_left_ankle_coord[optimal_knee_flexion_leftfoot_ground][0]

        if knee > ankle:
            sign= 'Yes'
        else:
            sign = 'No'

        seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                      all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground, 'Phase_Dectection_leftsideview_leftextremity_midstance_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # save Optimal frame
        save_optimal_frame(leftview_left_image_path, img_leftfoot_leave,
                           'Phase_Dectection_leftsideview_leftextremity_toe_off_phase_')
        save_optimal_frame(leftview_left_image_path, img_leftfoot_touch,
                           'Phase_Dectection_leftsideview_leftextremity_initial_contact_phase_')
        save_optimal_frame(leftview_right_image_path, img_rightfoot_leave,
                           'Phase_Dectection_leftsideview_rightextremity_toe_off_phase_')
        save_optimal_frame(leftview_right_image_path, img_rightfoot_touch,
                           'Phase_Dectection_leftsideview_rightextremity_initial_contact_phase_')
        save_optimal_frame(leftview_right_image_path, img_rightfoot_ground,
                           'Phase_Dectection_leftsideview_rightextremity_midstance_phase_')
        save_optimal_frame(leftview_left_image_path, img_leftfoot_ground,
                           'Phase_Dectection_leftsideview_leftextremity_midstance_phase_')

    if alignment == "right_side":
        # Find optimal frame when left foot is leaving ground
        optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(all_rightview_hip_leftankle_distance_y, all_rightview_hip_leftankle_distance_x, 'negative', write_image_rightview_right_hip_knee_ankle,
                                                                                                all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_left_knee_leg_angle, all_rightview_left_ankle_coord, all_rightview_right_ankle_coord)

        seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                      all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_leftfoot_leaving_ground, img_leftfoot_leave, 'Phase_Dectection_rightsideview_leftextremity_toe_off_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when left foot is touching ground
        optimal_leftfoot_touching_ground, img_leftfoot_touch = optimal_frame_foot_touching_ground(all_rightview_hip_leftankle_distance_y, all_rightview_hip_leftankle_distance_x, 'positive', write_image_rightview_right_hip_knee_ankle,
                                                                                                  all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_left_knee_leg_angle, all_rightview_left_ankle_coord, all_rightview_right_ankle_coord)

        seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                      all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_leftfoot_touching_ground, img_leftfoot_touch, 'Phase_Dectection_rightsideview_leftextremity_initial_contact_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when right foot is leaving ground
        optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(all_rightview_hip_rightankle_distance_y, all_rightview_hip_rightankle_distance_x, 'negative', write_image_rightview_right_hip_knee_ankle,
                                                                                                  all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_right_knee_leg_angle, all_rightview_right_ankle_coord, all_rightview_left_ankle_coord)

        path = video_frame_path+'/'+img_rightfoot_leave
        points_pelvis_tilt = all_rightview_right_pelvis_tilt[optimal_rightfoot_leaving_ground]
        angle_right_hip  = all_rightview_right_knee_hip_vertical[optimal_rightfoot_leaving_ground]
        try:
            # print("try")
            # image_processing_right(path,points_pelvis_tilt,rightview_left_image_path,img_rightfoot_leave)
            angle_pelvis_tilt = image_processing_right(path,points_pelvis_tilt,rightview_right_image_path,img_rightfoot_leave)
            angle_final = angle_right_hip - angle_pelvis_tilt
            print(angle_pelvis_tilt,angle_final)
        except:
            # print("catch")
            pass
        # # print(path,angle_pelvis_tilt,angle_right_hip)
        # angle_final = angle_right_hip - angle_pelvis_tilt


        seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                      all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_rightfoot_leaving_ground, img_rightfoot_leave, 'Phase_Dectection_rightsideview_rightextremity_toe_off_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when right foot is touching ground
        optimal_rightfoot_touching_ground, img_rightfoot_touch = optimal_frame_foot_touching_ground(all_rightview_hip_rightankle_distance_y, all_rightview_hip_rightankle_distance_x, 'positive', write_image_rightview_right_hip_knee_ankle,
                                                                                                    all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_right_knee_leg_angle, all_rightview_right_ankle_coord, all_rightview_left_ankle_coord)

        seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                      all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_rightfoot_touching_ground, img_rightfoot_touch, 'Phase_Dectection_rightsideview_rightextremity_initial_contact_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when right foot is on ground
        optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground = optimal_frame_max_knee_flexion(all_rightview_right_hip_coord, all_rightview_right_knee_coord, all_rightview_right_ankle_coord, all_rightview_right_knee_leg_angle,
                                                                                                     all_rightview_left_knee_coord, all_rightview_left_ankle_coord, write_image_rightview_right_hip_knee_ankle, all_rightview_right_shoulder_coord, all_rightview_hip_rightankle_distance_y)

        ankle = all_rightview_right_knee_coord[optimal_rightfoot_touching_ground][0]
        knee = all_rightview_right_ankle_coord[optimal_rightfoot_touching_ground][0]

        sign = ''
        if knee > ankle:
            sign= 'Yes'
        else:
            sign = 'No'
        seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                      all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground, 'Phase_Dectection_rightsideview_rightextremity_midstance_phase', args.image_ext, alignment,sign)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Find optimal frame when left foot is on ground
        optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground = optimal_frame_max_knee_flexion(all_rightview_right_hip_coord, all_rightview_left_knee_coord, all_rightview_left_ankle_coord, all_rightview_left_knee_leg_angle,
                                                                                                   all_rightview_right_knee_coord, all_rightview_right_ankle_coord, write_image_rightview_left_hip_knee_ankle, all_rightview_right_shoulder_coord, all_rightview_hip_leftankle_distance_y)

        seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                      all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground, 'Phase_Dectection_rightsideview_leftextremity_midstance_phase', args.image_ext, alignment)

        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # save Optimal frame
        save_optimal_frame(rightview_left_image_path, img_leftfoot_leave,
                           'Phase_Dectection_rightsideview_leftextremity_toe_off_phase_')
        save_optimal_frame(rightview_left_image_path, img_leftfoot_touch,
                           'Phase_Dectection_rightsideview_leftextremity_initial_contact_phase_')
        save_optimal_frame(rightview_right_image_path, img_rightfoot_leave,
                           'Phase_Dectection_rightsideview_rightextremity_toe_off_phase_')
        save_optimal_frame(rightview_right_image_path, img_rightfoot_touch,
                           'Phase_Dectection_rightsideview_rightextremity_initial_contact_phase_')
        save_optimal_frame(rightview_right_image_path, img_rightfoot_ground,
                           'Phase_Dectection_rightsideview_rightextremity_midstance_phase_')
        save_optimal_frame(rightview_left_image_path, img_leftfoot_ground,
                           'Phase_Dectection_rightsideview_leftextremity_midstance_phase_')

    if alignment == "back_side":
        # print("back_side")
        # optimal frame when right foot on ground
        # print(all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor)

        print("back right")
        optimal_rightfoot_on_ground, img_rightfoot_ground = optimal_frame_foot_on_ground_back_front(
            all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor)


        # Calculate cross_over_sign
        x_point, y_point = feet_inner_edge(optimal_rightfoot_on_ground,all_back_hip_difference,args.distance_cam,'right')
        mean_hip = all_back_hip_coor[optimal_rightfoot_on_ground][0]
        #x,y
        new_ankle_points = [int(all_back_right_ankle_coor[optimal_rightfoot_on_ground][0] + x_point),int(all_back_right_ankle_coor[optimal_rightfoot_on_ground][1] + y_point)]
        ankle_point_right = all_back_right_ankle_coor[optimal_rightfoot_on_ground][0]
        ankle_point_right = ankle_point_right + x_point

        img = cv2.imread(back_side_right_image_path+''+img_rightfoot_ground)

        # path_img = './outputs/lauren_backview_2_December31_1903_05.00PM/backside/right/images/'+''+img_rightfoot_ground
        path_img = back_side_right_image_path+''+img_rightfoot_ground
        # print('img_rightfoot_ground = ',img_rightfoot_ground, img_rightfoot_ground.split('.')[0])

        img_name = img_rightfoot_ground.split('.')[0]
        # print('path_img = ', img_name)

        # print('python apply_net.py dump configs/densepose_rcnn_R_101_FPN_s1x.yaml model_final_c6ab63.pkl {} --output results/img_{}.pkl -v'.format(path_img, img_name))
        subprocess.call('/home/ubuntu/anaconda3/envs/densepose/bin/python'+' apply_net.py'+' dump'+' configs/densepose_rcnn_R_101_FPN_s1x.yaml'+' model_final_c6ab63.pkl '+path_img+' --output'+' results/'+img_name+'.pkl'+' -v', shell=True)
        # os.system('python apply_net.py dump configs/densepose_rcnn_R_101_FPN_s1x.yaml model_final_c6ab63.pkl {} --output results/img_{}.pkl -v'.format(path_img, img_name))
        # os.system('python Detectron2_DensePose/apply_net.py dump Detectron2_DensePose/densepose_rcnn_R_101_FPN_s1x.yaml Detectron2_DensePose/model_final_c6ab63.pkl '+path_img+' --output Detectron2_DensePose/results'++'.pkl -v')

        map_l, dist_img_left, min_indcs_l, map_r, dist_img_right, min_indcs_r, x, y, x2, y2 = iuv(img_name)

        # min_indcs_l[0][0],[min_indcs_l[1][0]
        # print (min_indcs_l[0][0],min_indcs_l[1][0])
        # print (min_indcs_r[0][0],min_indcs_r[1][0])

        img = cv2.circle(img, (x+min_indcs_l[1][0],y+min_indcs_l[0][0]), radius=5, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, (x+min_indcs_r[1][0],y+min_indcs_r[0][0]), radius=5, color=(0, 0, 255), thickness=-1)

        img = cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 3)
        #
        # img = cv2.circle(img, (int(new_ankle_points[0]),int(new_ankle_points[1])), radius=0, color=(0, 255, 255), thickness=-1)
        # print (back_side_right_image_path+''+img_rightfoot_ground)
        # cv2.imwrite('./'+img_rightfoot_ground, img)
        cv2.imwrite(back_side_right_image_path+''+img_rightfoot_ground,img)
        # print(optimal_rightfoot_on_ground, img_rightfoot_ground,all_back_right_ankle_coor[optimal_rightfoot_on_ground],x_point, y_point,ankle_point_right)

        cross_over_sign_right = ''
        if ankle_point_right > mean_hip:
            cross_over_sign_right = 'No'
        else:
            cross_over_sign_right = 'Yes'

        seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                      all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground, img_rightfoot_ground, 'Phase_Dectection_backview_right_midstance_phase', args.image_ext, args.pelvis, alignment,cross_over_sign_right)
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # # Find optimal frame when right foot is on ground
        # optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground_angle = optimal_frame_max_knee_flexion(all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_right_knee_leg_angle,
        #                                                                                                    all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor, all_backview_hip_rightankle_distance_y)

        # seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
        #                                               all_back_left_hip_right_hip, all_back_trunk_lean, optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground_angle, 'Phase_Dectection_backview_right_midstance_phase_angle', args.image_ext, args.pelvis, alignment)

        # seq_list.append(seq_point_dict)
        # optimal_frame_seq_list.append(seq_point_dict)

        # #####################################################

        # Optimal frame when right foot on ground with max knee varus angle:
        # print("Knee varus right")
        optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor, all_back_right_knee_leg_angle,all_back_right_knee_leg_angle_lable , 2,'non_trunk')

        seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                      all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus, 'Phase_Dectection_backview_right_max_knee_varus', args.image_ext, args.pelvis, alignment,'right_knee')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # print("max_femour right")
        optimal_rightfoot_on_ground_max_femour, img_rightfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_knee_hip_vertical, all_back_shoulder_coor, all_back_right_knee_hip_vertical,all_back_right_knee_hip_vertical_label, 2,'non_trunk')

        # print(optimal_rightfoot_on_ground_max_femour, img_rightfoot_ground_max_femour)
        seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                      all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground_max_femour, img_rightfoot_ground_max_femour, 'Phase_Dectection_backview_right_max_femour', args.image_ext, args.pelvis, alignment,'right_femour')
        # print(seq_point_dict)
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # max trunk lane
        print("trunk right")
        optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
            all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_trunk_lean, all_back_shoulder_coor, all_back_trunk_lean, 'trunk')

        seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                      all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean, 'Phase_Dectection_backview_right_lateral_trunk_lean', args.image_ext, args.pelvis, alignment,'trunk')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)


        if args.pelvis == 'on':
            optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor, all_back_hip_coor_left_right,video_frame_path)
        # print(angle)
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                        all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop, 'Phase_Dectection_backview_right_pelvis_angle', args.image_ext, args.pelvis, alignment,'',angle)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # print("Argument is {}".format(args.pelvis))
            # optimal_frame_foot_on_ground_back_front_all_points(all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_left_hip_right_hip, all_back_shoulder_coor, all_back_hip_coor_left_right,video_frame_path,all_back_right_feet)
            # pelvis angle
            # optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_left_hip_right_hip, all_back_shoulder_coor, all_back_hip_coor_left_right,video_frame_path)
            # print(angle)
            # seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
            #                                             all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop, 'Phase_Dectection_backview_right_pelvis_angle', args.image_ext, args.pelvis, alignment,'',angle)
            # seq_list.append(seq_point_dict)
            # optimal_frame_seq_list.append(seq_point_dict)
        # #####################################################

        # optimal frame when left foot on ground
        print("back left")
        optimal_leftfoot_on_ground, img_leftfoot_ground = optimal_frame_foot_on_ground_back_front(
            all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor)

        # Find cross_over
        x_point, y_point = feet_inner_edge(optimal_leftfoot_on_ground,all_back_hip_difference,args.distance_cam,'left')
        mean_hip = all_back_hip_coor[optimal_leftfoot_on_ground][0]
        new_points_left = [int(all_back_left_ankle_coor[optimal_leftfoot_on_ground][0]+x_point),int(all_back_left_ankle_coor[optimal_leftfoot_on_ground][1]+y_point)]
        ankle_point_left = all_back_left_ankle_coor[optimal_leftfoot_on_ground][0]
        ankle_point_left = ankle_point_left + x_point

        img = cv2.imread(back_side_left_image_path+''+img_leftfoot_ground)
        img = cv2.circle(img, (int(new_points_left[0]),int(new_points_left[1])), radius=0, color=(0, 255, 255), thickness=-1)
        cv2.imwrite(back_side_left_image_path+''+img_leftfoot_ground,img)
        # print(optimal_leftfoot_on_ground, img_leftfoot_ground,all_back_left_ankle_coor[optimal_leftfoot_on_ground],x_point, y_point)
        cross_over_sign_left = ''
        if ankle_point_left > mean_hip:
            cross_over_sign_left = 'No'
        else:
            cross_over_sign_left = 'Yes'

        seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                      all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground, img_leftfoot_ground, 'Phase_Dectection_backview_left_midstance_phase', args.image_ext, args.pelvis, alignment,cross_over_sign_left)
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)


        # # Find optimal frame when left foot is on ground
        # optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground_angle = optimal_frame_max_knee_flexion(all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_left_knee_leg_angle,
        #                                                                                                  all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor, all_backview_hip_leftankle_distance_y)

        # seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
        #                                               all_back_left_hip_right_hip, all_back_trunk_lean, optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground_angle, 'Phase_Dectection_backview_left_midstance_phase_angle', args.image_ext, args.pelvis, alignment)

        # seq_list.append(seq_point_dict)
        # optimal_frame_seq_list.append(seq_point_dict)

        # #####################################################

        # Optimal frame when left foot on ground with max knee varus angle:

        # print("knee left")
        optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor, all_back_left_knee_leg_angle,all_back_left_knee_leg_angle_label, 1,'non_trunk')

        seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                      all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus, 'Phase_Dectection_backview_left_max_knee_varus', args.image_ext, args.pelvis, alignment,'left_knee')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # print("femour left")
        optimal_leftfoot_on_ground_max_femour, img_leftfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_knee_hip_vertical, all_back_shoulder_coor, all_back_left_knee_hip_vertical,all_back_left_knee_hip_vertical_label, 1,'non_trunk')

        seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                      all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground_max_femour, img_leftfoot_ground_max_femour, 'Phase_Dectection_backview_left_max_femour', args.image_ext, args.pelvis, alignment,'left_femour')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # max trunk lane
        print("trunk left")
        optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
            all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_trunk_lean, all_back_shoulder_coor, all_back_trunk_lean,all_back_trunk_lean_label, 1,'trunk')

        seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                      all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane, 'Phase_Dectection_backview_left_lateral_trunk_lean', args.image_ext, args.pelvis, alignment,'trunk')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        if args.pelvis == 'on':
            optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(
                all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor, all_back_hip_coor_left_right,video_frame_path)

            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                        all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop, 'Phase_Dectection_backview_left_pelvis_angle', args.image_ext, args.pelvis, alignment,'',angle)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)

        #     optimal_frame_foot_on_ground_back_front_all_points(all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_hip_right_hip, all_back_shoulder_coor, all_back_hip_coor_left_right,video_frame_path)
        # # pelvis angle
        #     optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(
        #         all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_hip_right_hip, all_back_shoulder_coor, all_back_hip_coor_left_right,back_side_left_image_path)

        #     seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
        #                                                 all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop, 'Phase_Dectection_backview_left_pelvis_angle', args.image_ext, args.pelvis, alignment,'',angle)
        #     seq_list.append(seq_point_dict)
        #     optimal_frame_seq_list.append(seq_point_dict)


        # #####################################################

        # Find max positive chest distance y axis
        distance_chest_keypoints = []
        yaxis_list = []

        if len(all_back_chest_yaxis_values) != 0:
            # print(all_back_chest_yaxis_values)
            for values in all_back_chest_yaxis_values:
                yaxis_list.append(values[1])
            max_value = max(yaxis_list)
            min_value = min(yaxis_list)
            max_index_frame = all_back_chest_yaxis_values[yaxis_list.index(max_value)]
            min_index_frame = all_back_chest_yaxis_values[yaxis_list.index(min_value)]

            distance = max_value - min_value

            distance_chest_keypoints.append(distance)
            distance_chest_keypoints.append(min_index_frame[0])
            distance_chest_keypoints.append(max_index_frame[0])

        if distance >0:
            seq_point_dict = back_chest('Phase_Dectection_backview_',distance_chest_keypoints)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)

        # save Optimal frame
        save_optimal_frame(back_side_right_image_path, img_rightfoot_ground,
                           'Phase_Dectection_backview_right_midstance_phase_')
        save_optimal_frame(back_side_right_image_path, img_rightfoot_ground,
                           'Phase_Dectection_backview_right_cross_over_')
        # save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_angle,
        #                    'Phase_Dectection_backview_right_midstance_phase_angle_')
        save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_max_varus,
                           'Phase_Dectection_backview_right_max_knee_varus_')
        save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_max_femour,
                           'Phase_Dectection_backview_right_max_femour_')
        save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_lateral_trunk_lean,
                           'Phase_Dectection_backview_right_lateral_trunk_lean_')

        save_optimal_frame(back_side_right_image_path, distance_chest_keypoints[1],
                           'Phase_Dectection_backview_chest_distance_min_')
        save_optimal_frame(back_side_right_image_path, distance_chest_keypoints[2],
                           'Phase_Dectection_backview_chest_distance_max_')


        if args.pelvis == 'on':
            save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_pelvis_drop,
                            'Phase_Dectection_backview_right_pelvis_angle_')

        save_optimal_frame(back_side_left_image_path, img_leftfoot_ground,
                           'Phase_Dectection_backview_left_midstance_phase_')
        save_optimal_frame(back_side_left_image_path, img_leftfoot_ground,
                           'Phase_Dectection_backview_left_cross_over_')
        # save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_angle,
        #                    'Phase_Dectection_backview_left_midstance_phase_angle_')
        save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_max_varus,
                           'Phase_Dectection_backview_left_max_knee_varus_')
        save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_max_femour,
                           'Phase_Dectection_backview_left_max_femour_')
        save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_max_trunk_lane,
                           'Phase_Dectection_backview_left_lateral_trunk_lean_')
        if args.pelvis == 'on':
            save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_pelvis_drop,
                            'Phase_Dectection_backview_left_pelvis_angle_')


    if alignment == "front_side":
        # print("right foot")
        # optimal frame when right foot on ground
        optimal_rightfoot_on_ground, img_rightfoot_ground = optimal_frame_foot_on_ground_back_front(
            all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_right_hip_knee_ankle, all_front_shoulder_coor)

        seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                      all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground, img_rightfoot_ground, 'Phase_Dectection_frontview_right_midstance_phase', args.image_ext, args.pelvis, alignment)
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # # Find optimal frame when right foot is on ground
        # optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground_angle = optimal_frame_max_knee_flexion(all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_right_knee_leg_angle,
        #                                                                                                    all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_right_hip_knee_ankle, all_back_shoulder_coor, all_frontview_hip_rightankle_distance_y)
        # seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
        #                                               all_front_left_hip_right_hip, all_front_trunk_lean, optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground_angle, 'Phase_Dectection_frontview_right_midstance_phase_angle', args.image_ext, args.pelvis, alignment)

        # seq_list.append(seq_point_dict)
        # optimal_frame_seq_list.append(seq_point_dict)

        # #####################################################

        # Optimal frame when right foot on ground with max knee varus angle
        optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_right_hip_knee_ankle, all_front_shoulder_coor, all_front_right_knee_leg_angle,all_front_right_knee_leg_angle_label, 2,'non_trunk')

        seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                      all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus, 'Phase_Dectection_frontview_right_max_knee_varus', args.image_ext, args.pelvis, alignment,'right_knee')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # Optimal frame when right foot on ground with max femour angle
        optimal_rightfoot_on_ground_max_femour, img_rightfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_right_knee_hip_vertical, all_front_shoulder_coor, all_front_right_knee_hip_vertical,all_front_right_knee_hip_vertical_label, 2,'non_trunk')

        seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                      all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground_max_femour, img_rightfoot_ground_max_femour, 'Phase_Dectection_frontview_right_max_femour', args.image_ext, args.pelvis, alignment,'right_femour')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # max trunk lane
        optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
            all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_trunk_lean, all_front_shoulder_coor, all_front_trunk_lean, "trunk")

        seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                      all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean, 'Phase_Dectection_frontview_lateral_trunk_lean', args.image_ext, args.pelvis, alignment,'trunk')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        if args.pelvis == 'on':
            # pelvis angle
            optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(
                all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_left_hip_right_hip, all_front_shoulder_coor, all_front_hip_coor_left_right, video_frame_path)

            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                        all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop, 'Phase_Dectection_frontview_pelvis_angle', args.image_ext, args.pelvis, alignment,angle)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)

        # #####################################################

        # optimal frame when left foot on ground
        optimal_leftfoot_on_ground, img_leftfoot_ground = optimal_frame_foot_on_ground_back_front(
            all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_left_hip_knee_ankle, all_front_shoulder_coor)

        seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                      all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground, img_leftfoot_ground, 'Phase_Dectection_frontview_left_midstance_phase', args.image_ext, args.pelvis, alignment)
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # # Find optimal frame when left foot is on ground
        # optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground_angle = optimal_frame_max_knee_flexion(all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_left_knee_leg_angle,
        #                                                                                                  all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_left_hip_knee_ankle, all_back_shoulder_coor, all_frontview_hip_leftankle_distance_y=[])

        # seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
        #                                               all_front_left_hip_right_hip, all_front_trunk_lean, optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground_angle, 'Phase_Dectection_frontview_left_midstance_phase_angle', args.image_ext, args.pelvis, alignment)

        # seq_list.append(seq_point_dict)
        # optimal_frame_seq_list.append(seq_point_dict)
        # #####################################################

        # Optimal frame when left foot on ground with max knee varus angle:

        optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_left_hip_knee_ankle, all_front_shoulder_coor, all_front_left_knee_leg_angle,all_front_left_knee_leg_angle_label, 1,'non_trunk')

        seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                      all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus, 'Phase_Dectection_frontview_left_max_knee_varus', args.image_ext, args.pelvis, alignment,'left_knee')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        optimal_leftfoot_on_ground_max__femour, img_leftfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_left_knee_hip_vertical, all_front_shoulder_coor, all_front_left_knee_hip_vertical,all_front_left_knee_hip_vertical_label, 1,'non_trunk')

        seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                      all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground_max__femour, img_leftfoot_ground_max_femour, 'Phase_Dectection_frontview_left_max_femour', args.image_ext, args.pelvis, alignment,'left_femour')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # max trunk lane
        optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
            all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_trunk_lean, all_front_shoulder_coor, all_front_trunk_lean,all_front_trunk_lean_label, 1,'trunk')

        seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                      all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane, 'Phase_Dectection_frontview_left_lateral_trunk_lean', args.image_ext, args.pelvis, alignment,'trunk')
        seq_list.append(seq_point_dict)
        optimal_frame_seq_list.append(seq_point_dict)

        # optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
        #     all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor,
        #     all_front_right_ankle_coor, write_image_front_trunk_lean, all_front_shoulder_coor, all_front_trunk_lean,
        #     all_front_trunk_lean_label, 1, 'trunk')
        # seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle,
        #                                               all_front_right_knee_hip_vertical,
        #                                               all_front_left_knee_hip_vertical,
        #                                               all_front_left_hip_right_hip, all_front_trunk_lean,
        #                                               optimal_leftfoot_on_ground_max_trunk_lane,
        #                                               img_leftfoot_ground_max_trunk_lane,
        #                                               'Phase_Dectection_frontview_left_lateral_trunk_lean',
        #                                               args.image_ext, args.pelvis, alignment, 'trunk')
        # seq_list.append(seq_point_dict)
        # optimal_frame_seq_list.append(seq_point_dict)

        if args.pelvis == 'on':
        # pelvis angle
            optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(
                all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_left_hip_right_hip, all_front_shoulder_coor, all_front_hip_coor_left_right,video_frame_path)

            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                        all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground_max__femour, img_leftfoot_ground_max_femour, 'Phase_Dectection_frontview_left_pelvis_angle', args.image_ext, args.pelvis, alignment,angle)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)

        # #####################################################


        # save Optimal frame
        save_optimal_frame(front_right_image_path, img_rightfoot_ground,
                           'Phase_Dectection_frontview_right_midstance_phase_')
        # save_optimal_frame(front_right_image_path, img_rightfoot_ground_angle,
        #                    'Phase_Dectection_frontview_right_midstance_phase_angle_')
        save_optimal_frame(front_right_image_path, img_rightfoot_ground_max_varus,
                           'Phase_Dectection_frontview_right_max_knee_varus_')
        save_optimal_frame(front_right_image_path, img_rightfoot_ground_max_femour,
                           'Phase_Dectection_frontview_right_max_femour')
        save_optimal_frame(front_right_image_path, img_rightfoot_ground_lateral_trunk_lean,
                           'Phase_Dectection_frontview_right_lateral_trunk_lean_')
        if args.pelvis == 'on':
            save_optimal_frame(front_right_image_path, img_rightfoot_ground_pelvis_drop,
                            'Phase_Dectection_frontview_pelvis_angle_')

        save_optimal_frame(front_left_image_path, img_leftfoot_ground,
                           'Phase_Dectection_frontview_left_midstance_phase_')
        # save_optimal_frame(front_left_image_path, img_leftfoot_ground_angle,
        #                    'Phase_Dectection_frontview_left_midstance_phase_angle_')
        save_optimal_frame(front_left_image_path, img_leftfoot_ground_max_varus,
                           'Phase_Dectection_frontview_left_max_knee_varus')
        save_optimal_frame(front_left_image_path, img_leftfoot_ground_max_femour,
                           'Phase_Dectection_frontview_left_max_femour')
        save_optimal_frame(front_left_image_path, img_leftfoot_ground_max_trunk_lane,
                           'Phase_Dectection_frontview_left_lateral_trunk_lean_')
        if args.pelvis == 'on':
            save_optimal_frame(front_left_image_path, img_leftfoot_ground_pelvis_drop,
                            'Phase_Dectection_frontview_left_pelvis_angle_')

    json_list['sequence'] = seq_list
    optimal_frame_dict['sequence'] = optimal_frame_seq_list
    # time_list = np.array(time_list)
    # totaltimeused = inprocess_endtime - inprocess_start
    # total_time.append(totaltimeused)
    # total_time = np.array(total_time)

    # totaltimeused = inprocess_endtime - inprocess_start
    # total_time.append(totaltimeused)
    # totaltimeaftermodel = postprocess_endtime - postprocess_start
    # post_time.append(totaltimeaftermodel)
    # post_time1 = np.array(post_time1)
    # post_time2 = np.array(post_time2)
    # post_time3 = np.array(post_time3)
    # for i in range(1, 5):
    #    times[i] = np.array(times[i])
    #    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #    print("#                   model  Time (Avg, Std): ({:.3f}s, {:.3f}s)                                 #".format(times[i].mean(), times[i].std()))
    #    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # print("###################################################################################################")
    # print("#                     Inference Time (Avg, Std): ({:.3f}s, {:.3f}s)                                 #".format(time_list.mean(), time_list.std()))
    # print("###################################################################################################")
    # # print('***************************************************** before processing time: {:.3f}s'.format(in_process_end - inprocess_start))
    # # print('***************************************************** before processing time: {:.3f}s'.format(end_time-start_time))
    # print("###################################################################################################")
    # print("#                     preprocesss time (Avg, Std): ({:.3f}s, {:.3f}s)                                 #".format(total_time.mean(), total_time.std()))
    # print("###################################################################################################")

    # print("#                     total 1st post-processing time (Avg, Std): ({:.3f}s, {:.3f}s)              #".format(post_time1.mean(), post_time1.std()))

    # print("#                     total 2nd post-processing time (Avg, Std): ({:.3f}s, {:.3f}s)               #".format(post_time2.mean(), post_time2.std()))

    # print("#                     total 3rd post-processing time (Avg, Std): ({:.3f}s, {:.3f}s)               #".format(post_time3.mean(), post_time3.std()))
    # print("###################################################################################################")
    # json.dump(json_list) #dump_json
    with open(json_folder, 'w') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)

    with open(optimal_json_name, 'w') as f:
        json.dump(optimal_frame_dict, f, ensure_ascii=False, indent=4)

    if(int(args.output_option) == 3 or int(args.output_option) == 2):
        cpoy_left_video_shell_script = 'cp ' + out_put_image_dir + '/left/movie/*.mp4 ' + args.output_dir + '/'
        cpoy_right_video_shell_script = 'cp ' + out_put_image_dir + \
            '/right/movie/*.mp4 ' + args.output_dir + '/'
        cpoy_json_shell_script = 'cp ' + out_put_image_dir + '/*.json ' + args.output_dir + '/'
        copy_left_optimalframe_shell_script = 'cp ' + out_put_image_dir + \
            '/left/images/Phase_Dectection* ' + args.output_dir + '/'
        copy_right_optimalframe_shell_script = 'cp ' + out_put_image_dir + \
            '/right/images/Phase_Dectection* ' + args.output_dir + '/'
        os.system(cpoy_left_video_shell_script)
        os.system(cpoy_right_video_shell_script)
        os.system(cpoy_json_shell_script)
        os.system(copy_left_optimalframe_shell_script)
        os.system(copy_right_optimalframe_shell_script)

    if(int(args.output_option) == 2):
        remove_olfiles_script = 'rm -rf ' + out_put_image_dir + ' ' + args.output_dir + '/images'
        remove_images_folder_script = 'rm -rf ' + args.output_dir + '/images'
        os.system(remove_olfiles_script)
        os.system(remove_images_folder_script)


def creation_date_fun(filename):
    '''
        Input: video file path
        return: creation date of that video
    '''
    parser = createParser(filename)
    metadata = extractMetadata(parser)
    return metadata.get('creation_date') + timedelta(hours=-7)



if __name__ == '__main__':
    # workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    # setup_logging(__name__)
    # print (os.getcwd())
    # print (os.listdir(os.getcwd()))
    # print (os.system("ls ."))
    # subprocess.call('/home/ubuntu/anaconda3/envs/densepose/bin/python' + ' apply_net.py' + ' dump' + ' configs/densepose_rcnn_R_101_FPN_s1x.yaml' + ' model_final_c6ab63.pkl ' + "46.PNG" + ' --output' + ' results/img_' + '46' + '.pkl' + ' -v',shell=True)
    # exit(0)
    args = parse_args()
    # merge_cfg_from_file(args.cfg)
    # cfg.NUM_GPUS = 1
    # args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    # assert_and_infer_cfg(cache_urls=False)
    # model = infer_engine.initialize_model_from_cfg(args.weights)
    # dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
    model = DefaultPredictor(cfg)

    # check if input type is video or not and if that path contains multiple video, it will read one by one video
    if args.input_type == "video" and os.path.isdir(args.video_path):
        input_dir = args.video_path
        output_dir = args.output_dir
        for filename in os.listdir(input_dir):
            if filename.endswith("." + args.video_type):
                print(os.path.join(input_dir, filename))
                #modified_video_path = os.path.join(input_dir, filename)
                args.video_path = os.path.join(input_dir, filename)
                args.output_dir = output_dir
                try:
                    main(args)
                    # points_keypoints= np.array(points_keypoints)
                    # np.save('/optdetectron2/tools/101/points.npy', points_keypoints)
                except Exception as e:
                    print("Error has been found in video 1 ----", args.video_path)
                    print("eeee", e)
                    logging.error(traceback.format_exc())
            else:
                print("No video is found with extension  ", args.video_type)

    else:
        try:
            main(args)
        except Exception as e:
            print("Error has been found in video 2 ----", args.video_path)
            logging.error(traceback.format_exc())
