"""
Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""
import subprocess
import matplotlib.pyplot as plt
from datetime import timedelta
import subprocess
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
from datetime import datetime
from FootModel import FootModel
from scipy.signal import savgol_filter
import pickle
from sklearn import preprocessing
from face_blurring import FaceBlurring
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
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
        '--auto_fps',
        dest='auto_fps',
        help='auto select frame rate',
        default='off',
        type=str
    )
    parser.add_argument(
        '--auto_resize',
        dest='auto_resize',
        help='compress lossless images',
        default='off',
        type=str
    )
    parser.add_argument(
        '--face_blur',
        dest='face_blur',
        help='model to blur person face',
        default='off',
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
        '--foot_model_path',
        dest='foot_model_path',
        help='predict foot on ground',
        default='/opt/detectron2/tools/slowmo2/weights_crop_ResNet_rm.best.hdf5',
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
        '--new-frames',
        dest='max_frame1',
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
        '--side_pelvis',
        dest='side_pelvis',
        help='on : Draw Side Pelvis angles and lines \n off: Don\'t Draw Side Pelvis',
        default='off',
        type=str
    )
    parser.add_argument(
        '--front_back_pelvis',
        dest='front_back_pelvis',
        help='on : Draw Front Back Pelvis angles and lines \n off: Don\'t Draw Side Pelvis',
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
def calc_angle(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return np.degrees(math.pi / 2)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
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
    seq_info = {}
    keypoint_info = {}
    try:
        keypoint_info['right_knee_varus_valgus'] = round(
            right_knee_leg_angle[optimal_point], 1)
        keypoint_info['left_knee_varus_valgus'] = round(
            left_knee_leg_angle[optimal_point], 1)
    except:
        keypoint_info['right_knee_varus_valgus'] = 0
        keypoint_info['left_knee_varus_valgus'] = 0
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
    try:
        keypoint_info['right_femur_angle'] = round(
            right_knee_hip_vertical[optimal_point], 1)
    except:
        keypoint_info['right_femur_angle'] = 0
    if cross_over_sign == 'right_femur':
        if right_knee_hip_vertical[optimal_point] <=-5:
            keypoint_info['right_femur_angle'] = 2
        elif right_knee_hip_vertical[optimal_point] <-10:
            keypoint_info['right_femur_angle'] = 1
        else:
            keypoint_info['right_femur_angle'] = 0
    try:
        keypoint_info['left_femur_angle'] = round(
            left_knee_hip_vertical[optimal_point], 1)
    except:
        keypoint_info['left_femur_angle'] = 0
    if cross_over_sign == 'left_femur':
        if left_knee_hip_vertical[optimal_point] <=-5:
            keypoint_info['left_femur_angle'] = 2
        elif left_knee_hip_vertical[optimal_point] <-10:
            keypoint_info['left_femur_angle'] = 1
        else:
            keypoint_info['left_femur_angle'] = 0
    try:
        keypoint_info['trunk_lateral_lean'] = round(trunk_lean[optimal_point], 1)
    except:
        keypoint_info['trunk_lateral_lean'] = 0
    if cross_over_sign == 'trunk':
        if trunk_lean[optimal_point] <= -5:
            keypoint_info['trunk_lateral_lean'] = 2
        elif trunk_lean[optimal_point] <= -10:
            keypoint_info['trunk_lateral_lean'] = 1
        else:
            keypoint_info['trunk_lateral_lean'] = 0
    if pelvis_status == 'on':
        try:
            keypoint_info['pelvis_angle'] = round(
                left_hip_right_hip[optimal_point], 1)
        except:
            keypoint_info['pelvis_angle'] = 0
    seq_info['frame_number'] = optimal_image_name + '_' + str(img_name)
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

def back_shoulder(name,min,max,difference):
    seq_info = {}
    difference_ratio=abs(difference)
    seq_info['Type'] = 'Max y-axis difference in shoulder keypoints'
    seq_info['chest_min_yaxis_frame'] = name+'min_value_'+str(min+1)+"."+str(args.image_ext)
    seq_info['chest_max_yaxis_frame'] = name+'max_value_'+str(max+1)+"."+str(args.image_ext)
    seq_info['ratio-distance'] = difference_ratio
    return seq_info

def crop_img(img,path,extn,im_name,bbox):
    x1 = bbox[0]
    x2 = bbox[2]
    y1 = bbox[1]
    y2 = bbox[3]
    img_width=x2-x1
    width = img.shape[1]
    xpantion_pixels=int(img_width*3.2)
    dec_x1 = x1-int(xpantion_pixels/2)
    inc_x2 = x2+int(xpantion_pixels/2)
    if dec_x1<0:
        dec_x1=0
    if inc_x2>width:
        inc_x2=width
    img1=img[y1:y2, dec_x1:inc_x2]
    # plt.imshow(img1)
    # plt.show()
    cv2.imwrite(path+extn+im_name,img1)



def left_right_optimal_keypoints(left_hip_ankle_dist, left_knee_leg_angle, left_knee_ankle_vertical_angle, left_knee_hip_vertical, right_hip_ankle_dist, right_knee_leg_angle, right_knee_ankle_vertical_angle, right_knee_hip_vertical, trunk_lean, optimal_point, img_name, optimal_image_name, img_ext, alignment,sign=''):
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
    return seq_info
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
                valid_index.append(index)
                frame_valid_knee_leg_angle.append(all_knee_leg_angle[index])
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
def optimal_frame_foot_on_ground_back_front(all_hip_coord, all_knee_coord, all_ankle_coord, all_knee_coord_anotherleg, all_ankle_coord_anotherleg, image_name_list, all_shoulder_coord):
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
    optimal_point = valid_index[sorted_index_valid_another_knee_ankle_disty[0]]
    optimal_image = os.path.basename(image_name_list[optimal_point])
    return optimal_point, optimal_image
def optimal_frame_foot_on_ground_back_front_max_angles(image_name_list,foot_label,angle_list,angle_image_list,or_image_list,type=''):#(all_hip_coord, all_knee_coord, all_ankle_coord, all_knee_coord_anotherleg, all_ankle_coord_anotherleg, image_name_list, all_shoulder_coord, angle_list, type=''):
    optimal_point = 0
    optimal_image = ''
    # print(len(image_name_list))
    # print(angle_image_list[:5])
    # print(len(angle_list))
    # print(len(or_image_list))
    # print(len(angle_list))
    optimal_angle_list=[]
    for image  in image_name_list:
        try:
            index=angle_image_list.index(image)
            optimal_angle_list.append(angle_list[index])
        except:
            pass
    # print(len(image_name_list))
    # print(len(optimal_angle_list))
    # print(angle_list[13])
    # print(optimal_angle_list[:6])
    # print(image_name_list[:6])
    # print(image_name_list[24])
    # print(optimal_angle_list[21])
    if type == 'non_trunk':
        max_negative_angle = 0
        for i in range(len(optimal_angle_list)):
            # image=optimal_image_list[i]
            # print(image)
            angle = optimal_angle_list[i]
            # print(image_name_list[i],angle)
            if angle < 0:
                if angle < max_negative_angle:
                    max_negative_angle = angle
                    # print(max_negative_angle)
                    optimal_point = i
        optimal_image= str(image_name_list[optimal_point])+'.'+str(args.image_ext)
        # optimal_image = os.path.basename(str(image_name_list[optimal_point]))
        # print(optimal_point)
    if type == "trunk":
        max_angle = 0
        for i in range(len(optimal_angle_list)):
            # image = optimal_image_list[i]
            # print(image)
            angle = optimal_angle_list[i]
            # print(image_name_list[i],angle)
            if abs(angle) > abs(max_angle):
                max_angle = angle
                optimal_point = i
        # print(optimal_point)
        optimal_image = str(image_name_list[optimal_point]) + '.'+str(args.image_ext)
        # optimal_image = os.path.basename(str(image_name_list[optimal_point]))
    # angle_index=optimal_angle_list[optimal_point]
    # orignal_optimal_point=angle_list.index(angle_index)
    orignal_optimal_point=image_name_list[optimal_point]-1
    # print(orignal_optimal_point)
    return orignal_optimal_point, optimal_image
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
def image_processing(img_path,points):
    try:
        im = cv2.imread(img_path)
        x1,y1,x2,y2 =image_points(points)
        img = im[y1:y2, x1:x2]
        height, width, _ = img.shape
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,600,apertureSize = 3)
        lines = cv2.HoughLinesP(edges, 2, math.pi/180, 15, 100, 30, 7)
        lines = np.squeeze(lines)
        mid_line = middle_line_image(width,height)
        min_line = (line_length(mid_line))/2
        middle_point = middle_point_coordinates(mid_line)
        line_distance = 1000
        final_angle = 100
        final_line = 0
        final_distance = 0
        for line in lines:
            angle = ang(mid_line,line)
            line_length_= int(line_length(line))
            line_mid_points = middle_point_coordinates(line)
            if angle < 15 and line_length_ > (min_line - 10):
                distance = middle_point[1] -  line_mid_points[1]
                if distance < line_distance:
                    line_distance= distance
                    if angle != 0:
                        if angle < final_angle:
                            final_angle = round(angle,1)
                            final_line = line
                            mid = mid_line
                            final_distance = distance
        return final_angle,mid,final_line,final_distance
    except:
        pass
def image_processing_right(img_path,points,right_side_path,img_name):
    try:
        im = cv2.imread(img_path)
        x1,y1,x2,y2 =image_points(points)
        img = im[y1:y2, x1:x2]
        height, width, _ = img.shape
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,100,600)
        lines = cv2.HoughLinesP(edges, 1, math.pi/180, 1, 5, 11, 7)
        lines = np.squeeze(lines)
        mid_line = middle_line_image(width,height)
        min_line = line_length(mid_line)
        middle_point = middle_point_coordinates(mid_line)
        color = (0, 255, 255)
        thickness = args.line_thickness
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
        newLine_difference =[int(line_select[0]-mid_line[0]),int(line_select[1]-mid_line[1]),int(line_select[2]-mid_line[0]),int(line_select[3]-mid_line[1])]
        newLine = [(new_mid_points[0]+newLine_difference[0]),(new_mid_points[1]+newLine_difference[1]),(new_mid_points[0]+newLine_difference[2]),(new_mid_points[1]+newLine_difference[3])]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        fontColor = (0, 0, 255)
        lineType = 2
        image_original = cv2.imread(right_side_path+''+img_name)
        image_original = cv2.line(image_original, (new_mid_points[0],new_mid_points[1]), (new_mid_points[2],new_mid_points[3]), color, thickness)
        image_original = cv2.line(image_original, (newLine[0],newLine[1]), (newLine[2],newLine[3]), color, thickness)
        image_original = cv2.putText(image_original, str((round(angle_list[0], 1))),
                        (int(newLine[2]+10), int(newLine[3])), font, fontScale, fontColor, lineType)
        cv2.imwrite(right_side_path+''+img_name,image_original)
        final_angle = angle_list[0]
        print(final_angle)
        return final_angle
    except:
        pass

def chest_keypoints(key_points):
    window_size = len(key_points)
    window_size = int(window_size / 2)
    # left_hip_x=key_points[:,11,0]
    # left_hip_y=key_points[:,11,1]
    # right_hip_x=key_points[:,12,0]
    # right_hip_y=key_points[:,12,1]
    # left_shoulder_x=key_points[:,5,0]
    left_shoulder_y=key_points[:,5,1]
    # right_shoulder_x=key_points[:,6,0]
    right_shoulder_y=key_points[:,6,1]
    # center_hip_x=np.add(left_hip_x,right_hip_x)
    # center_hip_x=center_hip_x/2
    # center_hip_y=np.add(left_hip_y,right_hip_y)
    # center_hip_y=center_hip_y/2
    # center_hip=np.stack((center_hip_x,center_hip_y), axis=1)
    # center_shoulder_x=np.add(left_shoulder_x,right_shoulder_x)
    # center_shoulder_x=center_shoulder_x/2
    center_shoulder_y=np.add(left_shoulder_y,right_shoulder_y)
    center_shoulder_y=center_shoulder_y/2
    # center_shoulder=np.stack((center_shoulder_x,center_shoulder_y), axis=1)
    # dist_list = []
    # for i in range(len(key_points)):
    #     dist = np.linalg.norm(center_hip[i] - center_shoulder[i])
    #     dist_list.append(dist)
    if window_size%2==0:
        window_size=window_size+1
    # max_values = max(center_shoulder_y)
    # min_values = min(center_shoulder_y)
    # norm_sholder_y = []
    # for i in center_shoulder_y:
    #     norm_value = (i - min_values) / (max_values - min_values)
    #     norm_sholder_y.append(norm_value)

    shoulder_savgol=savgol_filter(center_shoulder_y,polyorder=3,window_length=window_size)
    sublist = np.array(center_shoulder_y) - np.array(shoulder_savgol)
    min_indis=np.argmin(sublist)
    max_indis=np.argmax(sublist)
    # difference_y = center_shoulder_y[max_indis]-center_shoulder_y[min_indis]
    ratio_diff_y = center_shoulder_y[min_indis]/center_shoulder_y[max_indis]
    ratio_diff_y = round(ratio_diff_y,2)
    y_min = int(center_shoulder_y[min_indis])
    y_max = int(center_shoulder_y[max_indis])
    # ratio_diff_y = "{:.2f}".format(ratio_diff_y)
    # plt.plot(center_shoulder_y)
    # plt.scatter(min_indis, center_shoulder_y[min_indis], color="r")
    # plt.scatter(max_indis, center_shoulder_y[max_indis], color="g")
    # plt.title('max and min of chest in y axis')
    # plt.show()
    # plt.plot(sublist)
    # plt.scatter(min_indis, min(sublist), color="r")
    # plt.scatter(max_indis, max(sublist), color="g")
    # plt.title('normalized and savgol filter')
    # plt.show()
    # print(ratio_diff_y,y_min,y_max)
    min_img=str(min_indis+1)+"."+str(args.image_ext)
    max_img=str(max_indis+1)+"."+str(args.image_ext)
    img_min = cv2.imread(str(args.output_dir)+"/backside/" + "right/" + "images/"+min_img)
    img_max = cv2.imread(str(args.output_dir) + "/backside/" + "right/" + "images/"+max_img)
    model_output_min = model(img_min)
    boxes_min = model_output_min['instances'].pred_boxes
    boxes_min = boxes_min.tensor.cpu().numpy()
    box_min = boxes_min[0]
    box_min = box_min.astype(int)
    y_img_min = box_min[1]
    model_output_max = model(img_max)
    boxes_max = model_output_max['instances'].pred_boxes
    boxes_max = boxes_max.tensor.cpu().numpy()
    box_max = boxes_max[0]
    box_max = box_max.astype(int)
    y_img_max = box_max[1]
    # print(y_img_min,y_img_max)
    # plt.imshow(img_min)
    # plt.show()
    # plt.imshow(img_max)
    # plt.show()
    return min_indis,min_img,max_indis,max_img,ratio_diff_y



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
    for index in sorted_index_valid_another_knee_ankle_disty:
        optimal_image_list.append(valid_index[index])
    min_angle = 100
    image_point_final = None
    mid_points_final = 0
    line_final = 0
    path_final = ''
    distance = 0
    i = 0
    for image in optimal_image_list:
        i = i + 1
        image_points = angle_list[image]
        image_name_base = os.path.basename(image_name_list[image])
        # print(image_name_base)
        image_path = os.path.join(path,str(image_name_base))
        try:
            angle,mid,final_line,final_distance = image_processing(image_path,image_points)
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
        except:
            continue
    if min_angle != 0:
        img = cv2.imread(path_final)
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
            image = cv2.line(img, (new_mid_points[0],new_mid_points[1]), (new_mid_points[2],new_mid_points[3]), color, thickness)
            img = cv2.line(img, (newLine[0],newLine[1]), (newLine[2],newLine[3]), color, thickness)
            img = cv2.putText(img, str((round(min_angle, 1))),
                            (int(newLine[2]+10), int(newLine[3])), font, fontScale, fontColor, lineType)
        try:
            cv2.imwrite(path_final,img)
        except:
            pass
    optimal_image = os.path.basename(image_name_list[optimal_point])
    return optimal_point, optimal_image,min_angle
def read_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps
video_frame_path = ''
def main(args):
    check = 0
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
        if (args.face_blur=='on'):
            generate_frames = 'ffmpeg -i ' + str(args.video_path) + ' -r ' + str(args.input_fps) + ' -qscale:v 2 ' + video_frame_path + '/%d.' + 'PNG'
            print("generate frames is", generate_frames)
            os.system(generate_frames)
            blur = FaceBlurring()
            blur.start_face_blurring(video_frame_path,video_frame_path, args.image_ext,int(args.max_frame))
        else:
            generate_frames = 'ffmpeg -i ' + str(args.video_path)+' -r '+str(args.input_fps) + ' -qscale:v 2 ' + video_frame_path + '/%d.' + str(args.image_ext)
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
        try:
            im_list = sorted(im_list, key=lambda s: int(
                str(s).split("/")[-1][:-4]))
        except:
            im_list = sorted(im_list, key=lambda s: int(
                str(s).split("/")[-1][:-5]))
    json_output = []
    all_image_no = []
    keypoint_no = 0
    image_no = 0
    img_ind = 0
    all_right_knee_leg_angle = []
    write_image_info = []
    all_back_trunk_lean = []
    back_trunk_image =[]
    write_image_back_trunk_lean = []
    all_back_right_hip_left_hip = []
    write_image_back_right_hip_left_hip = []
    write_image_back_left_hip_right_hip = []
    all_back_left_hip_right_hip = []
    all_back_left_knee_hip_vertical = []
    hip_angle_image_left=[]
    write_image_back_left_knee_hip_vertical = []
    all_back_right_knee_hip_vertical = []
    hip_angle_image_right=[]
    write_image_back_right_knee_hip_vertical = []
    write_image_back_left_hip_knee_ankle = []
    all_back_left_knee_leg_angle = []
    knee_angle_image_left = []
    all_back_right_knee_leg_angle = []
    knee_angle_image_right = []
    write_image_back_right_hip_knee_ankle = []
    all_back_left_knee_coor = []
    all_back_left_ankle_coor = []
    all_back_right_knee_coor = []
    all_back_right_ankle_coor = []
    all_back_hip_coor = []
    all_back_shoulder_coor = []
    all_backview_hip_leftankle_distance_y = []
    all_backview_hip_rightankle_distance_y = []
    all_back_chest_yaxis_values = []
    all_back_hip_coor_left_right = []
    all_back_left_feet = []
    all_back_right_feet = []
    all_back_hip_difference = []
    all_front_trunk_lean = []
    front_trunck_image=[]
    write_image_front_trunk_lean = []
    all_front_right_hip_left_hip = []
    write_image_front_right_hip_left_hip = []
    write_image_front_left_hip_right_hip = []
    all_front_left_hip_right_hip = []
    all_front_left_knee_hip_vertical = []
    front_hip_knee_left =[]
    write_image_front_left_knee_hip_vertical = []
    all_front_right_knee_hip_vertical = []
    front_hip_knee_right =[]
    write_image_front_right_knee_hip_vertical = []
    write_image_front_left_hip_knee_ankle = []
    all_front_left_knee_leg_angle = []
    all_front_right_knee_leg_angle = []
    front_right_knee_image = []
    front_left_knee_image = []
    write_image_front_right_hip_knee_ankle = []
    all_front_left_knee_coor = []
    all_front_left_ankle_coor = []
    all_front_right_knee_coor = []
    all_front_right_ankle_coor = []
    all_front_hip_coor = []
    all_front_shoulder_coor = []
    all_frontview_hip_leftankle_distance_y = []
    all_frontview_hip_rightankle_distance_y = []
    all_front_hip_coor_left_right = []
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
    all_rightview_right_pelvis_tilt = []
    waistline_angle_list = []
    waistline_image_list = []
    list_of_all_images = []
    list_of_all_angles = []
    list_of_output_dir = []
    rightview_5frame_leftknee_ankle_distance = [0, 0, 0, 0, 0]
    rightview_5frame_rightknee_ankle_distance = [0, 0, 0, 0, 0]
    right_hip_knee_distance = []
    right_knee_ankle_distance = []
    left_hip_knee_distance = []
    left_knee_ankle_distance = []
    dist_right_hka_all = []
    dist_left_hka_all = []
    left_knee_ymax = []
    right_knee_ymax = []
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
    def draw_line(a, b, image):
        x1 = int(a[0])
        y1 = int(a[1])
        x2 = int(b[0])
        y2 = int(b[1])  # x3=c[0];y3=c[1]
        cv2.line(image, (x1, y1), (x2, y2),
                 (0, 255, 255), args.line_thickness)
        cv2.circle(image, (x1, y1), 5, (0, 255, 255), args.line_thickness)
        cv2.circle(image, (x2, y2), 5, (0, 255, 255), args.line_thickness)
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
        draw_line(rt_hip, rt_knee, rt_image)
        draw_line(lt_hip, lt_knee, lt_image)
        if align == 'left':
            draw_line(lt_shoulder, lt_hip, lt_image)
            draw_line(lt_shoulder, rt_hip, rt_image)
        if align == 'right':
            draw_line(rt_shoulder, lt_hip, lt_image)
            draw_line(rt_shoulder, rt_hip, rt_image)
        draw_line(lt_knee, lt_ankle, lt_image)
        draw_line(rt_knee, rt_ankle, rt_image)
    def return_angles_write_images(value1, value2, value3, output_dir, angle, img_opencv, ang_name, img_name):
        write_image = str(output_dir) + str(img_name) + \
            '.' + args.image_ext
        return write_image, img_opencv
    def calculate_hip_knee_ankle_front_back(hip, knee, ankle, output_dir, img_opencv, ang_name, img_name, ang_dir,
                                            txt_pos):
        angle = calc_angle(np.array(hip), np.array(knee), np.array(ankle))
        angle = (180 - angle) * ang_dir
        write_image, out_img = return_angles_write_images(
            hip, knee, ankle, output_dir, angle, img_opencv, ang_name, img_name)
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
        write_image = str(output_dir) + str(img_name) + \
            '.' + args.image_ext
        return write_image, distance_in_cm, img_opencv
    def calculate_hip_ankle_distance_pixel(hip, ankle):
        dist_in_x = ankle[0] - hip[0]
        dist_in_y = ankle[1] - hip[1]
        distance_in_pixel = math.sqrt(
            dist_in_x * dist_in_x + dist_in_y * dist_in_y)
        return dist_in_x, dist_in_y, distance_in_pixel
    process_starttime = time.time()
    skip_frames = 0
    start_tracking = 20
    num_keypoints = 17
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
    left_image_list = []
    right_image_list = []
    im_list2=im_list.copy()
    foot_model_output = FootModel(args.foot_model_path, 6)
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
        if (i != 0 and i % (skip_frames + 1) != 0 and skip_frames != 0 ):
            pass
        else:
            print(i)
            u.append(i)
            tik = time.time()
            model_output = model(im)
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            tok = time.time()
            print("Model Time: {}ms".format((tok-tik) * 1000))
            postprocess1_start = time.time()
            keypoints = model_output['instances'].pred_keypoints.cpu(
            ).numpy().squeeze().astype(int)
            boxes = model_output['instances'].pred_boxes
            # print(boxes.tensor.cpu().numpy())
            # print(type(boxes))
            # print("@" * 30,"tensor length :", len(boxes))
            if (len(boxes)==0):
                print("null tensor..")
                remove_single_img='rm '+str(im_name)
                os.system(remove_single_img)
                im_list2.remove(im_name)
            else:
                boxes2 = boxes.tensor.cpu().numpy()
                area = boxes.area().cpu().numpy()
                print("printing image location")
                relevant_person = np.argmax(area)
                print("relevant person")
                boxes2 = boxes2[relevant_person]
                boxes2 = boxes2.astype(int)
                cls = foot_model_output.predictfoot(im_rgb, boxes2, margin=20)
                clsvar=cls.numpy()
                if (clsvar == 1):
                    # cv2.imwrite("/opt/detectron2/tools/slowmo2/footground2/"+image_actual_name+".PNG",im)
                    # head_tail= os.path.split(im_name)
                    # image_name=head_tail[1]
                    # image_name_only=image_name.split(".")[0]
                    image_name_only=i+1
                    print("Model detected foot on ground on image : ",image_name_only)
                    left_image_list.append(image_name_only)
                if (clsvar == 2):
                    # cv2.imwrite("/opt/detectron2/tools/slowmo2/footground/" +image_actual_name+ ".PNG",im)
                    # head_tail= os.path.split(im_name)
                    # image_name=head_tail[1]
                    # image_name_only=image_name.split(".")[0]
                    image_name_only=i+1
                    print("Model detected foot on ground on image : ",image_name_only)
                    right_image_list.append(image_name_only)
                num_classes = model_output['instances'].pred_classes
                postprocess1_end = time.time()
                postprocess2_start = time.time()
                if (len(num_classes) == 1):
                    keypoints = [keypoints]
                relevant_keypoint = keypoints[relevant_person]
                num_keypoints = relevant_keypoint.shape[0] # 17x2
                all_image_no.append(im_list2[i])
                vid_keypoints.append(relevant_keypoint)
    # right_image_list = list(map(int, right_image_list))
    # left_image_list = list(map(int, left_image_list))
    print(len(right_image_list))
    print(len(left_image_list))
    first_right_image=right_image_list[0]
    first_left_image=left_image_list[0]
    image_no = 0
    umax+=1
    vid_keypoints = np.array(vid_keypoints)
    raw_keypoints = vid_keypoints.copy().astype(np.int)[:,:,:2]
    total_keypoints = len(vid_keypoints)
    tik = time.time()
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
    # print('FPS: ',filter_input_fps)
    tmp_cap.release()
    proc_vid_keypoints = keypoints_post_processing.smooth_hip_points(vid_keypoints,filter_input_fps)
    # proc_vid_keypoints[:,13:15,:] = savgol_filter(proc_vid_keypoints[:,13:15,:],polyorder=1,window_length=3,axis=0)
    # for _ in range(0,int(filter_input_fps/15)+1):
    #     proc_vid_keypoints[:,13:15,:] = savgol_filter(proc_vid_keypoints[:,13:15,:],polyorder=1,window_length=3,axis=0)
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
    np.save('kpts',interpolated_keypoints_final)
    save_videos = False
    if save_videos:
        import kpts_video_maker
        print('saving debug videos...')
        kpts_video_maker.to_video(args.video_path,raw_keypoints,'debug_vids/'+video_name[:-3]+'_raw.avi')
        kpts_video_maker.to_video(args.video_path,interpolated_keypoints_final.astype(np.int),'debug_vids/'+video_name[:-3]+'_processed.avi')
        print('done')
        return
    tok = time.time()
    print("PolySmoothingIntrp Average Per Frame Time: {:.2f}us".format((tok-tik) * 1000000/umax))
    print("PolySmoothingIntrp Total Time: {:.2f}ms".format((tok-tik) * 1000))
    print('Performing  Calculations on all keypoints...')
    waistline_color_cluster = None
    femur_angle_image_list=[]
    for i, im_name in tqdm(enumerate(im_list2)):
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
        shoulder_x = (right_shoulder[0] + left_shoulder[0]) / 2
        # print(left_hip,left_knee,right_hip,right_knee)
        # distance_left_hip_knee = np.sqrt((left_hip[0] - left_knee[0]) ** 2 + (left_hip[1] - left_knee[1]) ** 2)
        # print(distance_left_hip_knee)
        # img_opencv_right = cv2.imread(str(im_list2[i]))
        # img_opencv_right=cv2.line(img_opencv_right, tuple(left_hip), tuple(left_knee), (0, 255, 0), 2)
        # plt.imshow(img_opencv_right)
        # plt.show()
        if (image_no == 1 and args.input_type == 'video') or (args.pose_detection == 'on') or (args.input_type == 'image'):
            test_image  = cv2.imread(str(im_list2[i]))
            model_output = model(test_image)
            boxes_test = model_output['instances'].pred_boxes
            boxes_test = boxes_test.tensor.cpu().numpy()
            boxes_test = boxes_test[0].astype(int)
            print(boxes_test)
            x1_test = boxes_test[0]
            x2_test = boxes_test[2]
            y1_test = boxes_test[1]
            y2_test = boxes_test[3]
            img_t = test_image[y1_test:y2_test, x1_test:x2_test]
            bbox_diff = abs(x1_test-x2_test)
            shoulder_diff = abs(left_shoulder[0] - right_shoulder[0])
            print(bbox_diff,shoulder_diff)
            ratio_bbox= shoulder_diff/bbox_diff
            # plt.imshow(img_t)
            # plt.show()
            if (left_shoulder[0] < nose[0] and right_shoulder[0] > nose[0] and ratio_bbox > 0.37):
                alignment = "back_side"
                print("back_side")
            elif (left_knee[0] > right_knee[0]) and  ratio_bbox > 0.4:# and (left_hip[0] - right_hip[0]) > (left_ankle[0] - right_ankle[0])):
                alignment = "front_side"
                print("front_side")
            else:
                if (nose[0] - shoulder_x) > 0:
                    print("right_side")
                    alignment = "right_side"
                if (nose[0] - shoulder_x) < 0:
                    print("left_side")
                    alignment = "left_side"
        all_is = []
        # break
        print ("\n"*20,"alignment:",alignment,"\n"*20)
        if alignment == "back_side":  # or alignment=="front_side":
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
                back_trunk_lean_path = str(
                    args.output_dir) + "/backside/" + "trunk_lean/"
                back_side_left_image_path = str(args.output_dir) + \
                    "/backside/" + "left/" + "images/"
                back_side_right_image_path = str(args.output_dir) + \
                    "/backside/" + "right/" + "images/"
                back_both_side = str(
                    args.output_dir) + "backside/" + "both_side/" + "images/"
                os.makedirs(back_side_right_image_path)
                os.makedirs(back_side_left_image_path)
                os.makedirs(back_both_side)
                back_folder_created = True
            img_opencv_right = cv2.imread(str(im_list2[i]))
            from waistline_modules import waistline
            if args.front_back_pelvis=='on':
                _,waistline_angle =  waistline.draw_waistline_front_back(img_opencv_right,args.line_thickness, pts=relevant_keypoint, clt=waistline_color_cluster,
                                                                     angle_font_params=(font,fontScale,fontColor,lineType))
                waistline_angle_list.append(waistline_angle)
                waistline_image_list.append(i+1)
            else:
                waistline_angle = 0
            img_opencv_left = img_opencv_right.copy()
            if True:
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
                    knee_angle_image_right.append(i+1)
                    all_back_right_knee_leg_angle.append(angle)
                    keyangle_dict['right_knee_varus_valgus'] = round(angle, 1)
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
                    knee_angle_image_left.append(i+1)
                    all_back_left_knee_leg_angle.append(angle)
                    keyangle_dict['left_knee_varus_valgus'] = round(angle, 1)
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
                    hip_angle_image_right.append(i+1)
                    keyangle_dict['right_femur_angle'] = round(angle, 1)
                if len(left_knee) > 0 and len(left_hip) > 0:
                    if left_knee[0] >= left_hip[0]:
                        write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                            left_knee, left_hip, back_left_knee_hip_vertical_path, img_opencv_left, "Femur Angle", image_actual_name, -1, 'right')
                    if left_knee[0] < left_hip[0]:
                        write_image, angle, img_opencv_left = calculate_knee_hip_vertical_front_back(
                            left_knee, left_hip, back_left_knee_hip_vertical_path, img_opencv_left, "Femur Angle", image_actual_name, 1, 'right')
                    write_image_back_left_knee_hip_vertical.append(write_image)
                    all_back_left_knee_hip_vertical.append(angle)
                    hip_angle_image_left.append(i+1)
                    keyangle_dict['left_femur_angle'] = round(angle, 1)
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
                    back_trunk_image.append(i+1)
                    keyangle_dict['trunk_lateral_lean'] = round(angle, 1)
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
                shoulder_mean = []
                shoulder_mean.append(shoulder_meanx)
                shoulder_mean.append(
                    (left_shoulder[1] + right_shoulder[1]) / 2)
                all_back_left_knee_coor.append(left_knee)
                all_back_left_ankle_coor.append(left_ankle)
                all_back_right_knee_coor.append(right_knee)
                all_back_right_ankle_coor.append(right_ankle)
                all_back_hip_coor.append(hip_mean)
                all_back_shoulder_coor.append(shoulder_mean)
                hip_coor_left_right = []
                hip_coor_left_right.append(left_hip[0])
                hip_coor_left_right.append((left_hip[1]+left_shoulder[1])/2)
                hip_coor_left_right.append(right_hip[0])
                hip_coor_left_right.append(right_hip[1])
                all_back_hip_coor_left_right.append(hip_coor_left_right)
                feet_coordinate_left = []
                distance_left_y = (left_knee[1] + left_ankle[1])/2
                distance_left_image = (left_ankle[1] - left_knee[1])
                feet_coordinate_left.append(left_knee[0] - (distance_left_image/2))
                feet_coordinate_left.append(distance_left_y)
                feet_coordinate_left.append(left_ankle[0] + (distance_left_image/2))
                feet_coordinate_left.append(left_ankle[1] + distance_left_image)
                all_back_left_feet.append(feet_coordinate_left)
                feet_coordinate_right = []
                distance_right_y = (right_knee[1] + right_ankle[1])/2
                distance_right_image = ( right_ankle[1] - right_knee[1])
                feet_coordinate_right.append(right_knee[0] - (distance_right_image/2))
                feet_coordinate_right.append(distance_right_y)
                feet_coordinate_right.append(right_ankle[0] + (distance_right_image/2))
                feet_coordinate_right.append(right_ankle[1] + distance_right_image)
                all_back_right_feet.append(feet_coordinate_right)
                chest_y_axis = []
                chest_y_axis.append(str(image_actual_name) + "." + args.image_ext)
                chest_y_axis.append(shoulder_mean[1])
                all_back_chest_yaxis_values.append(chest_y_axis)
                distance_hip = []
                dist_x_hip = right_hip[0] - left_hip[0]
                distance_hip.append(dist_x_hip)
                dist_y_hip = right_hip[1] - left_hip[1]
                distance_hip.append(dist_y_hip)
                all_back_hip_difference.append(distance_hip)
            cv2.imwrite(str(back_side_left_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_left)
            # path_for_left = str(back_side_left_image_path) + str(image_actual_name)+ "." + args.image_ext
            # left_image_list.append(path_for_left)
            cv2.imwrite(str(back_side_right_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_right)
            # path_for_right = str(back_side_right_image_path) + str(image_actual_name)+ "." + args.image_ext
            # right_image_list.append(path_for_right)
            print(str(back_both_side) + str(image_actual_name) + "." + args.image_ext)
            cv2.imwrite(str(back_both_side) + str(image_actual_name)
                        + "." + args.image_ext, np.concatenate((img_opencv_left, img_opencv_right), axis=1))
            seq_dict['frame_number'] = str(
                image_actual_name) + "." + args.image_ext
            seq_dict['keyangle'] = keyangle_dict
            seq_dict['pose'] = alignment
            seq_list.append(seq_dict)
        if alignment == "front_side":  # or alignment=="front_side":
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
                front_trunk_lean_path = str(
                    args.output_dir) + "/frontside/" + "trunk_lean/"
                front_left_image_path = str(
                    args.output_dir) + "/frontside/" + "left/" + "images/"
                front_right_image_path = str(
                    args.output_dir) + "/frontside/" + "right/" + "images/"
                front_both_side = str(
                    args.output_dir) + "/frontside/" + "both_side/" + "images/"
                os.makedirs(front_left_image_path)
                os.makedirs(front_right_image_path)
                os.makedirs(front_both_side)
                front_folder_created = True
            img_opencv_right = cv2.imread(str(im_list2[i]))
            from waistline_modules import waistline
            if args.front_back_pelvis=='on':
                _,waistline_angle =  waistline.draw_waistline_front_back(img_opencv_right,args.line_thickness, pts=relevant_keypoint, clt=waistline_color_cluster,
                                                                     angle_font_params=(font,fontScale,fontColor,lineType))
                waistline_angle_list.append(waistline_angle)
                waistline_image_list.append(i+1)
            else:
                waistline_angle=0
            img_opencv_left = img_opencv_right.copy()
            if True:
                if True:
                    if True:
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
                            # print(write_image)
                            write_image_front_right_hip_knee_ankle.append(
                                write_image)
                            all_front_right_knee_leg_angle.append(angle)
                            front_right_knee_image.append(i+1)
                            keyangle_dict['right_knee_varus_valgus'] = round(
                                angle, 1)
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
                            front_left_knee_image.append(i+1)
                            keyangle_dict['left_knee_varus_valgus'] = round(
                                angle, 1)
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
                            front_hip_knee_right.append(i+1)
                            keyangle_dict['right_femur_angle'] = round(
                                angle, 1)
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
                            front_hip_knee_left.append(i+1)
                            keyangle_dict['left_femur_angle'] = round(angle, 1)
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
                                write_image)
                            all_front_left_hip_right_hip.append(waistline_angle)
                            keyangle_dict['pelvis_angle'] = round(waistline_angle, 1)
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
                            front_trunck_image.append(i+1)
                            keyangle_dict['trunk_lateral_lean'] = round(
                                angle, 1)
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
                        hip_coor_left_right = []
                        hip_coor_left_right.append(right_hip[0])
                        hip_coor_left_right.append((right_hip[1]+right_shoulder[1])/2)
                        hip_coor_left_right.append(left_hip[0])
                        hip_coor_left_right.append(left_hip[1])
                        all_front_hip_coor_left_right.append(hip_coor_left_right)
            cv2.imwrite(str(front_left_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_left)
            # path_for_left = str(front_left_image_path) + str(image_actual_name)+ "." + args.image_ext
            # left_image_list.append(path_for_left)
            cv2.imwrite(str(front_right_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_right)
            # path_for_right = str(front_right_image_path) + str(image_actual_name)+ "." + args.image_ext
            # right_image_list.append(path_for_right)
            cv2.imwrite(str(front_both_side) + str(image_actual_name)
                        + "." + args.image_ext, np.concatenate((img_opencv_right,img_opencv_left), axis=1))
            seq_dict['frame_number'] = str(
                image_actual_name) + "." + args.image_ext
            seq_dict['keyangle'] = keyangle_dict
            seq_dict['pose'] = alignment
            seq_list.append(seq_dict)
        if alignment == "left_side":  # or alignment=="front_side":
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
                os.makedirs(leftview_left_image_path)
                os.makedirs(leftview_right_image_path)
                left_folder_created = True
            img_opencv_left = cv2.imread(str(im_list2[i]))
            from waistline_modules import waistline
            # if args.side_pelvis == 'on':
            #     _,waistline_angle =  waistline.draw_waistline_side(img_opencv_left, pts=relevant_keypoint, clt=waistline_color_cluster,angle_font_params=(font,fontScale,fontColor,lineType))
            # else:
            #     waistline_angle=0
            img_opencv_right = img_opencv_left.copy()
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
                        if img_ind >= 5 and check_valid_ankle_point(right_knee, right_ankle, leftview_5frame_rightknee_ankle_distance) == 'notvalid' and check_valid_ankle_point(left_knee, left_ankle, leftview_5frame_leftknee_ankle_distance) == 'notvalid':
                            leftside_pass_flag = True
                        if leftside_pass_flag == True:
                            cv2.imwrite(str(leftview_left_image_path)
                                        + str(image_actual_name) + "." + args.image_ext, img_opencv_left)
                            # path_for_left=str(leftview_left_image_path)+ str(image_actual_name) + "." + args.image_ext
                            # left_image_list.append(path_for_left)

                            cv2.imwrite(str(leftview_right_image_path)
                                        + str(image_actual_name) + "." + args.image_ext, img_opencv_right)
                            # path_for_right=str(leftview_right_image_path)+ str(image_actual_name) + "." + args.image_ext
                            # right_image_list.append(path_for_right)
                            seq_dict['frame_number'] = str(
                                image_actual_name) + "." + args.image_ext
                            seq_dict['keyangle'] = keyangle_dict
                            seq_dict['pose'] = alignment
                            seq_list.append(seq_dict)
                            continue
                        right_hip[0] = (left_hip[0] + right_hip[0]) / 2
                        right_hip[1] = (left_hip[1] + right_hip[1]) / 2
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
                        img_ind = img_ind + 1
            cv2.imwrite(str(leftview_left_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_left)
            # path_for_left= str(leftview_left_image_path) + str(image_actual_name)+ "." + args.image_ext
            # left_image_list.append(path_for_left)
            cv2.imwrite(str(leftview_right_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_right)
            # path_for_right= str(leftview_right_image_path) + str(image_actual_name)+ "." + args.image_ext
            # right_image_list.append(path_for_right)
            seq_dict['frame_number'] = str(
                image_actual_name) + "." + args.image_ext
            seq_dict['keyangle'] = keyangle_dict
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
                os.makedirs(rightview_left_image_path)
                os.makedirs(rightview_right_image_path)
                right_folder_created = True
            img_opencv_left = cv2.imread(str(im_list2[i]))
            from waistline_modules import waistline
            # if args.side_pelvis=='on':
            #     _,waistline_angle =  waistline.draw_waistline_side(img_opencv_left, pts=relevant_keypoint, clt=waistline_color_cluster,
            #                                                          angle_font_params=(font,fontScale,fontColor,lineType))
            # else:
            #     wistline_angle=0
            img_opencv_right = img_opencv_left.copy()
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
                            # path_for_left=str(rightview_left_image_path)+ str(image_actual_name) + "." + args.image_ext
                            # left_image_list.append(path_for_left)
                            cv2.imwrite(str(rightview_right_image_path)
                                        + str(image_actual_name) + "." + args.image_ext, img_opencv_right)
                            # path_for_right=str(rightview_right_image_path)+ str(image_actual_name) + "." + args.image_ext
                            # right_image_list.append(path_for_right)
                            print('image saved is ---', image_actual_name,
                                  '-- with value of i is ----', i)
                            print('full path of image saved is -------', str(rightview_left_image_path)
                                  + str(image_actual_name) + "." + args.image_ext)
                            print('full path of image saved is -------', str(rightview_right_image_path)
                                  + str(image_actual_name) + "." + args.image_ext)
                            seq_dict['frame_number'] = str(
                                image_actual_name) + "." + args.image_ext
                            seq_dict['keyangle'] = keyangle_dict
                            seq_dict['pose'] = alignment
                            seq_list.append(seq_dict)
                            continue
                        left_hip[0] = (left_hip[0] + right_hip[0]) / 2
                        left_hip[1] = (left_hip[1] + right_hip[1]) / 2
                        all_is.append(i)
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
                            all_rightview_right_shoulder_coord.append(
                                right_shoulder)
                            all_rightview_left_knee_coord.append(left_knee)
                            all_rightview_left_ankle_coord.append(left_ankle)
                            all_rightview_right_knee_coord.append(right_knee)
                            all_rightview_right_ankle_coord.append(right_ankle)
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
            # path_for_left=str(rightview_left_image_path) + str(image_actual_name)+ "." + args.image_ext
            # left_image_list.append(path_for_left)
            cv2.imwrite(str(rightview_right_image_path) + str(image_actual_name)
                        + "." + args.image_ext, img_opencv_right)
            # path_for_right=str(rightview_right_image_path) + str(image_actual_name)+ "." + args.image_ext
            # right_image_list.append(path_for_right)
            seq_dict['frame_number'] = str(
                image_actual_name) + "." + args.image_ext
            seq_dict['keyangle'] = keyangle_dict
            seq_dict['pose'] = alignment
            seq_list.append(seq_dict)
    left_video_view_side = ""
    right_video__view_side = ""
    if alignment == "right_side":
        left_video_view_side = "rightviewside_leftextremity"
        right_video__view_side = "rightviewside_rightextremity"
        out_put_image_dir = str(args.output_dir) + "rightviewside/"
        json_folder = str(args.output_dir) + \
            "rightviewside/" + video_name + ".json"
        optimal_json_name = str(args.output_dir) + "rightviewside/" + \
            video_name + "_optimalFrame.json"
    elif alignment == "left_side":
        left_video_view_side = "leftviewside_leftextremity"
        right_video__view_side = "leftviewside_rightextremity"
        out_put_image_dir = str(args.output_dir) + "leftviewside/"
        json_folder = str(args.output_dir) + \
            "leftviewside/" + video_name + ".json"
        optimal_json_name = str(args.output_dir) + "leftviewside/" + \
            video_name + "_optimalFrame.json"
    if alignment == "front_side":
        left_video_view_side = "frontviewside_leftextremity"
        right_video__view_side = "frontviewside_rightextremity"
        out_put_image_dir = str(args.output_dir) + "frontside/"
        json_folder = str(args.output_dir) + "frontside/" + \
            video_name + ".json"
        optimal_json_name = str(args.output_dir) + \
            "frontside/" + video_name + "_optimalFrame.json"
    elif alignment == "back_side":
        left_video_view_side = "backviewside_leftextremity"
        right_video__view_side = "backviewside_rightextremity"
        out_put_image_dir = str(args.output_dir) + "backside/"
        json_folder = str(args.output_dir) + "backside/" + video_name + ".json"
        optimal_json_name = str(args.output_dir) + \
            "backside/" + video_name + "_optimalFrame.json"
    now_date_time = datetime.now()
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
    if args.auto_fps=='on':
        if args.raw_video == 'normal':
            left_shell_script = 'ffmpeg -framerate ' + str(read_fps(args.video_path)) + ' -i ' + str(out_put_image_dir) + 'left/images/' + '/%d.' + \
                args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
                str(out_put_image_dir) + 'left/' + 'movie' + '/Video_' + \
                left_video_view_side + '_' + video_name + '_' + creation_date + '.mp4'
            right_shell_script = 'ffmpeg -framerate ' + str(read_fps(args.video_path)) + ' -i ' + str(out_put_image_dir) + 'right/images/' + '/%d.' + \
                args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
                str(out_put_image_dir) + 'right/' + 'movie' + '/Video_' + \
                right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'
            if alignment == "back_side" or alignment == "front_side":
                both_shell_script = 'ffmpeg -framerate ' + str(read_fps(args.video_path)) + ' -i ' + str(out_put_image_dir) + 'both_side/images' + '/%d.' + \
                                    args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
                                    str(out_put_image_dir) + 'both_side/Video_merged_' + \
                                    left_video_view_side + '_and_' + right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'
                os.system(both_shell_script)
            print('raw_video =normal and left_shell_script----', left_shell_script)
            print('raw_video =normal and right_shell_script----', right_shell_script)
            os.system(left_shell_script)
            os.system(right_shell_script)
        if args.raw_video == 'raw':
            left_shell_script = 'ffmpeg -framerate ' + str(read_fps(args.video_path)) + ' -i ' + str(out_put_image_dir) + 'left/images/' + \
                '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
                str(out_put_image_dir) + 'left/' + 'movie' + '/Video_' + \
                left_video_view_side + '_' + video_name + '_' + creation_date + '.mp4'
            right_shell_script = 'ffmpeg -framerate ' + str(read_fps(args.video_path)) + ' -i ' + str(out_put_image_dir) + 'right/images/' + \
                '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
                str(out_put_image_dir) + 'right/' + 'movie' + '/Video_' + \
                right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'
            if alignment == "back_side" or alignment == "front_side":
                both_side_shell_script = 'ffmpeg -framerate ' + str(read_fps(args.video_path)) + ' -i ' + str(out_put_image_dir) + 'both_side/images' + \
                                     '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
                                     str(out_put_image_dir) + 'both_side/Video_' + \
                                     left_video_view_side + '_and_' + right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'
                os.system(both_side_shell_script)
            print('raw_video =raw and left_shell_script----', left_shell_script)
            print('raw_video =raw and right_shell_script----', right_shell_script)
            os.system(left_shell_script)
            os.system(right_shell_script)
    else:
        if args.raw_video == 'normal':
            left_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(
                out_put_image_dir) + 'left/images/' + '/%d.' + \
                                args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
                                str(out_put_image_dir) + 'left/' + 'movie' + '/Video_' + \
                                left_video_view_side + '_' + video_name + '_' + creation_date + '.mp4'
            right_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(
                out_put_image_dir) + 'right/images/' + '/%d.' + \
                                 args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
                                 str(out_put_image_dir) + 'right/' + 'movie' + '/Video_' + \
                                 right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'
            if alignment == "back_side" or alignment == "front_side":
                both_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(
                    out_put_image_dir) + 'both_side/images' + '/%d.' + \
                                    args.image_ext + ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + \
                                    str(out_put_image_dir) + 'both_side/Video_merged_' + \
                                    left_video_view_side + '_and_' + right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'
                os.system(both_shell_script)
            print('raw_video =normal and left_shell_script----', left_shell_script)
            print('raw_video =normal and right_shell_script----', right_shell_script)
            os.system(left_shell_script)
            os.system(right_shell_script)
        if args.raw_video == 'raw':
            left_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(
                out_put_image_dir) + 'left/images/' + \
                                '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
                                str(out_put_image_dir) + 'left/' + 'movie' + '/Video_' + \
                                left_video_view_side + '_' + video_name + '_' + creation_date + '.mp4'
            right_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(
                out_put_image_dir) + 'right/images/' + \
                                 '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
                                 str(out_put_image_dir) + 'right/' + 'movie' + '/Video_' + \
                                 right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'
            if alignment == "back_side" or alignment == "front_side":
                both_side_shell_script = 'ffmpeg -framerate ' + str(args.input_fps) + ' -i ' + str(
                    out_put_image_dir) + 'both_side/images' + \
                                         '/%d.' + args.image_ext + ' -c:v libx264 -crf 0 ' + \
                                         str(out_put_image_dir) + 'both_side/Video_' + \
                                         left_video_view_side + '_and_' + right_video__view_side + '_' + video_name + '_' + creation_date + '.mp4'
                os.system(both_side_shell_script)
            print('raw_video =raw and left_shell_script----', left_shell_script)
            print('raw_video =raw and right_shell_script----', right_shell_script)
            os.system(left_shell_script)
            os.system(right_shell_script)
    if (int(args.output_option) == 4):
        if alignment == "left_side":  # or alignment=="right_side":

            if args.side_pelvis == 'on':
                optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(
                    all_leftview_hip_leftankle_distance_y, all_leftview_hip_leftankle_distance_x, 'positive',
                    write_image_leftview_right_hip_knee_ankle,
                    all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_left_knee_leg_angle,
                    all_leftview_left_ankle_coord, all_leftview_right_ankle_coord)
                # print(img_leftfoot_leave)
                waistline_image_line=cv2.imread(str(out_put_image_dir)+'left/images/'+str(img_leftfoot_leave))
                orignal_image_left_foot = cv2.imread(str(im_list2[optimal_leftfoot_leaving_ground]))
                _,waistline_angle =  waistline.draw_waistline_side(orignal_image_left_foot,waistline_image_line, pts=interpolated_keypoints_final[optimal_leftfoot_leaving_ground], clt=waistline_color_cluster,angle_font_params=(font,fontScale,fontColor,lineType))
                cv2.imwrite(str(out_put_image_dir)+'left/images/'+str(img_leftfoot_leave),waistline_image_line)



            optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(
                all_leftview_hip_leftankle_distance_y, all_leftview_hip_leftankle_distance_x, 'positive',
                write_image_leftview_right_hip_knee_ankle,
                all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_left_knee_leg_angle,
                all_leftview_left_ankle_coord, all_leftview_right_ankle_coord)

            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist,
                                                          all_leftview_left_knee_leg_angle,
                                                          all_leftview_left_knee_ankle_vertical_angle,
                                                          all_leftview_left_knee_hip_vertical,
                                                          all_leftview_right_hip_ankle_dist,
                                                          all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle,
                                                          all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean,
                                                          optimal_leftfoot_leaving_ground, img_leftfoot_leave,
                                                          'Phase_Dectection_leftsideview_leftextremity_toe_off_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_leftfoot_touching_ground, img_leftfoot_touch = optimal_frame_foot_touching_ground(
                all_leftview_hip_leftankle_distance_y, all_leftview_hip_leftankle_distance_x, 'negative',
                write_image_leftview_right_hip_knee_ankle,
                all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_left_knee_leg_angle,
                all_leftview_left_ankle_coord, all_leftview_right_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist,
                                                          all_leftview_left_knee_leg_angle,
                                                          all_leftview_left_knee_ankle_vertical_angle,
                                                          all_leftview_left_knee_hip_vertical,
                                                          all_leftview_right_hip_ankle_dist,
                                                          all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle,
                                                          all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean,
                                                          optimal_leftfoot_touching_ground, img_leftfoot_touch,
                                                          'Phase_Dectection_leftsideview_leftextremity_initial_contact_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            if args.side_pelvis == 'on':
                optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(
                    all_leftview_hip_rightankle_distance_y, all_leftview_hip_rightankle_distance_x, 'positive',
                    write_image_leftview_right_hip_knee_ankle,
                    all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_right_knee_leg_angle,
                    all_leftview_right_ankle_coord, all_leftview_left_ankle_coord)
                waistline_image_line=cv2.imread(str(out_put_image_dir)+'right/images/'+str(img_rightfoot_leave))
                orignal_image_right_foot = cv2.imread(str(im_list2[optimal_rightfoot_leaving_ground]))
                _,waistline_angle =  waistline.draw_waistline_side(orignal_image_right_foot,waistline_image_line, pts=interpolated_keypoints_final[optimal_rightfoot_leaving_ground], clt=waistline_color_cluster,angle_font_params=(font,fontScale,fontColor,lineType))
                cv2.imwrite(str(out_put_image_dir)+'right/images/'+str(img_rightfoot_leave),waistline_image_line)

            optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(
                all_leftview_hip_rightankle_distance_y, all_leftview_hip_rightankle_distance_x, 'positive',
                write_image_leftview_right_hip_knee_ankle,
                all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_right_knee_leg_angle,
                all_leftview_right_ankle_coord, all_leftview_left_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist,
                                                          all_leftview_left_knee_leg_angle,
                                                          all_leftview_left_knee_ankle_vertical_angle,
                                                          all_leftview_left_knee_hip_vertical,
                                                          all_leftview_right_hip_ankle_dist,
                                                          all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle,
                                                          all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean,
                                                          optimal_rightfoot_leaving_ground, img_rightfoot_leave,
                                                          'Phase_Dectection_leftsideview_rightextremity_toe_off_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_rightfoot_touching_ground, img_rightfoot_touch = optimal_frame_foot_touching_ground(
                all_leftview_hip_rightankle_distance_y, all_leftview_hip_rightankle_distance_x, 'negative',
                write_image_leftview_right_hip_knee_ankle, all_leftview_left_hip_coord,
                all_leftview_left_shoulder_coord, all_leftview_right_knee_leg_angle, all_leftview_right_ankle_coord,
                all_leftview_left_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist,
                                                          all_leftview_left_knee_leg_angle,
                                                          all_leftview_left_knee_ankle_vertical_angle,
                                                          all_leftview_left_knee_hip_vertical,
                                                          all_leftview_right_hip_ankle_dist,
                                                          all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle,
                                                          all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean,
                                                          optimal_rightfoot_touching_ground, img_rightfoot_touch,
                                                          'Phase_Dectection_leftsideview_rightextremity_initial_contact_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground = optimal_frame_max_knee_flexion(
                all_leftview_left_hip_coord, all_leftview_right_knee_coord, all_leftview_right_ankle_coord,
                all_leftview_right_knee_leg_angle,
                all_leftview_left_knee_coord, all_leftview_left_ankle_coord, write_image_leftview_right_hip_knee_ankle,
                all_leftview_left_shoulder_coord, all_leftview_hip_rightankle_distance_y)
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist,
                                                          all_leftview_left_knee_leg_angle,
                                                          all_leftview_left_knee_ankle_vertical_angle,
                                                          all_leftview_left_knee_hip_vertical,
                                                          all_leftview_right_hip_ankle_dist,
                                                          all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle,
                                                          all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean,
                                                          optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground,
                                                          'Phase_Dectection_leftsideview_rightextremity_midstance_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground = optimal_frame_max_knee_flexion(
                all_leftview_left_hip_coord, all_leftview_left_knee_coord, all_leftview_left_ankle_coord,
                all_leftview_left_knee_leg_angle,
                all_leftview_right_knee_coord, all_leftview_right_ankle_coord,
                write_image_leftview_right_hip_knee_ankle, all_leftview_left_shoulder_coord,
                all_leftview_hip_leftankle_distance_y)
            knee = all_leftview_left_knee_coord[optimal_knee_flexion_leftfoot_ground][0]
            ankle = all_leftview_left_ankle_coord[optimal_knee_flexion_leftfoot_ground][0]
            if knee > ankle:
                sign = 'Yes'
            else:
                sign = 'No'
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist,
                                                          all_leftview_left_knee_leg_angle,
                                                          all_leftview_left_knee_ankle_vertical_angle,
                                                          all_leftview_left_knee_hip_vertical,
                                                          all_leftview_right_hip_ankle_dist,
                                                          all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle,
                                                          all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean,
                                                          optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground,
                                                          'Phase_Dectection_leftsideview_leftextremity_midstance_phase',
                                                          args.image_ext, alignment,sign)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
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

            if args.side_pelvis == 'on':
                optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(
                    all_rightview_hip_leftankle_distance_y, all_rightview_hip_leftankle_distance_x, 'negative',
                    write_image_rightview_right_hip_knee_ankle,
                    all_rightview_right_hip_coord, all_rightview_right_shoulder_coord,
                    all_rightview_left_knee_leg_angle,
                    all_rightview_left_ankle_coord, all_rightview_right_ankle_coord)
                waistline_image_line=cv2.imread(str(out_put_image_dir)+'left/images/'+str(img_leftfoot_leave))
                orignal_image_left_foot = cv2.imread(str(im_list2[optimal_leftfoot_leaving_ground]))
                _,waistline_angle =  waistline.draw_waistline_side(orignal_image_left_foot,waistline_image_line, pts=interpolated_keypoints_final[optimal_leftfoot_leaving_ground], clt=waistline_color_cluster,angle_font_params=(font,fontScale,fontColor,lineType))
                cv2.imwrite(str(out_put_image_dir)+'left/images/'+str(img_leftfoot_leave),waistline_image_line)
            optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(
                all_rightview_hip_leftankle_distance_y, all_rightview_hip_leftankle_distance_x, 'negative',
                write_image_rightview_right_hip_knee_ankle,
                all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_left_knee_leg_angle,
                all_rightview_left_ankle_coord, all_rightview_right_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist,
                                                          all_rightview_left_knee_leg_angle,
                                                          all_rightview_left_knee_ankle_vertical_angle,
                                                          all_rightview_left_knee_hip_vertical,
                                                          all_rightview_right_hip_ankle_dist,
                                                          all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle,
                                                          all_rightview_right_knee_hip_vertical,
                                                          all_rightview_trunk_lean, optimal_leftfoot_leaving_ground,
                                                          img_leftfoot_leave,
                                                          'Phase_Dectection_rightsideview_leftextremity_toe_off_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_leftfoot_touching_ground, img_leftfoot_touch = optimal_frame_foot_touching_ground(
                all_rightview_hip_leftankle_distance_y, all_rightview_hip_leftankle_distance_x, 'positive',
                write_image_rightview_right_hip_knee_ankle,
                all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_left_knee_leg_angle,
                all_rightview_left_ankle_coord, all_rightview_right_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist,
                                                          all_rightview_left_knee_leg_angle,
                                                          all_rightview_left_knee_ankle_vertical_angle,
                                                          all_rightview_left_knee_hip_vertical,
                                                          all_rightview_right_hip_ankle_dist,
                                                          all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle,
                                                          all_rightview_right_knee_hip_vertical,
                                                          all_rightview_trunk_lean, optimal_leftfoot_touching_ground,
                                                          img_leftfoot_touch,
                                                          'Phase_Dectection_rightsideview_leftextremity_initial_contact_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            if args.side_pelvis == 'on':
                optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(
                    all_rightview_hip_rightankle_distance_y, all_rightview_hip_rightankle_distance_x, 'negative',
                    write_image_rightview_right_hip_knee_ankle,
                    all_rightview_right_hip_coord, all_rightview_right_shoulder_coord,
                    all_rightview_right_knee_leg_angle, all_rightview_right_ankle_coord, all_rightview_left_ankle_coord)
                waistline_image_line = cv2.imread(str(out_put_image_dir) + 'right/images/' + str(img_rightfoot_leave))
                orignal_image_right_foot = cv2.imread(str(im_list2[optimal_rightfoot_leaving_ground]))
                _, waistline_angle = waistline.draw_waistline_side(orignal_image_right_foot,waistline_image_line,
                                                                   pts=interpolated_keypoints_final[
                                                                       optimal_rightfoot_leaving_ground],
                                                                   clt=waistline_color_cluster, angle_font_params=(
                    font, fontScale, fontColor, lineType))
                cv2.imwrite(str(out_put_image_dir) + 'right/images/' + str(img_rightfoot_leave), waistline_image_line)
                # plt.imshow(waistline_image_line)
                # plt.show()
                # plt.imshow(orignal_image_right_foot)
                # plt.show()

            optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(
                all_rightview_hip_rightankle_distance_y, all_rightview_hip_rightankle_distance_x, 'negative',
                write_image_rightview_right_hip_knee_ankle,
                all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_right_knee_leg_angle,
                all_rightview_right_ankle_coord, all_rightview_left_ankle_coord)
            path = video_frame_path + '/' + img_rightfoot_leave
            points_pelvis_tilt = all_rightview_right_pelvis_tilt[optimal_rightfoot_leaving_ground]
            angle_right_hip = all_rightview_right_knee_hip_vertical[optimal_rightfoot_leaving_ground]
            try:
                angle_pelvis_tilt = image_processing_right(path, points_pelvis_tilt, rightview_right_image_path,
                                                           img_rightfoot_leave)
                angle_final = angle_right_hip - angle_pelvis_tilt
                print(angle_pelvis_tilt, angle_final)
            except:
                pass
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist,
                                                          all_rightview_left_knee_leg_angle,
                                                          all_rightview_left_knee_ankle_vertical_angle,
                                                          all_rightview_left_knee_hip_vertical,
                                                          all_rightview_right_hip_ankle_dist,
                                                          all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle,
                                                          all_rightview_right_knee_hip_vertical,
                                                          all_rightview_trunk_lean, optimal_rightfoot_leaving_ground,
                                                          img_rightfoot_leave,
                                                          'Phase_Dectection_rightsideview_rightextremity_toe_off_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_rightfoot_touching_ground, img_rightfoot_touch = optimal_frame_foot_touching_ground(
                all_rightview_hip_rightankle_distance_y, all_rightview_hip_rightankle_distance_x, 'positive',
                write_image_rightview_right_hip_knee_ankle,
                all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_right_knee_leg_angle,
                all_rightview_right_ankle_coord, all_rightview_left_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist,
                                                          all_rightview_left_knee_leg_angle,
                                                          all_rightview_left_knee_ankle_vertical_angle,
                                                          all_rightview_left_knee_hip_vertical,
                                                          all_rightview_right_hip_ankle_dist,
                                                          all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle,
                                                          all_rightview_right_knee_hip_vertical,
                                                          all_rightview_trunk_lean, optimal_rightfoot_touching_ground,
                                                          img_rightfoot_touch,
                                                          'Phase_Dectection_rightsideview_rightextremity_initial_contact_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground = optimal_frame_max_knee_flexion(
                all_rightview_right_hip_coord, all_rightview_right_knee_coord, all_rightview_right_ankle_coord,
                all_rightview_right_knee_leg_angle,
                all_rightview_left_knee_coord, all_rightview_left_ankle_coord,
                write_image_rightview_right_hip_knee_ankle, all_rightview_right_shoulder_coord,
                all_rightview_hip_rightankle_distance_y)
            ankle = all_rightview_right_knee_coord[optimal_rightfoot_touching_ground][0]
            knee = all_rightview_right_ankle_coord[optimal_rightfoot_touching_ground][0]
            sign = ''
            if knee > ankle:
                sign = 'Yes'
            else:
                sign = 'No'
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist,
                                                          all_rightview_left_knee_leg_angle,
                                                          all_rightview_left_knee_ankle_vertical_angle,
                                                          all_rightview_left_knee_hip_vertical,
                                                          all_rightview_right_hip_ankle_dist,
                                                          all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle,
                                                          all_rightview_right_knee_hip_vertical,
                                                          all_rightview_trunk_lean,
                                                          optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground,
                                                          'Phase_Dectection_rightsideview_rightextremity_midstance_phase',
                                                          args.image_ext, alignment, sign)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground = optimal_frame_max_knee_flexion(
                all_rightview_right_hip_coord, all_rightview_left_knee_coord, all_rightview_left_ankle_coord,
                all_rightview_left_knee_leg_angle,
                all_rightview_right_knee_coord, all_rightview_right_ankle_coord,
                write_image_rightview_left_hip_knee_ankle, all_rightview_right_shoulder_coord,
                all_rightview_hip_leftankle_distance_y)
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist,
                                                          all_rightview_left_knee_leg_angle,
                                                          all_rightview_left_knee_ankle_vertical_angle,
                                                          all_rightview_left_knee_hip_vertical,
                                                          all_rightview_right_hip_ankle_dist,
                                                          all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle,
                                                          all_rightview_right_knee_hip_vertical,
                                                          all_rightview_trunk_lean,
                                                          optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground,
                                                          'Phase_Dectection_rightsideview_leftextremity_midstance_phase',
                                                          args.image_ext, alignment)
            seq_list.append(seq_point_dict)
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
            print("back right")
            optimal_rightfoot_on_ground, img_rightfoot_ground = optimal_frame_foot_on_ground_back_front(
                all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor,
                all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor)
            x_point, y_point = feet_inner_edge(optimal_rightfoot_on_ground, all_back_hip_difference, args.distance_cam,
                                               'right')
            mean_hip = all_back_hip_coor[optimal_rightfoot_on_ground][0]
            new_ankle_points = [int(all_back_right_ankle_coor[optimal_rightfoot_on_ground][0] + x_point),
                                int(all_back_right_ankle_coor[optimal_rightfoot_on_ground][1] + y_point)]
            ankle_point_right = all_back_right_ankle_coor[optimal_rightfoot_on_ground][0]
            ankle_point_right = ankle_point_right + x_point
            img = cv2.imread(back_side_right_image_path + '' + img_rightfoot_ground)
            path_img = back_side_right_image_path + '' + img_rightfoot_ground
            img_name = img_rightfoot_ground.split('.')[0]
            cv2.imwrite(back_side_right_image_path + '' + img_rightfoot_ground, img)
            cross_over_sign_right = ''
            if ankle_point_right > mean_hip:
                cross_over_sign_right = 'No'
            else:
                cross_over_sign_right = 'Yes'
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_rightfoot_on_ground, img_rightfoot_ground,
                                                          'Phase_Dectection_backview_right_midstance_phase',
                                                          args.image_ext, args.pelvis, alignment, cross_over_sign_right)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            if args.front_back_pelvis == 'on':
                optimal_rightfoot_on_ground_max_front_back_pelvis,img_rightfoot_on_ground_max_front_back_pelvis=optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,'right',waistline_angle_list,waistline_image_list,im_list2, 'trunk')
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_rightfoot_on_ground_max_front_back_pelvis,
                                                              img_rightfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_right_max_front_back_pelvis',
                                                              args.image_ext, args.pelvis, alignment, 'right_pelvis')
                seq_list.append(seq_point_dict)

                optimal_leftfoot_on_ground_max_front_back_pelvis, img_leftfoot_on_ground_max_front_back_pelvis = optimal_frame_foot_on_ground_back_front_max_angles(
                    left_image_list, 'left', waistline_angle_list, waistline_image_list, im_list2, 'trunk')
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_leftfoot_on_ground_max_front_back_pelvis,
                                                              img_leftfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_left_max_front_back_pelvis',
                                                              args.image_ext, args.pelvis, alignment,
                                                              'left_pelvis')
                seq_list.append(seq_point_dict)
            # optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor,
            #     all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor,
            #     all_back_right_knee_leg_angle, 'non_trunk')
            optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,'right',all_back_right_knee_leg_angle,knee_angle_image_right,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_rightfoot_on_ground_max_varus,
                                                          img_rightfoot_ground_max_varus,
                                                          'Phase_Dectection_backview_right_max_knee_varus',
                                                          args.image_ext, args.pelvis, alignment, 'right_knee')
            seq_list.append(seq_point_dict)
            # optimal_rightfoot_on_ground_max_femour, img_rightfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor,
            #     all_back_left_ankle_coor, write_image_back_right_knee_hip_vertical, all_back_shoulder_coor,
            #     all_back_right_knee_hip_vertical, 'non_trunk')
            optimal_rightfoot_on_ground_max_femur, img_rightfoot_ground_max_femur = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,
                'right',all_back_right_knee_hip_vertical,hip_angle_image_right,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_rightfoot_on_ground_max_femur,
                                                          img_rightfoot_ground_max_femur,
                                                          'Phase_Dectection_backview_right_max_femur', args.image_ext,
                                                          args.pelvis, alignment, 'right_femur')
            seq_list.append(seq_point_dict)
            print("trunk right")
            # optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor,
            #     all_back_left_ankle_coor, write_image_back_trunk_lean, all_back_shoulder_coor, all_back_trunk_lean,
            #     'trunk')
            optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list, 'right',all_back_trunk_lean,back_trunk_image,im_list2,
                'trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_rightfoot_on_ground_lateral_trunk_lean,
                                                          img_rightfoot_ground_lateral_trunk_lean,
                                                          'Phase_Dectection_backview_right_lateral_trunk_lean',
                                                          args.image_ext, args.pelvis, alignment, 'trunk')
            seq_list.append(seq_point_dict)
            if args.pelvis == 'on':
                optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop, angle = optimal_frame_foot_on_ground_back_front_all_points(
                    all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor,
                    all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor,
                    all_back_hip_coor_left_right, video_frame_path)
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_rightfoot_on_ground_pelvis_drop,
                                                              img_rightfoot_ground_pelvis_drop,
                                                              'Phase_Dectection_backview_right_pelvis_angle',
                                                              args.image_ext, args.pelvis, alignment, '', angle)
                seq_list.append(seq_point_dict)
            print("back left")
            optimal_leftfoot_on_ground, img_leftfoot_ground = optimal_frame_foot_on_ground_back_front(
                all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor,
                all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor)
            x_point, y_point = feet_inner_edge(optimal_leftfoot_on_ground, all_back_hip_difference, args.distance_cam,
                                               'left')
            mean_hip = all_back_hip_coor[optimal_leftfoot_on_ground][0]
            new_points_left = [int(all_back_left_ankle_coor[optimal_leftfoot_on_ground][0] + x_point),
                               int(all_back_left_ankle_coor[optimal_leftfoot_on_ground][1] + y_point)]
            ankle_point_left = all_back_left_ankle_coor[optimal_leftfoot_on_ground][0]
            ankle_point_left = ankle_point_left + x_point
            img = cv2.imread(back_side_left_image_path + '' + img_leftfoot_ground)
            img = cv2.circle(img, (int(new_points_left[0]), int(new_points_left[1])), radius=0, color=(0, 255, 255),
                             thickness=-1)
            cv2.imwrite(back_side_left_image_path + '' + img_leftfoot_ground, img)
            cross_over_sign_left = ''
            if ankle_point_left > mean_hip:
                cross_over_sign_left = 'No'
            else:
                cross_over_sign_left = 'Yes'
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_leftfoot_on_ground, img_leftfoot_ground,
                                                          'Phase_Dectection_backview_left_midstance_phase',
                                                          args.image_ext, args.pelvis, alignment, cross_over_sign_left)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor,
            #     all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor,
            #     all_back_left_knee_leg_angle, 'non_trunk')
            optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,
                'left',all_back_left_knee_leg_angle,knee_angle_image_left,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_leftfoot_on_ground_max_varus,
                                                          img_leftfoot_ground_max_varus,
                                                          'Phase_Dectection_backview_left_max_knee_varus',
                                                          args.image_ext, args.pelvis, alignment, 'left_knee')
            seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max_femour, img_leftfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor,
            #     all_back_right_ankle_coor, write_image_back_left_knee_hip_vertical, all_back_shoulder_coor,
            #     all_back_left_knee_hip_vertical, 'non_trunk')
            optimal_leftfoot_on_ground_max_femur, img_leftfoot_ground_max_femur = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,
                'left',all_back_left_knee_hip_vertical,hip_angle_image_left,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_leftfoot_on_ground_max_femur,
                                                          img_leftfoot_ground_max_femur,
                                                          'Phase_Dectection_backview_left_max_femur', args.image_ext,
                                                          args.pelvis, alignment, 'left_femur')
            seq_list.append(seq_point_dict)
            print("trunk left")
            # optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor,
            #     all_back_right_ankle_coor, write_image_back_trunk_lean, all_back_shoulder_coor, all_back_trunk_lean,
            #     'trunk')
            optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list, 'left',all_back_trunk_lean,back_trunk_image,im_list2,
                'trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_leftfoot_on_ground_max_trunk_lane,
                                                          img_leftfoot_ground_max_trunk_lane,
                                                          'Phase_Dectection_backview_left_lateral_trunk_lean',
                                                          args.image_ext, args.pelvis, alignment, 'trunk')
            seq_list.append(seq_point_dict)
            optimal_max_chest,img_optimal_max_chest,optimal_min_chest,img_optimal_min_chest,difference=chest_keypoints(interpolated_keypoints_final)
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_min_chest,
                                                          img_optimal_min_chest,
                                                          'Phase_Dectection_backview_min_chest_keypoint',
                                                          args.image_ext, args.pelvis, alignment, '')
            seq_list.append(seq_point_dict)
            img_max_y = cv2.imread(back_side_right_image_path + str(img_optimal_max_chest))
            model_output = model(img_max_y)
            boxes1 = model_output['instances'].pred_boxes
            boxes1 = boxes1.tensor.cpu().numpy()
            boxes2 = boxes1[0]
            boxes2 = boxes2.astype(int)
            crop_img(img_max_y, back_side_right_image_path, 'Phase_Dectection_backview_max_chest_keypoints_',
                     img_optimal_max_chest, boxes2)
            img_min_y = cv2.imread(back_side_right_image_path + str(img_optimal_min_chest))
            model_output = model(img_min_y)
            boxes3 = model_output['instances'].pred_boxes
            boxes3 = boxes3.tensor.cpu().numpy()
            boxes4 = boxes3[0]
            boxes4 = boxes4.astype(int)
            crop_img(img_min_y, back_side_right_image_path, 'Phase_Dectection_backview_min_chest_keypoints_',
                     img_optimal_min_chest, boxes4)
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_max_chest,
                                                          img_optimal_max_chest,
                                                          'Phase_Dectection_backview_max_chest_keypoint',
                                                          args.image_ext, args.pelvis, alignment, '')
            seq_list.append(seq_point_dict)
            if args.pelvis == 'on':
                optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop, angle = optimal_frame_foot_on_ground_back_front_all_points(
                    all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor,
                    all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor,
                    all_back_hip_coor_left_right, video_frame_path)
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_leftfoot_on_ground_pelvis_drop,
                                                              img_leftfoot_ground_pelvis_drop,
                                                              'Phase_Dectection_backview_left_pelvis_angle',
                                                              args.image_ext, args.pelvis, alignment, '', angle)
                seq_list.append(seq_point_dict)
            distance_chest_keypoints = []
            yaxis_list = []
            if len(all_back_chest_yaxis_values) != 0:
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
            # if distance > 0:
            #     seq_point_dict = back_chest('Phase_Dectection_backview_', distance_chest_keypoints)
            #     seq_list.append(seq_point_dict)
            seq_point_dict = back_shoulder('Phase_Dectection_backview_',optimal_max_chest,optimal_min_chest,difference)
            seq_list.append(seq_point_dict)
            save_optimal_frame(back_side_right_image_path, img_rightfoot_ground,
                               'Phase_Dectection_backview_right_midstance_phase_')
            save_optimal_frame(back_side_right_image_path, str(first_right_image)+"."+str(args.image_ext),
                               'Phase_Dectection_backview_right_cross_over_')
            save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_max_varus,
                               'Phase_Dectection_backview_right_max_knee_varus_')
            save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_max_femur,
                               'Phase_Dectection_backview_right_max_femur_')
            save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_lateral_trunk_lean,
                               'Phase_Dectection_backview_right_lateral_trunk_lean_')
            # save_optimal_frame(back_side_right_image_path, img_optimal_min_chest,
            #                    'Phase_Dectection_backview_min_chest_keypoints_')
            # save_optimal_frame(back_side_right_image_path, img_optimal_max_chest,
            #                    'Phase_Dectection_backview_max_chest_keypoints_')
            # save_optimal_frame(back_side_right_image_path, distance_chest_keypoints[1],
            #                    'Phase_Dectection_backview_chest_distance_min_')
            # save_optimal_frame(back_side_right_image_path, distance_chest_keypoints[2],
            #                    'Phase_Dectection_backview_chest_distance_max_')
            if args.pelvis == 'on':
                save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_pelvis_drop,
                                   'Phase_Dectection_backview_right_pelvis_angle_')
            if args.front_back_pelvis == 'on':
                save_optimal_frame(back_side_right_image_path,img_rightfoot_on_ground_max_front_back_pelvis,'Phase_Dectection_backview_right_max_front_back_pelvis_')
            save_optimal_frame(back_side_left_image_path, img_leftfoot_ground,
                               'Phase_Dectection_backview_left_midstance_phase_')
            save_optimal_frame(back_side_left_image_path, str(first_left_image)+"."+str(args.image_ext),
                               'Phase_Dectection_backview_left_cross_over_')
            save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_max_varus,
                               'Phase_Dectection_backview_left_max_knee_varus_')
            save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_max_femur,
                               'Phase_Dectection_backview_left_max_femur_')
            save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_max_trunk_lane,
                               'Phase_Dectection_backview_left_lateral_trunk_lean_')
            if args.pelvis == 'on':
                save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_pelvis_drop,
                                   'Phase_Dectection_backview_left_pelvis_angle_')
            if args.front_back_pelvis == 'on':
                save_optimal_frame(back_side_left_image_path,img_leftfoot_on_ground_max_front_back_pelvis,'Phase_Dectection_backview_left_max_front_back_pelvis_')
        if alignment == "front_side":
            optimal_rightfoot_on_ground, img_rightfoot_ground = optimal_frame_foot_on_ground_back_front(
                all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor,
                all_front_left_ankle_coor, write_image_front_right_hip_knee_ankle, all_front_shoulder_coor)
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle,
                                                          all_front_right_knee_hip_vertical,
                                                          all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean,
                                                          optimal_rightfoot_on_ground, img_rightfoot_ground,
                                                          'Phase_Dectection_frontview_right_midstance_phase',
                                                          args.image_ext, args.pelvis, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)

            if args.front_back_pelvis == 'on':
                optimal_rightfoot_on_ground_max_front_back_pelvis,img_rightfoot_on_ground_max_front_back_pelvis=optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,'right',waistline_angle_list,waistline_image_list,im_list2, 'trunk')
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_rightfoot_on_ground_max_front_back_pelvis,
                                                              img_rightfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_right_max_front_back_pelvis',
                                                              args.image_ext, args.pelvis, alignment, 'right_pelvis')
                seq_list.append(seq_point_dict)

                optimal_leftfoot_on_ground_max_front_back_pelvis, img_leftfoot_on_ground_max_front_back_pelvis = optimal_frame_foot_on_ground_back_front_max_angles(
                    left_image_list, 'left', waistline_angle_list, waistline_image_list, im_list2, 'trunk')
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_leftfoot_on_ground_max_front_back_pelvis,
                                                              img_leftfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_left_max_front_back_pelvis',
                                                              args.image_ext, args.pelvis, alignment,
                                                              'left_pelvis')
                seq_list.append(seq_point_dict)

            # optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor,
            #     all_front_left_ankle_coor, write_image_front_right_hip_knee_ankle, all_front_shoulder_coor,
            #     all_front_right_knee_leg_angle, 'non_trunk')
            optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,
                'right',all_front_right_knee_leg_angle,front_right_knee_image,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle,
                                                          all_front_right_knee_hip_vertical,
                                                          all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean,
                                                          optimal_rightfoot_on_ground_max_varus,
                                                          img_rightfoot_ground_max_varus,
                                                          'Phase_Dectection_frontview_right_max_knee_varus',
                                                          args.image_ext, args.pelvis, alignment, 'right_knee')
            seq_list.append(seq_point_dict)
            # optimal_rightfoot_on_ground_max_femour, img_rightfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor,
            #     all_front_left_ankle_coor, write_image_front_right_knee_hip_vertical, all_front_shoulder_coor,
            #     all_front_right_knee_hip_vertical, 'non_trunk')
            optimal_rightfoot_on_ground_max_femur, img_rightfoot_ground_max_femur = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,
                'right',all_front_right_knee_hip_vertical,front_hip_knee_right,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle,
                                                          all_front_right_knee_hip_vertical,
                                                          all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean,
                                                          optimal_rightfoot_on_ground_max_femur,
                                                          img_rightfoot_ground_max_femur,
                                                          'Phase_Dectection_frontview_right_max_femur', args.image_ext,
                                                          args.pelvis, alignment, 'right_femur')
            seq_list.append(seq_point_dict)
            # optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor,
            #     all_front_left_ankle_coor, write_image_front_trunk_lean, all_front_shoulder_coor, all_front_trunk_lean,
            #     "trunk")
            optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list, 'right',all_front_trunk_lean,front_trunck_image,im_list2,
                "trunk")
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle,
                                                          all_front_right_knee_hip_vertical,
                                                          all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean,
                                                          optimal_rightfoot_on_ground_lateral_trunk_lean,
                                                          img_rightfoot_ground_lateral_trunk_lean,
                                                          'Phase_Dectection_frontview_right_lateral_trunk_lean',
                                                          args.image_ext, args.pelvis, alignment, 'trunk')
            seq_list.append(seq_point_dict)
            if args.pelvis == 'on':
                optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop, angle = optimal_frame_foot_on_ground_back_front_all_points(
                    all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor,
                    all_front_left_ankle_coor, write_image_front_left_hip_right_hip, all_front_shoulder_coor,
                    all_front_hip_coor_left_right, video_frame_path)
                seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle,
                                                              all_front_left_knee_leg_angle,
                                                              all_front_right_knee_hip_vertical,
                                                              all_front_left_knee_hip_vertical,
                                                              all_front_left_hip_right_hip, all_front_trunk_lean,
                                                              optimal_rightfoot_on_ground_pelvis_drop,
                                                              img_rightfoot_ground_pelvis_drop,
                                                              'Phase_Dectection_frontview_right_pelvis_angle', args.image_ext,
                                                              args.pelvis, alignment, angle)
                seq_list.append(seq_point_dict)
            optimal_leftfoot_on_ground, img_leftfoot_ground = optimal_frame_foot_on_ground_back_front(
                all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor,
                all_front_right_ankle_coor, write_image_front_left_hip_knee_ankle, all_front_shoulder_coor)
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle,
                                                          all_front_right_knee_hip_vertical,
                                                          all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean,
                                                          optimal_leftfoot_on_ground, img_leftfoot_ground,
                                                          'Phase_Dectection_frontview_left_midstance_phase',
                                                          args.image_ext, args.pelvis, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor,
            #     all_front_right_ankle_coor, write_image_front_left_hip_knee_ankle, all_front_shoulder_coor,
            #     all_front_left_knee_leg_angle, 'non_trunk')
            optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,
                'left',all_front_left_knee_leg_angle,front_left_knee_image,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle,
                                                          all_front_right_knee_hip_vertical,
                                                          all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean,
                                                          optimal_leftfoot_on_ground_max_varus,
                                                          img_leftfoot_ground_max_varus,
                                                          'Phase_Dectection_frontview_left_max_knee_varus',
                                                          args.image_ext, args.pelvis, alignment, 'left_knee')
            seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max__femour, img_leftfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor,
            #     all_front_right_ankle_coor, write_image_front_left_knee_hip_vertical, all_front_shoulder_coor,
            #     all_front_left_knee_hip_vertical, 'non_trunk')
            optimal_leftfoot_on_ground_max__femur, img_leftfoot_ground_max_femur = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,
                'left',all_front_left_knee_hip_vertical,front_hip_knee_left,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle,
                                                          all_front_right_knee_hip_vertical,
                                                          all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean,
                                                          optimal_leftfoot_on_ground_max__femur,
                                                          img_leftfoot_ground_max_femur,
                                                          'Phase_Dectection_frontview_left_max_femur', args.image_ext,
                                                          args.pelvis, alignment, 'left_femur')
            seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor,
            #     all_front_right_ankle_coor, write_image_front_trunk_lean, all_front_shoulder_coor,
            #     all_front_left_knee_hip_vertical, 'trunk')
            optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,
                'left',all_front_trunk_lean,front_trunck_image,im_list2, 'trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle,
                                                          all_front_right_knee_hip_vertical,
                                                          all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean,
                                                          optimal_leftfoot_on_ground_max_trunk_lane,
                                                          img_leftfoot_ground_max_trunk_lane,
                                                          'Phase_Dectection_frontview_left_lateral_trunk_lean',
                                                          args.image_ext, args.pelvis, alignment, 'trunk')
            seq_list.append(seq_point_dict)
            if args.pelvis == 'on':
                optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop, angle = optimal_frame_foot_on_ground_back_front_all_points(
                    all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor,
                    all_front_right_ankle_coor, write_image_front_left_hip_right_hip, all_front_shoulder_coor,
                    all_front_hip_coor_left_right, video_frame_path)
                seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle,
                                                              all_front_left_knee_leg_angle,
                                                              all_front_right_knee_hip_vertical,
                                                              all_front_left_knee_hip_vertical,
                                                              all_front_left_hip_right_hip, all_front_trunk_lean,
                                                              optimal_leftfoot_on_ground_max__femur,
                                                              img_leftfoot_ground_max_femur,
                                                              'Phase_Dectection_frontview_left_pelvis_angle',
                                                              args.image_ext, args.pelvis, alignment, angle)
                seq_list.append(seq_point_dict)
            save_optimal_frame(front_right_image_path, img_rightfoot_ground,
                               'Phase_Dectection_frontview_right_midstance_phase_')
            save_optimal_frame(front_right_image_path, img_rightfoot_ground_max_varus,
                               'Phase_Dectection_frontview_right_max_knee_varus_')
            save_optimal_frame(front_right_image_path, img_rightfoot_ground_max_femur,
                               'Phase_Dectection_frontview_right_max_femur_')
            save_optimal_frame(front_right_image_path, img_rightfoot_ground_lateral_trunk_lean,
                               'Phase_Dectection_frontview_right_lateral_trunk_lean_')
            if args.pelvis == 'on':
                save_optimal_frame(front_right_image_path, img_rightfoot_ground_pelvis_drop,
                                   'Phase_Dectection_frontview_pelvis_angle_')
            if args.front_back_pelvis == 'on':
                save_optimal_frame(front_right_image_path, img_rightfoot_on_ground_max_front_back_pelvis,
                                   'Phase_Dectection_backview_right_max_front_back_pelvis_')
            save_optimal_frame(front_left_image_path, img_leftfoot_ground,
                               'Phase_Dectection_frontview_left_midstance_phase_')
            save_optimal_frame(front_left_image_path, img_leftfoot_ground_max_varus,
                               'Phase_Dectection_frontview_left_max_knee_varus_')
            save_optimal_frame(front_left_image_path, img_leftfoot_ground_max_femur,
                               'Phase_Dectection_frontview_left_max_femur_')
            save_optimal_frame(front_left_image_path, img_leftfoot_ground_max_trunk_lane,
                               'Phase_Dectection_frontview_left_lateral_trunk_lean_')
            if args.pelvis == 'on':
                save_optimal_frame(front_left_image_path, img_leftfoot_ground_pelvis_drop,
                                   'Phase_Dectection_frontview_left_pelvis_angle_')
            if args.front_back_pelvis == 'on':
                save_optimal_frame(front_left_image_path, img_leftfoot_on_ground_max_front_back_pelvis,
                                   'Phase_Dectection_backview_left_max_front_back_pelvis_')
        json_list['sequence'] = seq_list
        optimal_frame_dict['sequence'] = optimal_frame_seq_list
        with open(json_folder, 'w') as f:
            json.dump(json_list, f, ensure_ascii=False, indent=4)
        with open(optimal_json_name, 'w') as f:
            json.dump(optimal_frame_dict, f, ensure_ascii=False, indent=4)
    else:
        if alignment == "left_side":  # or alignment=="right_side":
            if args.side_pelvis == 'on':

                optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(
                    all_leftview_hip_leftankle_distance_y, all_leftview_hip_leftankle_distance_x, 'positive',
                    write_image_leftview_right_hip_knee_ankle,
                    all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_left_knee_leg_angle,
                    all_leftview_left_ankle_coord, all_leftview_right_ankle_coord)
                waistline_image_line = cv2.imread(str(out_put_image_dir) + 'left/images/' + str(img_leftfoot_leave))
                orignal_image_left_foot = cv2.imread(str(im_list2[optimal_leftfoot_leaving_ground]))
                _, waistline_angle = waistline.draw_waistline_side(orignal_image_left_foot,waistline_image_line,
                                                                   pts=interpolated_keypoints_final[
                                                                       optimal_leftfoot_leaving_ground],
                                                                   clt=waistline_color_cluster, angle_font_params=(
                    font, fontScale, fontColor, lineType))
                # plt.title(img_leftfoot_leave)
                # plt.imshow(waistline_image_line)
                # plt.show()
                cv2.imwrite(str(out_put_image_dir) + 'left/images/' + str(img_leftfoot_leave), waistline_image_line)

            optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(all_leftview_hip_leftankle_distance_y, all_leftview_hip_leftankle_distance_x, 'positive', write_image_leftview_right_hip_knee_ankle,
                                                                                                    all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_left_knee_leg_angle, all_leftview_left_ankle_coord, all_leftview_right_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_leftfoot_leaving_ground, img_leftfoot_leave, 'Phase_Dectection_leftsideview_leftextremity_toe_off_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_leftfoot_touching_ground, img_leftfoot_touch = optimal_frame_foot_touching_ground(all_leftview_hip_leftankle_distance_y, all_leftview_hip_leftankle_distance_x, 'negative', write_image_leftview_right_hip_knee_ankle,
                                                                                                      all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_left_knee_leg_angle, all_leftview_left_ankle_coord, all_leftview_right_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_leftfoot_touching_ground, img_leftfoot_touch, 'Phase_Dectection_leftsideview_leftextremity_initial_contact_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            if args.side_pelvis == 'on':
                optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(
                    all_leftview_hip_rightankle_distance_y, all_leftview_hip_rightankle_distance_x, 'positive',
                    write_image_leftview_right_hip_knee_ankle,
                    all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_right_knee_leg_angle,
                    all_leftview_right_ankle_coord, all_leftview_left_ankle_coord)
                waistline_image_line = cv2.imread(str(out_put_image_dir) + 'right/images/' + str(img_rightfoot_leave))
                orignal_image_right_foot = cv2.imread(str(im_list2[optimal_rightfoot_leaving_ground]))
                _, waistline_angle = waistline.draw_waistline_side(orignal_image_right_foot,waistline_image_line,
                                                                   pts=interpolated_keypoints_final[
                                                                       optimal_rightfoot_leaving_ground],
                                                                   clt=waistline_color_cluster, angle_font_params=(
                    font, fontScale, fontColor, lineType))
                # plt.title(img_rightfoot_leave)
                # plt.imshow(waistline_image_line)
                # plt.show()
                cv2.imwrite(str(out_put_image_dir) + 'right/images/' + str(img_rightfoot_leave), waistline_image_line)
                # print("image 36 updated",img_rightfoot_leave)
            optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(all_leftview_hip_rightankle_distance_y, all_leftview_hip_rightankle_distance_x, 'positive', write_image_leftview_right_hip_knee_ankle,
                                                                                                      all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_right_knee_leg_angle, all_leftview_right_ankle_coord, all_leftview_left_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_rightfoot_leaving_ground, img_rightfoot_leave, 'Phase_Dectection_leftsideview_rightextremity_toe_off_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_rightfoot_touching_ground, img_rightfoot_touch = optimal_frame_foot_touching_ground(all_leftview_hip_rightankle_distance_y, all_leftview_hip_rightankle_distance_x, 'negative',
                                                                                                        write_image_leftview_right_hip_knee_ankle, all_leftview_left_hip_coord, all_leftview_left_shoulder_coord, all_leftview_right_knee_leg_angle, all_leftview_right_ankle_coord, all_leftview_left_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_rightfoot_touching_ground, img_rightfoot_touch, 'Phase_Dectection_leftsideview_rightextremity_initial_contact_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground = optimal_frame_max_knee_flexion(all_leftview_left_hip_coord, all_leftview_right_knee_coord, all_leftview_right_ankle_coord, all_leftview_right_knee_leg_angle,
                                                                                                         all_leftview_left_knee_coord, all_leftview_left_ankle_coord, write_image_leftview_right_hip_knee_ankle, all_leftview_left_shoulder_coord, all_leftview_hip_rightankle_distance_y)
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_knee_flexion_rightfoot_ground, img_rightfoot_ground, 'Phase_Dectection_leftsideview_rightextremity_midstance_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground = optimal_frame_max_knee_flexion(all_leftview_left_hip_coord, all_leftview_left_knee_coord, all_leftview_left_ankle_coord, all_leftview_left_knee_leg_angle,
                                                                                                       all_leftview_right_knee_coord, all_leftview_right_ankle_coord, write_image_leftview_right_hip_knee_ankle, all_leftview_left_shoulder_coord, all_leftview_hip_leftankle_distance_y)
            knee = all_leftview_left_knee_coord[optimal_knee_flexion_leftfoot_ground][0]
            ankle = all_leftview_left_ankle_coord[optimal_knee_flexion_leftfoot_ground][0]
            if knee > ankle:
                sign= 'Yes'
            else:
                sign = 'No'
            seq_point_dict = left_right_optimal_keypoints(all_leftview_left_hip_ankle_dist, all_leftview_left_knee_leg_angle, all_leftview_left_knee_ankle_vertical_angle, all_leftview_left_knee_hip_vertical, all_leftview_right_hip_ankle_dist, all_leftview_right_knee_leg_angle,
                                                          all_leftview_right_knee_ankle_vertical_angle, all_leftview_right_knee_hip_vertical, all_leftview_trunk_lean, optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground, 'Phase_Dectection_leftsideview_leftextremity_midstance_phase', args.image_ext, alignment,sign)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
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

            if args.side_pelvis == 'on':
                optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(
                    all_rightview_hip_leftankle_distance_y, all_rightview_hip_leftankle_distance_x, 'negative',
                    write_image_rightview_right_hip_knee_ankle,
                    all_rightview_right_hip_coord, all_rightview_right_shoulder_coord,
                    all_rightview_left_knee_leg_angle, all_rightview_left_ankle_coord, all_rightview_right_ankle_coord)
                waistline_image_line = cv2.imread(str(out_put_image_dir) + 'left/images/' + str(img_leftfoot_leave))
                orignal_image_left_foot = cv2.imread(str(im_list2[optimal_leftfoot_leaving_ground]))
                _, waistline_angle = waistline.draw_waistline_side(orignal_image_left_foot,waistline_image_line,
                                                                   pts=interpolated_keypoints_final[
                                                                       optimal_leftfoot_leaving_ground],
                                                                   clt=waistline_color_cluster, angle_font_params=(
                    font, fontScale, fontColor, lineType))
                cv2.imwrite(str(out_put_image_dir) + 'left/images/' + str(img_leftfoot_leave), waistline_image_line)

            optimal_leftfoot_leaving_ground, img_leftfoot_leave = optimal_frame_foot_leaving_ground(all_rightview_hip_leftankle_distance_y, all_rightview_hip_leftankle_distance_x, 'negative', write_image_rightview_right_hip_knee_ankle,
                                                                                                    all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_left_knee_leg_angle, all_rightview_left_ankle_coord, all_rightview_right_ankle_coord)

            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_leftfoot_leaving_ground, img_leftfoot_leave, 'Phase_Dectection_rightsideview_leftextremity_toe_off_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_leftfoot_touching_ground, img_leftfoot_touch = optimal_frame_foot_touching_ground(all_rightview_hip_leftankle_distance_y, all_rightview_hip_leftankle_distance_x, 'positive', write_image_rightview_right_hip_knee_ankle,
                                                                                                      all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_left_knee_leg_angle, all_rightview_left_ankle_coord, all_rightview_right_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_leftfoot_touching_ground, img_leftfoot_touch, 'Phase_Dectection_rightsideview_leftextremity_initial_contact_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            if args.side_pelvis == 'on':
                optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(
                    all_rightview_hip_rightankle_distance_y, all_rightview_hip_rightankle_distance_x, 'negative',
                    write_image_rightview_right_hip_knee_ankle,
                    all_rightview_right_hip_coord, all_rightview_right_shoulder_coord,
                    all_rightview_right_knee_leg_angle, all_rightview_right_ankle_coord, all_rightview_left_ankle_coord)
                orignal_image_right_foot = cv2.imread(str(im_list2[optimal_rightfoot_leaving_ground]))
                waistline_image_line = cv2.imread(str(out_put_image_dir) + 'right/images/' + str(img_rightfoot_leave))
                _, waistline_angle = waistline.draw_waistline_side(orignal_image_right_foot,waistline_image_line,
                                                                   pts=interpolated_keypoints_final[
                                                                       optimal_rightfoot_leaving_ground],
                                                                   clt=waistline_color_cluster, angle_font_params=(
                    font, fontScale, fontColor, lineType))
                cv2.imwrite(str(out_put_image_dir) + 'right/images/' + str(img_rightfoot_leave), waistline_image_line)
                # plt.imshow(waistline_image_line)
                # plt.show()
                # plt.imshow(orignal_image_right_foot)
                # plt.show()


            optimal_rightfoot_leaving_ground, img_rightfoot_leave = optimal_frame_foot_leaving_ground(all_rightview_hip_rightankle_distance_y, all_rightview_hip_rightankle_distance_x, 'negative', write_image_rightview_right_hip_knee_ankle,
                                                                                                      all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_right_knee_leg_angle, all_rightview_right_ankle_coord, all_rightview_left_ankle_coord)
            # test_right_foot=cv2.imread(str(out_put_image_dir) + 'right/images/' + str(img_rightfoot_leave))
            # plt.imshow(test_right_foot)
            # plt.show()

            path = video_frame_path+'/'+img_rightfoot_leave
            points_pelvis_tilt = all_rightview_right_pelvis_tilt[optimal_rightfoot_leaving_ground]
            angle_right_hip  = all_rightview_right_knee_hip_vertical[optimal_rightfoot_leaving_ground]
            try:
                angle_pelvis_tilt = image_processing_right(path,points_pelvis_tilt,rightview_right_image_path,img_rightfoot_leave)
                angle_final = angle_right_hip - angle_pelvis_tilt
                print(angle_pelvis_tilt,angle_final)
            except:
                pass
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_rightfoot_leaving_ground, img_rightfoot_leave, 'Phase_Dectection_rightsideview_rightextremity_toe_off_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_rightfoot_touching_ground, img_rightfoot_touch = optimal_frame_foot_touching_ground(all_rightview_hip_rightankle_distance_y, all_rightview_hip_rightankle_distance_x, 'positive', write_image_rightview_right_hip_knee_ankle,
                                                                                                        all_rightview_right_hip_coord, all_rightview_right_shoulder_coord, all_rightview_right_knee_leg_angle, all_rightview_right_ankle_coord, all_rightview_left_ankle_coord)
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_rightfoot_touching_ground, img_rightfoot_touch, 'Phase_Dectection_rightsideview_rightextremity_initial_contact_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
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
            optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground = optimal_frame_max_knee_flexion(all_rightview_right_hip_coord, all_rightview_left_knee_coord, all_rightview_left_ankle_coord, all_rightview_left_knee_leg_angle,
                                                                                                       all_rightview_right_knee_coord, all_rightview_right_ankle_coord, write_image_rightview_left_hip_knee_ankle, all_rightview_right_shoulder_coord, all_rightview_hip_leftankle_distance_y)
            seq_point_dict = left_right_optimal_keypoints(all_rightview_left_hip_ankle_dist, all_rightview_left_knee_leg_angle, all_rightview_left_knee_ankle_vertical_angle, all_rightview_left_knee_hip_vertical, all_rightview_right_hip_ankle_dist, all_rightview_right_knee_leg_angle,
                                                          all_rightview_right_knee_ankle_vertical_angle, all_rightview_right_knee_hip_vertical, all_rightview_trunk_lean, optimal_knee_flexion_leftfoot_ground, img_leftfoot_ground, 'Phase_Dectection_rightsideview_leftextremity_midstance_phase', args.image_ext, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
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
            print("back right")
            optimal_rightfoot_on_ground, img_rightfoot_ground = optimal_frame_foot_on_ground_back_front(
                all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor)
            x_point, y_point = feet_inner_edge(optimal_rightfoot_on_ground,all_back_hip_difference,args.distance_cam,'right')
            mean_hip = all_back_hip_coor[optimal_rightfoot_on_ground][0]
            new_ankle_points = [int(all_back_right_ankle_coor[optimal_rightfoot_on_ground][0] + x_point),int(all_back_right_ankle_coor[optimal_rightfoot_on_ground][1] + y_point)]
            ankle_point_right = all_back_right_ankle_coor[optimal_rightfoot_on_ground][0]
            ankle_point_right = ankle_point_right + x_point
            img = cv2.imread(back_side_right_image_path+''+img_rightfoot_ground)
            path_img = back_side_right_image_path+''+img_rightfoot_ground
            img_name = img_rightfoot_ground.split('.')[0]
            cv2.imwrite(back_side_right_image_path+''+img_rightfoot_ground,img)
            cross_over_sign_right = ''
            if ankle_point_right > mean_hip:
                cross_over_sign_right = 'No'
            else:
                cross_over_sign_right = 'Yes'
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground, img_rightfoot_ground, 'Phase_Dectection_backview_right_midstance_phase', args.image_ext, args.pelvis, alignment,cross_over_sign_right)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor, all_back_right_knee_leg_angle,'non_trunk')
            optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,
                'right',all_back_right_knee_leg_angle,knee_angle_image_right,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus, 'Phase_Dectection_backview_right_max_knee_varus', args.image_ext, args.pelvis, alignment,'right_knee')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_rightfoot_on_ground_max_femour, img_rightfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_knee_hip_vertical, all_back_shoulder_coor, all_back_right_knee_hip_vertical,'non_trunk')
            optimal_rightfoot_on_ground_max_femur, img_rightfoot_ground_max_femur = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,
                'right',all_back_right_knee_hip_vertical,hip_angle_image_right,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground_max_femur, img_rightfoot_ground_max_femur, 'Phase_Dectection_backview_right_max_femur', args.image_ext, args.pelvis, alignment,'right_femur')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            print("trunk right")
            # optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_trunk_lean, all_back_shoulder_coor, all_back_trunk_lean, 'trunk')
            optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list, 'right',all_back_trunk_lean,back_trunk_image,im_list2,
                'trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean, 'Phase_Dectection_backview_right_lateral_trunk_lean', args.image_ext, args.pelvis, alignment,'trunk')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            if args.front_back_pelvis == 'on':
                optimal_rightfoot_on_ground_max_front_back_pelvis,img_rightfoot_on_ground_max_front_back_pelvis=optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,'right',waistline_angle_list,waistline_image_list,im_list2, 'trunk')
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_rightfoot_on_ground_max_front_back_pelvis,
                                                              img_rightfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_right_max_front_back_pelvis',
                                                              args.image_ext, args.pelvis, alignment, 'right_pelvis')
                seq_list.append(seq_point_dict)
                optimal_frame_seq_list.append(seq_point_dict)

                optimal_leftfoot_on_ground_max_front_back_pelvis, img_leftfoot_on_ground_max_front_back_pelvis = optimal_frame_foot_on_ground_back_front_max_angles(
                    left_image_list, 'left', waistline_angle_list, waistline_image_list, im_list2, 'trunk')
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_leftfoot_on_ground_max_front_back_pelvis,
                                                              img_leftfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_left_max_front_back_pelvis',
                                                              args.image_ext, args.pelvis, alignment,
                                                              'left_pelvis')
                seq_list.append(seq_point_dict)
                optimal_frame_seq_list.append(seq_point_dict)
            if args.pelvis == 'on':
                optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(all_back_hip_coor, all_back_right_knee_coor, all_back_right_ankle_coor, all_back_left_knee_coor, all_back_left_ankle_coor, write_image_back_right_hip_knee_ankle, all_back_shoulder_coor, all_back_hip_coor_left_right,video_frame_path)
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                            all_back_left_hip_right_hip, all_back_trunk_lean, optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop, 'Phase_Dectection_backview_right_pelvis_angle', args.image_ext, args.pelvis, alignment,'',angle)
                seq_list.append(seq_point_dict)
                optimal_frame_seq_list.append(seq_point_dict)
            print("back left")
            optimal_leftfoot_on_ground, img_leftfoot_ground = optimal_frame_foot_on_ground_back_front(
                all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor)
            x_point, y_point = feet_inner_edge(optimal_leftfoot_on_ground,all_back_hip_difference,args.distance_cam,'left')
            mean_hip = all_back_hip_coor[optimal_leftfoot_on_ground][0]
            new_points_left = [int(all_back_left_ankle_coor[optimal_leftfoot_on_ground][0]+x_point),int(all_back_left_ankle_coor[optimal_leftfoot_on_ground][1]+y_point)]
            ankle_point_left = all_back_left_ankle_coor[optimal_leftfoot_on_ground][0]
            ankle_point_left = ankle_point_left + x_point
            img = cv2.imread(back_side_left_image_path+''+img_leftfoot_ground)
            img = cv2.circle(img, (int(new_points_left[0]),int(new_points_left[1])), radius=0, color=(0, 255, 255), thickness=-1)
            cv2.imwrite(back_side_left_image_path+''+img_leftfoot_ground,img)
            cross_over_sign_left = ''
            if ankle_point_left > mean_hip:
                cross_over_sign_left = 'No'
            else:
                cross_over_sign_left = 'Yes'
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground, img_leftfoot_ground, 'Phase_Dectection_backview_left_midstance_phase', args.image_ext, args.pelvis, alignment,cross_over_sign_left)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor, all_back_left_knee_leg_angle,'non_trunk')
            optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,
                'left',all_back_left_knee_leg_angle,knee_angle_image_left,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus, 'Phase_Dectection_backview_left_max_knee_varus', args.image_ext, args.pelvis, alignment,'left_knee')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max_femour, img_leftfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_knee_hip_vertical, all_back_shoulder_coor, all_back_left_knee_hip_vertical,'non_trunk')
            optimal_leftfoot_on_ground_max_femur, img_leftfoot_ground_max_femur = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,
                'left',all_back_left_knee_hip_vertical,hip_angle_image_left,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground_max_femur, img_leftfoot_ground_max_femur, 'Phase_Dectection_backview_left_max_femur', args.image_ext, args.pelvis, alignment,'left_femur')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            print("trunk left")
            # optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_trunk_lean, all_back_shoulder_coor, all_back_trunk_lean,'trunk')
            optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list, 'left',all_back_trunk_lean,back_trunk_image,im_list2,
                'trunk')
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane, 'Phase_Dectection_backview_left_lateral_trunk_lean', args.image_ext, args.pelvis, alignment,'trunk')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            optimal_max_chest, img_optimal_max_chest, optimal_min_chest, img_optimal_min_chest, difference = chest_keypoints(
                interpolated_keypoints_final)
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_min_chest,
                                                          img_optimal_min_chest,
                                                          'Phase_Dectection_backview_min_chest_keypoint',
                                                          args.image_ext, args.pelvis, alignment, '')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle,
                                                          all_back_right_knee_hip_vertical,
                                                          all_back_left_knee_hip_vertical,
                                                          all_back_left_hip_right_hip, all_back_trunk_lean,
                                                          optimal_max_chest,
                                                          img_optimal_max_chest,
                                                          'Phase_Dectection_backview_max_chest_keypoint',
                                                          args.image_ext, args.pelvis, alignment, '')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            img_max_y=cv2.imread(back_side_right_image_path+str(img_optimal_max_chest))
            model_output = model(img_max_y)
            boxes1 = model_output['instances'].pred_boxes
            boxes1 = boxes1.tensor.cpu().numpy()
            boxes2 = boxes1[0]
            boxes2 = boxes2.astype(int)
            crop_img(img_max_y,back_side_right_image_path,'Phase_Dectection_backview_max_chest_keypoints_',img_optimal_max_chest,boxes2)
            img_min_y=cv2.imread(back_side_right_image_path+str(img_optimal_min_chest))
            model_output = model(img_min_y)
            boxes3 = model_output['instances'].pred_boxes
            boxes3 = boxes3.tensor.cpu().numpy()
            boxes4 =boxes3[0]
            boxes4 = boxes4.astype(int)
            crop_img(img_min_y,back_side_right_image_path,'Phase_Dectection_backview_min_chest_keypoints_',img_optimal_min_chest,boxes4)
            if args.pelvis == 'on':
                optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(
                    all_back_hip_coor, all_back_left_knee_coor, all_back_left_ankle_coor, all_back_right_knee_coor, all_back_right_ankle_coor, write_image_back_left_hip_knee_ankle, all_back_shoulder_coor, all_back_hip_coor_left_right,video_frame_path)
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle, all_back_left_knee_leg_angle, all_back_right_knee_hip_vertical, all_back_left_knee_hip_vertical,
                                                            all_back_left_hip_right_hip, all_back_trunk_lean, optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop, 'Phase_Dectection_backview_left_pelvis_angle', args.image_ext, args.pelvis, alignment,'',angle)
                seq_list.append(seq_point_dict)
                optimal_frame_seq_list.append(seq_point_dict)
            distance_chest_keypoints = []
            yaxis_list = []
            if len(all_back_chest_yaxis_values) != 0:
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
            # if distance >0:
            #     seq_point_dict = back_chest('Phase_Dectection_backview_',distance_chest_keypoints)
            #     seq_list.append(seq_point_dict)
            #     optimal_frame_seq_list.append(seq_point_dict)
            seq_point_dict = back_shoulder('Phase_Dectection_backview_', optimal_max_chest, optimal_min_chest,difference)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            save_optimal_frame(back_side_right_image_path, img_rightfoot_ground,
                               'Phase_Dectection_backview_right_midstance_phase_')
            save_optimal_frame(back_side_right_image_path, str(first_right_image)+"."+str(args.image_ext),
                               'Phase_Dectection_backview_right_cross_over_')
            save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_max_varus,
                               'Phase_Dectection_backview_right_max_knee_varus_')
            save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_max_femur,
                               'Phase_Dectection_backview_right_max_femur_')
            save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_lateral_trunk_lean,
                               'Phase_Dectection_backview_right_lateral_trunk_lean_')
            # save_optimal_frame(back_side_right_image_path, img_optimal_min_chest,
            #                    'Phase_Dectection_backview_min_chest_keypoints_')
            # save_optimal_frame(back_side_right_image_path, img_optimal_max_chest,
            #                    'Phase_Dectection_backview_max_chest_keypoints_')
            # save_optimal_frame(back_side_right_image_path, distance_chest_keypoints[1],
            #                    'Phase_Dectection_backview_chest_distance_min_')
            # save_optimal_frame(back_side_right_image_path, distance_chest_keypoints[2],
            #                    'Phase_Dectection_backview_chest_distance_max_')
            if args.pelvis == 'on':
                save_optimal_frame(back_side_right_image_path, img_rightfoot_ground_pelvis_drop,
                                'Phase_Dectection_backview_right_pelvis_angle_')
            if args.front_back_pelvis == 'on':
                save_optimal_frame(back_side_right_image_path, img_rightfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_right_max_front_back_pelvis_')
            save_optimal_frame(back_side_left_image_path, img_leftfoot_ground,
                               'Phase_Dectection_backview_left_midstance_phase_')
            save_optimal_frame(back_side_left_image_path, str(first_left_image)+"."+str(args.image_ext),
                               'Phase_Dectection_backview_left_cross_over_')
            save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_max_varus,
                               'Phase_Dectection_backview_left_max_knee_varus_')
            save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_max_femur,
                               'Phase_Dectection_backview_left_max_femur_')
            save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_max_trunk_lane,
                               'Phase_Dectection_backview_left_lateral_trunk_lean_')
            if args.pelvis == 'on':
                save_optimal_frame(back_side_left_image_path, img_leftfoot_ground_pelvis_drop,
                                'Phase_Dectection_backview_left_pelvis_angle_')
            if args.front_back_pelvis == 'on':
                save_optimal_frame(back_side_left_image_path, img_leftfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_left_max_front_back_pelvis_')
        if alignment == "front_side":
            optimal_rightfoot_on_ground, img_rightfoot_ground = optimal_frame_foot_on_ground_back_front(
                all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_right_hip_knee_ankle, all_front_shoulder_coor)
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground, img_rightfoot_ground, 'Phase_Dectection_frontview_right_midstance_phase', args.image_ext, args.pelvis, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_right_hip_knee_ankle, all_front_shoulder_coor, all_front_right_knee_leg_angle,'non_trunk')
            optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,
                'right',all_front_right_knee_leg_angle,front_right_knee_image,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground_max_varus, img_rightfoot_ground_max_varus, 'Phase_Dectection_frontview_right_max_knee_varus', args.image_ext, args.pelvis, alignment,'right_knee')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_rightfoot_on_ground_max_femour, img_rightfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_right_knee_hip_vertical, all_front_shoulder_coor, all_front_right_knee_hip_vertical,'non_trunk')
            optimal_rightfoot_on_ground_max_femur, img_rightfoot_ground_max_femur = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,
                'left',all_front_right_knee_hip_vertical,front_hip_knee_right,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground_max_femur, img_rightfoot_ground_max_femur, 'Phase_Dectection_frontview_right_max_femur', args.image_ext, args.pelvis, alignment,'right_femur')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
                # all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_trunk_lean, all_front_shoulder_coor, all_front_trunk_lean, "trunk")
            optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean = optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list, 'right',all_front_trunk_lean,front_trunck_image,im_list2,
                "trunk")
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground_lateral_trunk_lean, img_rightfoot_ground_lateral_trunk_lean, 'Phase_Dectection_frontview_right_lateral_trunk_lean', args.image_ext, args.pelvis, alignment,'trunk')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            if args.front_back_pelvis == 'on':
                optimal_rightfoot_on_ground_max_front_back_pelvis,img_rightfoot_on_ground_max_front_back_pelvis=optimal_frame_foot_on_ground_back_front_max_angles(
                right_image_list,'right',waistline_angle_list,waistline_image_list,im_list2, 'trunk')
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_rightfoot_on_ground_max_front_back_pelvis,
                                                              img_rightfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_right_max_front_back_pelvis',
                                                              args.image_ext, args.pelvis, alignment, 'right_pelvis')
                seq_list.append(seq_point_dict)
                optimal_frame_seq_list.append(seq_point_dict)

                optimal_leftfoot_on_ground_max_front_back_pelvis, img_leftfoot_on_ground_max_front_back_pelvis = optimal_frame_foot_on_ground_back_front_max_angles(
                    left_image_list, 'left', waistline_angle_list, waistline_image_list, im_list2, 'trunk')
                seq_point_dict = front_back_optimal_keypoints(all_back_right_knee_leg_angle,
                                                              all_back_left_knee_leg_angle,
                                                              all_back_right_knee_hip_vertical,
                                                              all_back_left_knee_hip_vertical,
                                                              all_back_left_hip_right_hip, all_back_trunk_lean,
                                                              optimal_leftfoot_on_ground_max_front_back_pelvis,
                                                              img_leftfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_left_max_front_back_pelvis',
                                                              args.image_ext, args.pelvis, alignment,
                                                              'left_pelvis')
                seq_list.append(seq_point_dict)
                optimal_frame_seq_list.append(seq_point_dict)

            if args.pelvis == 'on':
                optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(
                    all_front_hip_coor, all_front_right_knee_coor, all_front_right_ankle_coor, all_front_left_knee_coor, all_front_left_ankle_coor, write_image_front_left_hip_right_hip, all_front_shoulder_coor, all_front_hip_coor_left_right, video_frame_path)
                seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                            all_front_left_hip_right_hip, all_front_trunk_lean, optimal_rightfoot_on_ground_pelvis_drop, img_rightfoot_ground_pelvis_drop, 'Phase_Dectection_frontview_pelvis_angle', args.image_ext, args.pelvis, alignment,angle)
                seq_list.append(seq_point_dict)
                optimal_frame_seq_list.append(seq_point_dict)
            optimal_leftfoot_on_ground, img_leftfoot_ground = optimal_frame_foot_on_ground_back_front(
                all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_left_hip_knee_ankle, all_front_shoulder_coor)
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground, img_leftfoot_ground, 'Phase_Dectection_frontview_left_midstance_phase', args.image_ext, args.pelvis, alignment)
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_left_hip_knee_ankle, all_front_shoulder_coor, all_front_left_knee_leg_angle,'non_trunk')
            optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,'left',all_front_left_knee_leg_angle,front_left_knee_image,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground_max_varus, img_leftfoot_ground_max_varus, 'Phase_Dectection_frontview_left_max_knee_varus', args.image_ext, args.pelvis, alignment,'left_knee')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max__femour, img_leftfoot_ground_max_femour = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_left_knee_hip_vertical, all_front_shoulder_coor, all_front_left_knee_hip_vertical,'non_trunk')
            optimal_leftfoot_on_ground_max__femur, img_leftfoot_ground_max_femur = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,'left',all_front_left_knee_hip_vertical,front_hip_knee_left,im_list2, 'non_trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground_max__femur, img_leftfoot_ground_max_femur, 'Phase_Dectection_frontview_left_max_femur', args.image_ext, args.pelvis, alignment,'left_femur')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            # optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
            #     all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_trunk_lean, all_front_shoulder_coor, all_front_left_knee_hip_vertical,'trunk')
            optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane = optimal_frame_foot_on_ground_back_front_max_angles(
                left_image_list,
                'left',all_front_trunk_lean,front_trunck_image,im_list2, 'trunk')
            seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                          all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground_max_trunk_lane, img_leftfoot_ground_max_trunk_lane, 'Phase_Dectection_frontview_left_lateral_trunk_lean', args.image_ext, args.pelvis, alignment,'trunk')
            seq_list.append(seq_point_dict)
            optimal_frame_seq_list.append(seq_point_dict)
            if args.pelvis == 'on':
                optimal_leftfoot_on_ground_pelvis_drop, img_leftfoot_ground_pelvis_drop,angle = optimal_frame_foot_on_ground_back_front_all_points(
                    all_front_hip_coor, all_front_left_knee_coor, all_front_left_ankle_coor, all_front_right_knee_coor, all_front_right_ankle_coor, write_image_front_left_hip_right_hip, all_front_shoulder_coor, all_front_hip_coor_left_right,video_frame_path)
                seq_point_dict = front_back_optimal_keypoints(all_front_right_knee_leg_angle, all_front_left_knee_leg_angle, all_front_right_knee_hip_vertical, all_front_left_knee_hip_vertical,
                                                            all_front_left_hip_right_hip, all_front_trunk_lean, optimal_leftfoot_on_ground_max__femur, img_leftfoot_ground_max_femur, 'Phase_Dectection_frontview_left_pelvis_angle', args.image_ext, args.pelvis, alignment,angle)
                seq_list.append(seq_point_dict)
                optimal_frame_seq_list.append(seq_point_dict)
            save_optimal_frame(front_right_image_path, img_rightfoot_ground,
                               'Phase_Dectection_frontview_right_midstance_phase_')
            # print("2")
            save_optimal_frame(front_right_image_path, img_rightfoot_ground_max_varus,
                               'Phase_Dectection_frontview_right_max_knee_varus_')
            # print("3")
            save_optimal_frame(front_right_image_path, img_rightfoot_ground_max_femur,
                               'Phase_Dectection_frontview_right_max_femur_')
            # print("4")
            save_optimal_frame(front_right_image_path, img_rightfoot_ground_lateral_trunk_lean,
                               'Phase_Dectection_frontview_right_lateral_trunk_lean_')
            # print("5")
            if args.pelvis == 'on':
                save_optimal_frame(front_right_image_path, img_rightfoot_ground_pelvis_drop,
                                'Phase_Dectection_frontview_pelvis_angle_')
            # print("6")
            if args.front_back_pelvis == 'on':
                save_optimal_frame(front_right_image_path, img_rightfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_right_max_front_back_pelvis_')
            # print("7")
            save_optimal_frame(front_left_image_path, img_leftfoot_ground,
                               'Phase_Dectection_frontview_left_midstance_phase_')
            # print("8")
            save_optimal_frame(front_left_image_path, img_leftfoot_ground_max_varus,
                               'Phase_Dectection_frontview_left_max_knee_varus_')
            # print("9")
            save_optimal_frame(front_left_image_path, img_leftfoot_ground_max_femur,
                               'Phase_Dectection_frontview_left_max_femur_')
            # print("10")
            save_optimal_frame(front_left_image_path, img_leftfoot_ground_max_trunk_lane,
                               'Phase_Dectection_frontview_left_lateral_trunk_lean_')
            # print("11")
            if args.pelvis == 'on':
                save_optimal_frame(front_left_image_path, img_leftfoot_ground_pelvis_drop,
                                'Phase_Dectection_frontview_left_pelvis_angle_')
            # print("12")
            if args.front_back_pelvis == 'on':
                save_optimal_frame(front_left_image_path, img_leftfoot_on_ground_max_front_back_pelvis,
                                                              'Phase_Dectection_backview_left_max_front_back_pelvis_')
            # print("13")
        json_list['sequence'] = seq_list
        optimal_frame_dict['sequence'] = optimal_frame_seq_list
        with open(json_folder, 'w') as f:
            json.dump(json_list, f, ensure_ascii=False, indent=4)
        with open(optimal_json_name, 'w') as f:
            json.dump(optimal_frame_dict, f, ensure_ascii=False, indent=4)
    if(int(args.output_option) == 2 ):
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
    if (int(args.output_option) == 3 or int(args.output_option) == 4):
        cpoy_left_video_shell_script = 'cp ' + out_put_image_dir + '/left/movie/*.mp4 ' + args.output_dir + '/'
        cpoy_right_video_shell_script = 'cp ' + out_put_image_dir + \
                                        '/right/movie/*.mp4 ' + args.output_dir + '/'
        cpoy_json_shell_script = 'cp ' + out_put_image_dir + '/*.json ' + args.output_dir + '/'
        copy_left_optimalframe_shell_script = 'mv ' + out_put_image_dir + \
                                              '/left/images/Phase_Dectection* ' + args.output_dir + '/'
        copy_right_optimalframe_shell_script = 'mv ' + out_put_image_dir + \
                                               '/right/images/Phase_Dectection* ' + args.output_dir + '/'
        side = ""
        if alignment == "front_side":
            side = "frontview"
            cpoy_frontbothside_video_shell_script = 'mv ' + out_put_image_dir + 'both_side/*.mp4 ' + args.output_dir + '/'
            os.system(cpoy_frontbothside_video_shell_script)
            new_image_name_dir = "Merged_frontview_left_and_right_extremity"
            new_directory_merged_images = args.output_dir + '/' + new_image_name_dir + '/'
            os.makedirs(new_directory_merged_images)
            copy_merged_images_shell_script = 'cp ' + out_put_image_dir + \
                                            '/both_side/images/* ' + new_directory_merged_images + '/'
            os.system(copy_merged_images_shell_script)
            if(args.auto_resize=='on'):
                resize_images_merged = 'optipng '+ args.output_dir + '*'+str(args.image_ext)
                os.system(resize_images_merged)
        if alignment == "back_side":
            side = "backview"
            cpoy_backbothside_video_shell_script = 'mv ' + out_put_image_dir + 'both_side/*.mp4 ' + args.output_dir + '/'
            os.system(cpoy_backbothside_video_shell_script)
            new_image_name_dir = "Merged_backview_left_and_right_extremity"
            new_directory_merged_images = args.output_dir + '/' + new_image_name_dir + '/'
            os.makedirs(new_directory_merged_images)
            copy_merged_images_shell_script = 'cp ' + out_put_image_dir + \
                                              '/both_side/images/* ' + new_directory_merged_images + '/'
            os.system(copy_merged_images_shell_script)
            if(args.auto_resize=='on'):
                resize_images_merged = 'optipng '+ args.output_dir + '*'+str(args.image_ext)
                os.system(resize_images_merged)
        if alignment == "right_side":
            side = "rightview"
        if alignment == "left_side":
            side = "leftview"
        images_folder_name_left = 'Processed_frames_' + side +"_leftextremity_"   + video_name + '_' + creation_date
        images_folder_name_right= 'Processed_frames_' + side +"_rightextremity_" + video_name + '_' + creation_date
        new_directory_left = args.output_dir + '/' + images_folder_name_left + '/'
        new_directory_right = args.output_dir + '/' + images_folder_name_right + '/'
        os.makedirs(new_directory_left)
        os.makedirs(new_directory_right)
        copy_left_images_shell_script = 'cp ' + out_put_image_dir + \
                                              '/left/images/* ' + new_directory_left + '/'
        copy_right_images_shell_script = 'cp ' + out_put_image_dir + \
                                               '/right/images/* ' + new_directory_right + '/'
        if(int(args.output_option) == 4 and alignment == "right_side"):
            os.system(cpoy_right_video_shell_script)
            os.system(cpoy_json_shell_script)
            os.system(copy_left_optimalframe_shell_script)
            os.system(copy_right_optimalframe_shell_script)
            os.system(copy_left_images_shell_script)
            os.system(copy_right_images_shell_script)
        elif(int(args.output_option) == 4 and alignment == "left_side"):
            os.system(cpoy_left_video_shell_script)
            os.system(cpoy_json_shell_script)
            os.system(copy_left_optimalframe_shell_script)
            os.system(copy_right_optimalframe_shell_script)
            os.system(copy_left_images_shell_script)
            os.system(copy_right_images_shell_script)
        else:
            os.system(cpoy_left_video_shell_script)
            os.system(cpoy_right_video_shell_script)
            os.system(cpoy_json_shell_script)
            os.system(copy_left_optimalframe_shell_script)
            os.system(copy_right_optimalframe_shell_script)
            os.system(copy_left_images_shell_script)
            os.system(copy_right_images_shell_script)
        if(args.auto_resize=='on'):
            resize_images_left='optipng '+ args.output_dir + '*'+str(args.image_ext)
            resize_images_right = 'optipng ' + args.output_dir + '*'+str(args.image_ext)
            os.system(resize_images_left)
            os.system(resize_images_right)

    if(int(args.output_option) == 2 or int(args.output_option) == 3 or int(args.output_option) == 4):
        remove_olfiles_script = 'rm -rf ' + out_put_image_dir + ' ' + args.output_dir + '/images'
        remove_images_folder_script = 'rm -rf ' + args.output_dir + '/images'
        os.system(remove_olfiles_script)
        os.system(remove_images_folder_script)
    if (int(args.output_option) == 3 or int(args.output_option) == 4):
        if alignment == "back_side" or alignment == "front_side":
            remove_both_images = 'rm -rf ' + out_put_image_dir + 'both_side/'
            os.system(remove_both_images)
        if int(args.output_option) == 4 and alignment == "left_side" :
            remove_processed_images_folder_script = 'rm -rf ' + args.output_dir + '/Processed_frames_leftview_right*'
            os.system(remove_processed_images_folder_script)

        if int(args.output_option) == 4 and alignment == "right_side" :
            remove_processed_images_folder_script = 'rm -rf ' + args.output_dir + '/Processed_frames_rightview_left*'
            os.system(remove_processed_images_folder_script)
def creation_date_fun(filename):
    '''
        Input: video file path
        return: creation date of that video
    '''
    parser = createParser(filename)
    metadata = extractMetadata(parser)
    return metadata.get('creation_date') + timedelta(hours=-7)
def test(args):
    try:
        main(args)
        return 0
    except:
        return -1
if __name__ == '__main__':
    args = parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
    model = DefaultPredictor(cfg)
    if args.input_type == "video" and os.path.isdir(args.video_path):
        input_dir = args.video_path
        output_dir = args.output_dir
        for filename in os.listdir(input_dir):
            if filename.endswith("." + args.video_type):
                args.video_path = os.path.join(input_dir, filename)
                args.output_dir = output_dir
                try:
                    main(args)
                    # msg=test(args)
                    # a = os.popen(str(msg))
                except Exception as e:
                    print("Error has been found in video 1 ----", args.video_path)
                    print("eeee", e)
                    logging.error(traceback.format_exc())
            else:
                print("No video is found with extension  ", args.video_type)
    else:
        try:
            main(args)
            # msg=test(args)
            # a = os.popen(str(msg))
        except Exception as e:
            print("Error has been found in video 2 ----", args.video_path)
            logging.error(traceback.format_exc())
