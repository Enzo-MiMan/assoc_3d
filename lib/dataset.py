import cv2
import glob
import os
# from lib.read_data import *
from os.path import join
import numpy as np


class Single_Loader():

    def __init__(self, data_path):
        self.data_path = data_path
        self.depth_gap4_list_dst = sorted(glob.glob(join(data_path, 'enzo_depth_gt_dst/*.txt')))
        self.depth_gap4_list_src = sorted(glob.glob(join(data_path, 'enzo_depth_gt_src/*.txt')))

    def __getitem__(self, index):

        _, file_name_dst = os.path.split(self.depth_gap4_list_dst[index])
        _, file_name_sec = os.path.split(self.depth_gap4_list_src[index])
        timestamp_dst, _ = os.path.splitext(file_name_dst)
        timestamp_src, _ = os.path.splitext(file_name_sec)

        _, file_name_src = os.path.split(self.depth_gap4_list_src[index])
        timestamp_src, _ = os.path.splitext(file_name_src)

        img_file_dst = join(self.data_path, 'enzo_depth', timestamp_dst+'.png')
        img_file_src = join(self.data_path, 'enzo_depth', timestamp_src+'.png')

        image_dst_org = cv2.imread(img_file_dst)
        image_src_org = cv2.imread(img_file_src)

        image_dst = cv2.cvtColor(image_dst_org, cv2.COLOR_BGR2GRAY)
        image_src = cv2.cvtColor(image_src_org, cv2.COLOR_BGR2GRAY)

        image_dst = image_dst.reshape(1, image_dst.shape[0], image_dst.shape[1])
        image_src = image_src.reshape(1, image_src.shape[0], image_src.shape[1])

        return timestamp_dst, timestamp_src, image_dst, image_src, image_dst_org, image_src_org

    def __len__(self):
        return len(self.depth_gap4_list_dst)+1


class Pair_Loader():

    def __init__(self, data_path):
        self.data_path = data_path
        self.depth_gap4_list_dst = sorted(glob.glob(join(data_path, 'enzo_depth_gt_dst/*.txt')))
        self.depth_gap4_list_src = sorted(glob.glob(join(data_path, 'enzo_depth_gt_src/*.txt')))


    def __getitem__(self, index):
        _, file_name_dst = os.path.split(self.depth_gap4_list_dst[index])
        timestamp_dst, _ = os.path.splitext(file_name_dst)

        _, file_name_src = os.path.split(self.depth_gap4_list_src[index])
        timestamp_src, _ = os.path.splitext(file_name_src)

        img_file_dst = join(self.data_path, 'enzo_depth', timestamp_dst+'.png')
        img_file_src = join(self.data_path, 'enzo_depth', timestamp_src+'.png')

        image_dst_org = cv2.imread(img_file_dst)
        image_src_org = cv2.imread(img_file_src)

        image_dst = cv2.cvtColor(image_dst_org, cv2.COLOR_BGR2GRAY)
        image_src = cv2.cvtColor(image_src_org, cv2.COLOR_BGR2GRAY)

        image_dst = image_dst.reshape(1, image_dst.shape[0], image_dst.shape[1])
        image_src = image_src.reshape(1, image_src.shape[0], image_src.shape[1])

        return timestamp_dst, timestamp_src, image_dst, image_src, image_dst_org, image_src_org

    def __len__(self):
        return len(self.depth_gap4_list_dst)
