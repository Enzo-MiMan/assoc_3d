import cv2
import glob
import os
# from lib.read_data import *
from os.path import join
import numpy as np


class Scan_Loader():
    def __init__(self, data_path):
        self.data_path = data_path
        self.gt_files_dst = sorted(glob.glob(join(data_path, 'depth_gt_dst/*.txt')))
        self.gt_files_src = sorted(glob.glob(join(data_path, 'depth_gt_src/*.txt')))

    def read_img(self, gt_file):
        file_path, full_flname = os.path.split(gt_file)
        timestamp, _ = os.path.splitext(full_flname)
        image_path = join(self.data_path, 'depth_enzo', timestamp + '.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        return image, timestamp
        
    def __getitem__(self, index):

        # ------------ read gt ------------
        gt_dst_file = self.gt_files_dst[index]
        gt_src_file = self.gt_files_src[index]

        # ------------ read image pair ------------
        image_dst, dst_timestamp = self.read_img(gt_dst_file)
        image_src, src_timestamp = self.read_img(gt_src_file)

        return image_dst, image_src, dst_timestamp, src_timestamp


    def __len__(self):
        return len(self.gt_files_dst)-1


class Scan_Loader_NoLabel():
    """cdc"""


    def __init__(self, data_path):
        self.data_path = data_path
        self.depth_gap4_list_dst = sorted(glob.glob(join(data_path, 'depth_gt_dst/*.txt')))
        self.depth_gap4_list_src = sorted(glob.glob(join(data_path, 'depth_gt_src/*.txt')))


    def __getitem__(self, index):
        _, file_name_dst = os.path.split(self.depth_gap4_list_dst[index])
        timestamp_dst, _ = os.path.splitext(file_name_dst)

        _, file_name_src = os.path.split(self.depth_gap4_list_src[index])
        timestamp_src, _ = os.path.splitext(file_name_src)

        img_file_dst = join(self.data_path, 'depth_enzo', timestamp_dst+'.png')
        img_file_src = join(self.data_path, 'depth_enzo', timestamp_src+'.png')

        image_dst_org = cv2.imread(img_file_dst)
        image_src_org = cv2.imread(img_file_src)

        image_dst = cv2.cvtColor(image_dst_org, cv2.COLOR_BGR2GRAY)
        image_src = cv2.cvtColor(image_src_org, cv2.COLOR_BGR2GRAY)

        image_dst = image_dst.reshape(1, image_dst.shape[0], image_dst.shape[1])
        image_src = image_src.reshape(1, image_src.shape[0], image_src.shape[1])

        return timestamp_dst, timestamp_src, image_dst, image_src, image_dst_org, image_src_org

    def __len__(self):
        return len(self.depth_gap4_list_dst)




