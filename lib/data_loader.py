import cv2
import glob
import os
from os.path import join
import numpy as np


class Pair_Loader():

    def __init__(self, sequence_path):
        self.data_path = sequence_path
        self.depth_gap4_list_src = sorted(glob.glob(join(sequence_path, 'enzo_depth_gt_src/*.txt')))
        self.depth_gap4_list_dst = sorted(glob.glob(join(sequence_path, 'enzo_depth_gt_dst/*.txt')))

    def __getitem__(self, index):
        """only get depth image for training, do not read ground truth"""

        _, file_name_dst = os.path.split(self.depth_gap4_list_dst[index])
        _, file_name_sec = os.path.split(self.depth_gap4_list_src[index])
        timestamp_dst, _ = os.path.splitext(file_name_dst)
        timestamp_src, _ = os.path.splitext(file_name_sec)

        # read image
        img_file_dst = join(self.data_path, 'enzo_depth', str(timestamp_dst)+'.png')
        img_file_src = join(self.data_path, 'enzo_depth', str(timestamp_src)+'.png')

        image_dst_org = cv2.imread(img_file_dst, cv2.COLOR_BGR2GRAY)
        image_src_org = cv2.imread(img_file_src, cv2.COLOR_BGR2GRAY)

        image_dst = image_dst_org.reshape(1, image_dst_org.shape[0], image_dst_org.shape[1])
        image_src = image_src_org.reshape(1, image_src_org.shape[0], image_src_org.shape[1])

        return timestamp_dst, timestamp_src, image_dst, image_src

    def __len__(self):
        return len(self.depth_gap4_list_dst)




