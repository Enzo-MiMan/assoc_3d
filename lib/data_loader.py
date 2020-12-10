import cv2
import glob
import os
from os.path import join
import numpy as np


class Pair_Loader():

    def __init__(self, sequence_path):
        self.sequence_path = sequence_path
        timestamps_path = join(sequence_path, 'enzo_timestamp_matches.txt')
        self.mm_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

    def __getitem__(self, index):
        """only get depth image for training, do not read ground truth"""

        timestamp_dst = self.mm_timestamps[index]
        timestamp_src = self.mm_timestamps[index+1]

        # read image
        img_file_dst = join(self.sequence_path, 'enzo_depth_images', str(timestamp_dst)+'.png')
        img_file_src = join(self.sequence_path, 'enzo_depth_images', str(timestamp_src)+'.png')
        image_dst_org = cv2.imread(img_file_dst, cv2.COLOR_BGR2GRAY)
        image_src_org = cv2.imread(img_file_src, cv2.COLOR_BGR2GRAY)

        image_dst = image_dst_org.reshape(1, image_dst_org.shape[0], image_dst_org.shape[1])
        image_src = image_src_org.reshape(1, image_src_org.shape[0], image_src_org.shape[1])

        return timestamp_dst, timestamp_src, image_dst, image_src

    def __len__(self):
        return len(self.mm_timestamps) - 1




