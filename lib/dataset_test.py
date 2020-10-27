import cv2
import glob
import os
# from lib.read_data import *
from os.path import join
import numpy as np


class Scan_Loader():
    def __init__(self, data_path):
        self.data_path = data_path
        self.coordinate_files = sorted(glob.glob(join(data_path, 'all_point_depth_pixel_coordinate/*.txt')))


        
    def __getitem__(self, index):

        # ------------ read gt ------------
        point_dst_file = self.coordinate_files[4*index]
        if (4*index+4) > len(self.coordinate_files):
            return
        else:
            point_src_file = self.coordinate_files[4*index+4]

        # ------------ read image pair ------------

        dst_filepath, dst_fullflname = os.path.split(point_dst_file)
        src_filepath, src_fullflname = os.path.split(point_src_file)
        dst_timestamp, _ = os.path.splitext(dst_fullflname)
        src_timestamp, _ = os.path.splitext(src_fullflname)

        image_dst_path = join(self.data_path, 'depth_enzo', dst_timestamp+'.png')
        image_src_path = join(self.data_path, 'depth_enzo', src_timestamp+'.png')

        image_dst = cv2.imread(image_dst_path)
        image_src = cv2.imread(image_src_path)

        image_dst = cv2.cvtColor(image_dst, cv2.COLOR_BGR2GRAY)
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)

        image_dst = image_dst.reshape(1, image_dst.shape[0], image_dst.shape[1])
        image_src = image_src.reshape(1, image_src.shape[0], image_src.shape[1])

        return image_dst, image_src, dst_timestamp, src_timestamp


    def __len__(self):
        return len(self.coordinate_files)







