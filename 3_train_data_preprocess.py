import numpy as np
import os
from os.path import join
import yaml
import cv2
import tqdm
from lib.utils import re_mkdir_dir, extract_frames, data_prepare
from lib.pcl2depth import filter_point, velo_points_2_pano

if __name__ == '__main__':

    # ------------------------ get config ------------------------

    project_dir = os.getcwd()
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    exp_names = cfg['radar']['exp_name']
    all_sequences = cfg['radar']['all_sequences']
    train_sequences = cfg['radar']['training']
    valid_sequences = cfg['radar']['validating']
    test_sequences = cfg['radar']['testing']

    v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
    h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))
    v_res = cfg['pcl2depth']['v_res']
    h_res = cfg['pcl2depth']['h_res']
    max_v = cfg['pcl2depth']['max_v']
    mmwave_dist_thre = cfg['pcl2depth']['mmwave_dist_thre']


    sequeence_num = 0
    all_sequences_num = len(all_sequences)
    for sequence in all_sequences:

        if not os.path.exists(join(data_dir, str(sequence))):
            continue
        print("sequence: {}/{},   {}".format(sequeence_num, all_sequences_num, sequence))
        sequence_dir = join(data_dir, str(sequence))

        # ----------------------- get data ---------------------------

        frames = extract_frames(join(sequence_dir, 'enzo_LMR_xyz'))

        ts_matches = join(sequence_dir, 'enzo_timestamp_matches.txt')
        mm_timestamps = np.loadtxt(ts_matches, delimiter=' ', usecols=[0], dtype=np.int64)

        # ------------------------- pcl to depth -------------------------

        # output file rebuild
        depth_image_dir = re_mkdir_dir(join(sequence_dir, 'enzo_depth_images'))
        pixel_coord_dir = re_mkdir_dir(join(sequence_dir, 'enzo_all_pixel'))
        world_coord_dir = re_mkdir_dir(join(sequence_dir, 'enzo_all_world'))

        corrd_dict = dict()
        for timestamp in tqdm.tqdm(mm_timestamps, total=len(mm_timestamps)):
            frame = frames[str(timestamp)]

            # --------------------- filter effective points -------------------
            """only select those points with the certain range (in meters) - 5.12 meter for this TI board"""
            eff_rows_idx = (frame[:, 0] ** 2 + frame[:, 1] ** 2) ** 0.5 < mmwave_dist_thre
            eff_points = frame[eff_rows_idx, :]

            """ filter points based on v_fov, h_fov  """
            valid_index = filter_point(eff_points, v_fov, h_fov)
            frame = eff_points[valid_index, :]

            # ------------------------- pcl to depth -------------------------

            pano_img, pixel_coord, world_coord = velo_points_2_pano(frame, v_res, h_res,
                                                                    v_fov, h_fov, max_v, depth=True)

            img_path = join(depth_image_dir, '{}.png'.format(timestamp))
            cv2.imwrite(img_path, pano_img)

            pixel_coord_file = join(pixel_coord_dir, '{}.txt'.format(timestamp))
            with open(pixel_coord_file, 'a+') as myfile:
                for x, y in pixel_coord:
                    myfile.write(str(x) + " " + str(y) + '\n')

            world_coord_file = join(world_coord_dir, '{}.txt'.format(timestamp))
            with open(world_coord_file, 'a+') as myfile:
                for x, y, z in world_coord:
                    myfile.write(str(x) + " " + str(y) + ' ' + str(z) + '\n')

        print('finished process {}'.format(sequence))
        sequeence_num += 1
