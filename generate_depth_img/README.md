### file details:   
     
      mmwave_bag.py:
            input:    e.g. " _slash_mmWaveDataHdl_slash_RScan_middle"
            
            return dictionary:  {'1574955812821671721': [[x, y, z], [x, y, z], ...]], 
                                 '1574955812921587941': [[x, y, z], [x, y, z], ...]], 
                                  ...} 
                                                   
      pcl2depth.py:
            input:   velo_points_2_pano(frame[eff_rows_idx, :], v_res, h_res, v_fov, h_fov, max_v, depth=True)
            return:  depth_img, pixel_coordinate, world_coordinate
            
            
      radar_pcl2depth.py (main code):
          input:
                data_dir = cfg['base_conf']['data_base']
                exp_names = cfg['radar']['exp_name']
                sequence_names = cfg['radar']['all_sequences']

                middle_transform = np.array(cfg['radar']['translation_matrix']['middle'])
                left_transform = np.array(cfg['radar']['translation_matrix']['left'])
                right_transform = np.array(cfg['radar']['translation_matrix']['right'])

                align_interval = np.array(cfg['radar']['align_interval'])

                v_fov = tuple(map(int, cfg['pcl2depth']['v_fov'][1:-1].split(',')))
                h_fov = tuple(map(int, cfg['pcl2depth']['h_multi_fov'][1:-1].split(',')))

                topic_middle = '_slash_mmWaveDataHdl_slash_RScan_middle'
                topic_left = '_slash_mmWaveDataHdl_slash_RScan_left'
                topic_right = '_slash_mmWaveDataHdl_slash_RScan_right'
            
            output:
                1. depth image
                2. pixel coordinate 
                3. world coordinate
                
            notes: 
                1. points within 5.12m   --->  (x**2 +y**2) ** 0.5 < 5.12
                2. all points in the frame. can be used to test
            
                
            
            


