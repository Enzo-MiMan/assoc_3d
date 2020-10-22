### Data allocation

    training:
        - 2019-10-27-14-28-21
        - 2019-10-27-14-39-09
        - 2019-10-27-14-42-16
        - 2019-10-27-14-47-00
        - 2019-10-27-14-50-17
        - 2019-10-27-14-56-27
        - 2019-10-27-15-02-46
        - 2019-10-27-15-06-06
        - 2019-10-27-15-09-33
        - 2019-10-27-15-13-03
        - 2019-10-27-15-18-07
        - 2019-10-27-15-24-29
        - 2019-10-27-15-26-35
        - 2019-10-27-15-33-43
        - 2019-11-04-20-29-51
        - 2019-11-04-20-38-57
        - 2019-11-04-20-44-42
        - 2019-11-04-21-14-16
        - 2019-11-22-10-10-00
    validating:
        - 2019-11-22-10-14-01
        - 2019-11-22-10-22-48
        - 2019-11-22-10-26-42
        - 2019-11-22-10-34-57
    testing:
        - 2019-11-22-10-36-00
        - 2019-11-22-10-37-42
        - 2019-11-22-10-38-47
        - 2019-11-28-15-40-10
        - 2019-11-28-15-43-32



### Access the dataset:

    ssh xxlu@gate.stats.ox.ac.uk
    ssh xxlu@greytail.stats.ox.ac.uk       
    # you can use "ssh oxford" to login directly (replace the above 2 steps), because you had set up the config file
    squeue              # find the JOBID that USER is xxlu and NAME is runsc.sh
    jinru <JOBID>       # e.g.   jinru 773276
    cd /data/greyostrich/not-backed-up/aims/aimsre/xxlu/assoc/workspace/indoor_data
    
  
    
    
### Data pre-process

    1. timestamp matches: matching mm-wave timestamps and gamapping timestamps      
        Find the closest gamapping timestamp for each mm-wave timestamp
        
        code file: timestamp_match_mm_gmapping.py
        
        input:
            1. mm-wave middle board data: _slash_mmWaveDataHdl_slash_RScan_middle.csv
            2. gmapping data: true_delta_gmapping.csv

        parameter:
            gap = 4   # read from config.yaml

        return:
            timestamp matching: mm-wave_timestamps, gmapping_timestamps  # for the first and second columns respectively

   
   -----------------------------------------------------------------------
   
    2. extract gmapping: read gmapping translation and rotation

        code file:
            gmapping_R_T_from_csv.py
            
        input:
            true_delta_gmapping.csv  # include fields: timestamp, x, y, z, qx, qy, qz, qw

        return:
            translation, rotation    # array (frame_num-1, 4),  array (frame_num-1, 10)

   -----------------------------------------------------------------------

    3. overlay the left, middle, and right point clounds:
    
        input file:
            _slash_mmWaveDataHdl_slash_RScan_left.csv        
            _slash_mmWaveDataHdl_slash_RScan_middle.csv        
            _slash_mmWaveDataHdl_slash_RScan_right.csv 


        output file:
                LMR_xyz/
                        1574955812871341906.xyz
                        1574955812921587941.xyz
                        .....

            
        
### Main code

    code file: corres_gmapping_aided.py

    input:
        1. timestamp matches:   mm-wave_timestamps, gmapping_timestamps
        2. gmapping:  translation, rotation
        3. LMR point cloud:   LMR_xyz/1574955812871341906.xyz                            
                                      1574955812921587941.xyz   
                                      .....
                       
    output:
        source point cloud: mm_src_gt_3d.txt  
        matched destination point cloud:  mm_dts_gt_3d.txt 
        
        
        
### Training data and ground truth (example with a single sequence:2019-10-27-14-28-21)

    training data:
        2019-10-27-14-28-21/LMR_xyz/                             # LMR_xy: a folder include all frames of a sequence
                                    1574955812871341906.xyz      # 1574955812871341906.xyz: a frame of point cloud
                                    1574955812921587941.xyz
                                    .....
            
    ground truth:
        2019-10-27-14-28-21/mm_src_gt_3d.txt
        2019-10-27-14-28-21/mm_dts_gt_3d.txt
        
        format:
           --------------------------------------------------------------------------------------------------------------------------------------
          |      mm_src_gt_3d.txt                                              |    mm_dts_gt_3d.txt                                             |
          |--------------------------------------------------------------------|-----------------------------------------------------------------|
          | timestamp  x1 y1 z1 x2 y2 z2 x3 y3 z3 .....     # frame k + 4      | timestamp  x1 y1 z1 x2 y2 z2 x3 y3 z3 .....     # frame k       |
          | timestamp  x1 y1 z1 x2 y2 z2 x3 y3 z3 .....     # frame k + 8      | timestamp  x1 y1 z1 x2 y2 z2 x3 y3 z3 .....     # frame k + 4   |
          | timestamp  x1 y1 z1 x2 y2 z2 x3 y3 z3 .....     # frame k + 12     | timestamp  x1 y1 z1 x2 y2 z2 x3 y3 z3 .....     # frame k + 8   | 
          | .....                                                              | .....                                                           |
           --------------------------------------------------------------------------------------------------------------------------------------
          x1 x2 x3 of frame k+4 in mm_src_gt_3d.txt  is corresponding to x1 x2 x3 of frame k in mm_src_gt_3d.txt
          x1 x2 x3 of frame k+8 in mm_src_gt_3d.txt  is corresponding to x1 x2 x3 of frame k+4 in mm_src_gt_3d.txt
          
          note: 
            1. the number of values in each frame(row) can be divisible by 3, since they are all made up of x, y, and z
            2. the number of values between differnent frames(columns) are different.
            3. the associated frame pairs (source point cloud and destination point cloud) have the same number of points
            4. the number of frames in training data (namely the number of files in folder LMR_xyz) is quadruple than ground truth frames, 
               because we use the gap=4 to implement the point cloud matching. From our experiment, we find that with gap=4, the point clound correspondence 
               is more accurate. The frequency of mm-wave radar collecting data is pretty high, so with smaller gap, the correspondences may have 
               lower precision.
            
  
        
### Download data
        step 1： pull data (both folder and file) from greyostrich to greytail  
                    rsync -avz  /data/path/in/greyostrich   greytail:/destination/path/in/greytail
         
        step 2： pull data file from greytail to local 
                    scp  oxford:/file/path/in/greytail   /your/local/path

                 or pull data folder from greytail to local 
                    scp -r oxford:/folder/path/in/greytail  /your/local/path


### The data you need are:   
        2019-10-27-14-28-21
                LMR_xyz    
                mm_src_gt_3d.txt
                mm_dts_gt_3d.txt
                
        2019-10-27-14-39-09
                LMR_xyz   
                mm_src_gt_3d.txt
                mm_dts_gt_3d.txt
                
        ....



