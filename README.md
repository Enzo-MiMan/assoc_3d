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
    squeue              # find the JOBID that USER is xxlu and NAME is runsc.sh
    jinru <JOBID>       # e.g.   jinru 773276
    cd /data/greyostrich/not-backed-up/aims/aimsre/xxlu/assoc/workspace/indoor_data
    
  
    
    
### Data pre-process

    1. timestamp matches
        match mm-wave timestamps and gamapping timestamps. If the time difference is less than , 
        we consider them as the matched point cloud that had been collected at the same time.
        
        code file: timestamp_match_mm_gmapping.py
   
   
   -----------------------------------------------------------------------
   
    2. extract gmapping: read gmapping translation and rotation

        code file:
            gmapping_R_T_from_csv.py
            
        input file:
            true_delta_gmapping.csv

        output file:
            gmapping_T.txt, gmapping_R_matrix.txt  

   -----------------------------------------------------------------------

    3. overlay the left, middle, and right point clounds:
    overlay 
    
    
    input file:
        _slash_mmWaveDataHdl_slash_RScan_left.csv        
        _slash_mmWaveDataHdl_slash_RScan_middle.csv        
        _slash_mmWaveDataHdl_slash_RScan_right.csv 
        
        
    output file:
    
    
    
    
    
    

            
            
        
### main code -- corres_gmapping_aided.py

    input:
        1. timestamp matches:   mm_gmapping_timestamp_match.txt
        2. gmapping:  gmapping_T.txt, gmapping_R_matrix.txt
        3. LMR point cloud:   

    parameter:
        gap = 4   # read from config.yaml

    output:
        source point cloud:  mm_src_GA_sample.txt
        matched destination point cloud:  mm_dts_GA_sample.txt