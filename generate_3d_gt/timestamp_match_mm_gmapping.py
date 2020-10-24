"""
input:
    1. mm-wave middle board data: _slash_mmWaveDataHdl_slash_RScan_middle.csv
    2. gmapping data: true_delta_gmapping.csv

parameter:
    gap = 4   # read from config.yaml

output:
    timestamp matching: timestamp matching: mm-wave, gmapping # for the first and second columns respectively

"""

import numpy as np
import pandas as pd
import yaml
import os


def closest_timestamp(timestamp, gmapping_ts):
    min_distance = timestamp
    closest_timestamp = gmapping_ts[0]

    for ts in gmapping_ts:
        distance = abs(ts - timestamp)
        
        if distance < min_distance:
            min_distance = distance
            closest_timestamp = ts

    return closest_timestamp



def timestamp_match(data_dir, sequence, gap):
    # ------------------------ read mm-wave timestamps from .csv file ------------------------
    mm_path = os.path.join(data_dir, str(sequence), '_slash_mmWaveDataHdl_slash_RScan_middle.csv')
    mm = pd.read_csv(mm_path)
    mm_ts = np.array(mm['rosbagTimestamp'])

    mm_ts_gap = list()
    for i in range(0, len(mm_ts), gap):
        mm_ts_gap.append(mm_ts[i])


    # ------------------------ read gmapping timestamps ------------------------

    mapping_path = os.path.join(data_dir, str(sequence), 'true_delta_gmapping.csv')
    gmapping = pd.read_csv(mapping_path)
    gmapping_ts = np.array(gmapping['timestamp'])


    # ------------------------ match ------------------------

    matches = list()
    for timestamp in mm_ts_gap:
        gmapping_mathced_ts = closest_timestamp(timestamp, gmapping_ts)
        matches.append((timestamp, gmapping_mathced_ts))

    df = pd.DataFrame(matches, columns=['one', 'two'])
    df.drop_duplicates('two', keep='first', inplace=True)
    matches = np.array(df)

    return matches


    # ------------------------ save ------------------------

    # with open(os.path.join(data_dir, str(sequence), 'ts_match_gap.txt'),'a+') as file:
    #     for pair in matches:
    #         file.write(str(pair[0]) + " " + str(pair[1]) + '\n')

