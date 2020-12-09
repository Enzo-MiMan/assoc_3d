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
from os.path import join
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


if __name__ == "__main__":

    project_dir = os.path.dirname(os.getcwd())
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = join(os.path.dirname(project_dir), 'indoor_data')
    exp_names = cfg['radar']['exp_name']
    all_sequences = cfg['radar']['all_sequences']
    train_sequences = cfg['radar']['training']
    valid_sequences = cfg['radar']['validating']
    test_sequences = cfg['radar']['testing']
    gap = cfg['radar']['gap']

    for sequence in all_sequences:

        path = join(data_dir, str(sequence))
        if not os.path.exists(path):
            continue

        matches = timestamp_match(data_dir, sequence, gap)
        with open(os.path.join(data_dir, str(sequence), 'enzo_ts_match_gap4.txt'), 'a+') as file:
            for pair in matches:
                file.write(str(pair[0]) + " " + str(pair[1]) + '\n')

