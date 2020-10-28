import struct
import csv
import numpy as np


def make_frames_from_csv(csv_path):
    readings_dict = dict()
    with open(csv_path, 'r') as input_file:
        reader = csv.reader(input_file)
        next(reader)
        for row in reader:
            pts = list()
            # add timestamp
            timestamp = row[0]  # timestamp = row[4] + row[5].zfill(9)
            # parsing
            try:
                offset_col = row[37]
            except:
                offset_col = row[29]

            pt_cloud = np.fromstring(offset_col[1:-1], dtype=int, sep=',')

            for i in range(0, int(len(pt_cloud) / 32)):
                point = list()
                # x
                tmp = struct.pack('4B', int(pt_cloud[32 * i]), int(pt_cloud[32 * i + 1]), int(pt_cloud[32 * i + 2]),
                                  int(pt_cloud[32 * i + 3]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # y
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 4]), int(pt_cloud[32 * i + 5]), int(pt_cloud[32 * i + 6]),
                                  int(pt_cloud[32 * i + 7]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                # z
                tmp = struct.pack('4B', int(pt_cloud[32 * i + 8]), int(pt_cloud[32 * i + 9]),
                                  int(pt_cloud[32 * i + 10]),
                                  int(pt_cloud[32 * i + 11]))
                tempf = struct.unpack('1f', tmp)
                point.append(tempf[0])
                pts.append(point)
            readings_dict[timestamp] = pts

        return readings_dict