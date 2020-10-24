import numpy as np
import pandas as pd


def normalize_depth(val, min_v, max_v):
    """
    print 'nomalized depth value'
    nomalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
    """

    return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)


def normalize_val(val, min_v, max_v):
    """
    print 'nomalized depth value'
    nomalize values to 0-255 & close distance value has low value.
    """
    return (((val - min_v) / (max_v - min_v)) * 255).astype(np.uint8)


def in_h_range_points(x, y, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180), np.arctan2(y, x) < (-fov[0] * np.pi / 180))


def in_v_range_points(d, z, fov):
    """ extract vertical in-range points """
    return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), np.arctan2(z, d) > (fov[0] * np.pi / 180))


def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """ filter points based on h,v FOV  """
    h_points = in_h_range_points(x, y, h_fov)
    v_points = in_v_range_points(np.sqrt(x ** 2 + y ** 2), z, v_fov)
    return points[np.logical_and(h_points, v_points)]



def velo_points_2_pano(points, v_res, h_res, v_fov, h_fov, max_v, depth=True):

    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # project point cloud to 2D point map
    x_img = np.arctan2(-y, x) / (h_res * (np.pi / 180))
    y_img = -(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) / (v_res * (np.pi / 180)))

    """ filter points based on h,v FOV  """
    x_img = fov_setting(x_img, x, y, z, dist, h_fov, v_fov)
    y_img = fov_setting(y_img, x, y, z, dist, h_fov, v_fov)
    dist = fov_setting(dist, x, y, z, dist, h_fov, v_fov)

    """ directly return dist if dist empty  """
    if dist.size == 0:
        return dist

    x_size = int(np.ceil((h_fov[1] - h_fov[0]) / h_res))
    y_size = int(np.ceil((v_fov[1] - v_fov[0]) / v_res))

    # shift negative points to positive points (shift minimum value to 0)
    x_offset = h_fov[0] / h_res
    x_img = np.trunc(x_img - x_offset).astype(np.int32)
    y_offset = v_fov[1] / v_res
    y_fine_tune = 1
    y_img = np.trunc(y_img + y_offset + y_fine_tune).astype(np.int32)


    if depth:
        # nomalize distance value & convert to depth map
        dist = normalize_depth(dist, min_v=0, max_v=max_v)
    else:
        dist = normalize_val(dist, min_v=0, max_v=max_v)

    # array to img
    img = np.zeros([y_size + 1, x_size + 2], dtype=np.uint8)
    img[y_img, x_img] = dist

    point_info = np.array([y_img, x_img, dist]).T

    df = pd.DataFrame(point_info, columns=['row', 'col', 'dist'])
    df.drop_duplicates(subset=['row', 'col'], keep='last', inplace=True)
    point_info = np.array(df)

    return img, point_info