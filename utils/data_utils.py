import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample

from utils.sig_proc_utils import notch_filter, baseline_correction


def window_slice(data, window_size, stride, channel_mode='channel_last'):
    assert len(data.shape) == 2
    if channel_mode == 'channel_first':
        data = np.transpose(data)
    elif channel_mode == 'channel_last':
        pass
    else:
        raise Exception('Unsupported channel mode')
    assert window_size <= len(data)
    assert stride > 0
    rtn = np.expand_dims(data, axis=0) if window_size == len(data) else []
    for i in range(window_size, len(data), stride):
        rtn.append(data[i - window_size:i])
    return np.array(rtn)


def modify_indice_to_cover(i1, i2, coverage, tolerance=3):
    assert i1 < i2
    assert abs(coverage - (i2 - i1)) <= tolerance
    is_modifying_i1 = True
    if i2 - i1 > coverage:
        while i2 - i1 != coverage:
            if is_modifying_i1:
                i1 += 1
            else:
                i2 -= 1
        print('Modified')

    elif i2 - i1 < coverage:
        while i2 - i1 != coverage:
            if is_modifying_i1:
                i1 -= 1
            else:
                i2 += 1
        print('Modified')

    return i1, i2



def interp_negative(y):
    idx = y < 0
    x = np.arange(len(y))
    y_interp = np.copy(y)
    y_interp[idx] = np.interp(x[idx], x[~idx], y[~idx])
    return y_interp


def clutter_removal(cur_frame, clutter, signal_clutter_ratio):
    if clutter is None:
        clutter = cur_frame
    else:
        clutter = signal_clutter_ratio * clutter + (1 - signal_clutter_ratio) * cur_frame
    return cur_frame - clutter, clutter


def integer_one_hot(a, num_classes):
    a = a.astype(int)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).astype(int)


def corrupt_frame_padding(time_series_data, min_threshold=np.NINF, max_threshold=np.PINF, frame_channel_first=True):
    if not frame_channel_first:
        time_series_data = np.moveaxis(time_series_data, -1, 0)

    if np.min(time_series_data[0]) < min_threshold or np.max(time_series_data[0]) > max_threshold:
        print('error: first frame is broken')
        return

    if np.min(time_series_data[-1]) < min_threshold or np.max(time_series_data[-1]) > max_threshold:
        print('error: last frame is broken')
        return

    broken_frame_counter = 0

    # check first and last frame
    for frame_index in range(1, len(time_series_data) - 1):
        data = np.squeeze(time_series_data[frame_index], axis=-1)
        if np.min(time_series_data[frame_index]) < min_threshold or np.max(
                time_series_data[frame_index]) > max_threshold:
            # find broken frame, padding with frame +1 and frame -1
            broken_frame_before = time_series_data[frame_index - 1]
            broken_frame = time_series_data[frame_index]
            broken_frame_next = time_series_data[frame_index + 1]
            if np.min(time_series_data[frame_index + 1]) >= min_threshold and np.max(
                    time_series_data[frame_index + 1]) < max_threshold:
                time_series_data[frame_index] = (time_series_data[frame_index - 1] + time_series_data[
                    frame_index + 1]) * 0.5
                broken_frame_counter += 1
                print('find broken frame at index:', frame_index, ' interpolate by the frame before and after.')
            else:
                time_series_data[frame_index] = time_series_data[frame_index - 1]
                print('find two continues broken frames at index: ', frame_index, ', equalize with previous frame.')

    if not frame_channel_first:
        time_series_data = np.moveaxis(time_series_data, 0, -1)

    print('pad broken frame: ', broken_frame_counter)
    return time_series_data


def time_series_static_clutter_removal(time_series_data, init_clutter=None, signal_clutter_ratio=0.1,
                                       frame_channel_first=True):
    if not frame_channel_first:
        time_series_data = np.moveaxis(time_series_data, -1, 0)

    clutter = None
    if init_clutter:
        clutter = init_clutter
    else:  # using first two frames as the init_clutter
        clutter = (time_series_data[0] + time_series_data[1]) * 0.5

    for frame_index in range(0, len(time_series_data)):
        clutter_removal_frame, clutter = clutter_removal(
            cur_frame=time_series_data[frame_index],
            clutter=clutter,
            signal_clutter_ratio=signal_clutter_ratio)

        time_series_data[frame_index] = clutter_removal_frame

    if not frame_channel_first:
        time_series_data = np.moveaxis(time_series_data, 0, -1)

    return time_series_data

def is_broken_frame(frame, min_threshold=np.NINF, max_threshold=np.PINF):
    if np.min(frame) < min_threshold or np.max(frame) > max_threshold:
        return True
    else:
        return False


def levenshtein_ratio_and_distance(s, t, ratio_calc=False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                     distance[row][col - 1] + 1,  # Cost of insertions
                                     distance[row - 1][col - 1] + cost)  # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])

def replace_special(target_str: str, replacement_dict):
    for special, replacement in replacement_dict.items():
        # print('replacing ' + special)
        target_str = target_str.replace(special, replacement)
    return target_str