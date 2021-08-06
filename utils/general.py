import json
import os

import brainflow
import numpy as np




def slice_len_for(slc, seqlen):
    start, stop, step = slc.indices(seqlen)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def load_all_lslStream_presets(lsl_preset_roots='Presets/LSLPresets'):
    preset_file_names = os.listdir(lsl_preset_roots)
    preset_file_paths = [os.path.join(lsl_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        preset_dict = load_LSL_preset(loaded_preset_dict)
        stream_name = preset_dict['StreamName']
        presets[stream_name] = preset_dict
    return presets


def load_all_Device_presets(device_preset_roots='Presets/DevicePresets'):
    preset_file_names = os.listdir(device_preset_roots)
    preset_file_paths = [os.path.join(device_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        preset_dict = load_LSL_preset(loaded_preset_dict)
        stream_name = preset_dict['StreamName']
        presets[stream_name] = preset_dict
    return presets


def load_all_experiment_presets(exp_preset_roots='Presets/ExperimentPresets'):
    preset_file_names = os.listdir(exp_preset_roots)
    preset_file_paths = [os.path.join(exp_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        presets[loaded_preset_dict['ExperimentName']] = loaded_preset_dict['PresetStreamNames']
    return presets


def load_LSL_preset(preset_dict):
    if 'ChannelNames' not in preset_dict.keys():
        preset_dict['ChannelNames'] = None
    if 'GroupChannelsInPlot' not in preset_dict.keys():
        preset_dict['GroupChannelsInPlot'] = None
        preset_dict['PlotGroupSlices'] = None
    if 'NominalSamplingRate' not in preset_dict.keys():
        preset_dict['NominalSamplingRate'] = None
    return preset_dict


def create_LSL_preset(stream_name, channel_names=None, plot_group_slices=None):
    preset_dict = {'StreamName': stream_name, 'ChannelNames': channel_names, 'PlotGroupSlices': plot_group_slices}
    preset_dict = load_LSL_preset(preset_dict)
    return preset_dict


def process_LSL_plot_group(preset_dict):
    preset_dict["PlotGroupSlices"] = []
    head = 0
    for x in preset_dict['GroupChannelsInPlot']:
        preset_dict["PlotGroupSlices"].append((head, x))
        head = x
    if head != preset_dict['NumChannels']:
        preset_dict["PlotGroupSlices"].append(
            (head, preset_dict['NumChannels']))  # append the last group
    return preset_dict




