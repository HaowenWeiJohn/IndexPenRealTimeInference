import time
from collections import deque

import cv2
import pyqtgraph as pg
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal
from pylsl import local_clock
from config import config_ui
from interfaces.LSLInletInterface import LSLInletInterface
from utils.data_utils import is_broken_frame, clutter_removal
from utils.sim import sim_openBCI_eeg, sim_unityLSL, sim_inference, sim_imp, sim_heatmap, sim_detected_points

import pyautogui

import numpy as np

from utils.ui_utils import dialog_popup


class MmWaveLSLInletInferenceWorker(QObject):
    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    tick_signal = pyqtSignal()

    def __init__(self, LSLInlet_interface: LSLInletInterface, indexpen_interpreter, *args, **kwargs):
        super(MmWaveLSLInletInferenceWorker, self).__init__()
        self.tick_signal.connect(self.process_on_tick)

        self._lslInlet_interface = LSLInlet_interface

        self._indexpen_interpreter = indexpen_interpreter
        # self._indexpen_interpreter.allocate_tensors()

        self.IndexPenRealTimePreprocessor = IndexPenRealTimePreprocessor()

        self.is_streaming = False

        self.start_time = time.time()
        self.num_samples = 0

        self.rd_hist_buffer = deque(maxlen=120)
        self.ra_hist_buffer = deque(maxlen=120)

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            frames, timestamps = self._lslInlet_interface.process_frames()  # get all data and remove it from internal buffer

            self.num_samples += len(timestamps)
            try:
                sampling_rate = self.num_samples / (time.time() - self.start_time) if self.num_samples > 0 else 0
            except ZeroDivisionError:
                sampling_rate = 0

################### preprocessing
            for frame in frames:
                current_rd = np.array(frame[0:128]).reshape((8,16))
                current_ra = np.array(frame[128:640]).reshape((8,64))
                rd_cr, ra_cr = self.IndexPenRealTimePreprocessor.data_preprocessing(current_rd=current_rd,
                                                                                    current_ra=current_ra)
                if rd_cr is not None and ra_cr is not None:
                    # append to hist buffer
                    self.rd_hist_buffer.append(rd_cr)
                    self.ra_hist_buffer.append(ra_cr)

            if self.ra_hist_buffer.__len__() == self.ra_hist_buffer.maxlen:
                prediction_data_set =




            else:



            data_dict = {'prediction result'}

            self.signal_data.emit(data_dict)

    def start_stream(self):
        try:
            self._lslInlet_interface.start_sensor()
        except AttributeError as e:
            dialog_popup(e)
            return
        self.is_streaming = True

        self.num_samples = 0
        self.start_time = time.time()

    def stop_stream(self):
        self._lslInlet_interface.stop_sensor()
        self.is_streaming = False


class IndexPenRealTimePreprocessor:
    def __init__(self, data_buffer_len=3, rd_cr_ratio=0.8, ra_cr_ratio=0.8, rd_threshold=(-1000, 1500),
                 ra_threshold=(0, 2500)):
        self.data_buffer_len = data_buffer_len
        self.rd_buffer = deque(maxlen=data_buffer_len)
        self.ra_buffer = deque(maxlen=data_buffer_len)
        self.rd_cr_ratio = rd_cr_ratio
        self.ra_cr_ratio = ra_cr_ratio

        self.rd_threshold = rd_threshold
        self.ra_threshold = ra_threshold

        self.rd_clutter = None
        self.ra_clutter = None

    def data_preprocessing(self, current_rd, current_ra):
        # check index 1 data is corrupt or not
        self.rd_buffer.append(current_rd)
        self.ra_buffer.append(current_ra)
        if len(self.rd_buffer) == self.data_buffer_len:

            # corrupt frame removal
            self.rd_buffer = self.data_padding(self.rd_buffer, threshold=self.rd_threshold)
            self.ra_buffer = self.data_padding(self.ra_buffer, threshold=self.ra_threshold)

            # return index 0 data with clutter removal
            rd_cr_frame, self.rd_clutter = clutter_removal(self.rd_buffer[0], self.rd_clutter, self.rd_cr_ratio)
            ra_cr_frame, self.ra_clutter = clutter_removal(self.ra_buffer[0], self.ra_clutter, self.ra_cr_ratio)

            return rd_cr_frame, ra_cr_frame
        else:
            return None, None

    def data_padding(self, data_buffer, threshold):
        if is_broken_frame(data_buffer[1], min_threshold=threshold[0], max_threshold=threshold[1]) \
                and not is_broken_frame(data_buffer[2], min_threshold=threshold[0], max_threshold=threshold[1]):
            data_buffer[1] = (data_buffer[0] + data_buffer[2]) * 0.5
            print('broken frame pad with frame before and after')
        elif is_broken_frame(data_buffer[1], min_threshold=threshold[0], max_threshold=threshold[1]) \
                and is_broken_frame(data_buffer[2], min_threshold=threshold[0], max_threshold=threshold[1]):
            data_buffer[1] = data_buffer[0]
            print('two continuous borken frame, equalize with previous one')

        return data_buffer

