import time
import cv2
import pyqtgraph as pg
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal
from pylsl import local_clock
from config import config_ui
from interfaces.LSLInletInterface import LSLInletInterface
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

        self.is_streaming = False

        self.start_time = time.time()
        self.num_samples = 0

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            frames, timestamps = self._lslInlet_interface.process_frames()  # get all data and remove it from internal buffer

            self.num_samples += len(timestamps)
            try:
                sampling_rate = self.num_samples / (time.time() - self.start_time) if self.num_samples > 0 else 0
            except ZeroDivisionError:
                sampling_rate = 0
            data_dict = {'lsl_data_type': self._lslInlet_interface.lsl_data_type, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            print(data_dict)
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