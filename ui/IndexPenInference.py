# This Python file uses the following encoding: utf-8
import pickle
from collections import deque

from PyQt5 import QtCore, uic
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from pyautogui import typewrite

from pynput.keyboard import Key, Controller
import numpy as np
from datetime import datetime
import pyqtgraph as pg
from pyqtgraph import PlotDataItem

from interfaces.LSLInletInterface import LSLInletInterface
from threadings.workers import MmWaveLSLInletInferenceWorker
from threadings import workers
from utils.data_utils import levenshtein_ratio_and_distance

from utils.ui_utils import *
from config import config_path, config_signal
from config import config
from utils.sound import *

import datetime
import glob
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

import sys


class IndexPenInference(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()

        self.inference_state = 'idle'
        self.indexpen_activated = False

        self.ui = uic.loadUi("ui/IndexPenInference.ui", self)

        # Left objectName: indexpeninference_display_Widget layoutName: indexpeninference_display_vertical_layout
        self.indexpeninference_display_container, self.indexpeninference_display_layout = init_container \
            (parent=self.indexpeninference_display_vertical_layout, vertical=True, label='IndexPen Real Time Inference')

        self.levenshtein_distance_ration_label = QLabel(text='Levenshtein Distance Ratio:')
        self.indexpeninference_display_layout.addWidget(self.levenshtein_distance_ration_label)
        self.levenshtein_distance_ration_label.setAlignment(QtCore.Qt.AlignCenter)
        self.levenshtein_distance_ration_label.adjustSize()

        self.indexpeninference_text_layout, self.indexpeninference_text_input = init_textEditInputBox(
            parent=self.indexpeninference_display_layout,
            label='IndexPen Inference Text Input :',
            default_input=config_ui.indexPen_text_input_default,
            vertical=True)

        # task list comb_box
        self.task_dict = generate_sentence_task()
        self.task_dict.insert(0, config_ui.indexPen_classes_default)
        self.task_dict_combbox = self.task_dict.copy()
        for i in range(0, len(self.task_dict_combbox)):
            self.task_dict_combbox[i] =  ''.join((str(i), '. ', self.task_dict_combbox[i]))


        self.task_combo_box = init_combo_box(parent=self.indexpeninference_display_layout, label=None,
                                          item_list=self.task_dict_combbox)


        self.calculate_levenshtein_ratio_and_distance_btn = init_button(parent=self.indexpeninference_display_layout,
                                                   label='Calculate Levenshtein', function=self.calculate_levenshtein_ratio_and_distance_btn_clicked)





        # plotting container
        self.indexpeninference_plot_container, self.indexpeninference_plot_layout = init_container \
            (parent=self.indexpeninference_display_layout, vertical=True, label='IndexPen Inference Visualization')

        # init inference plot block
        self.fs_label = QLabel(text='Prediction rate = ')
        self.ts_label = QLabel(text='Current Time Stamp = ')
        self.indexpeninference_plot_layout.addWidget(self.fs_label)
        self.indexpeninference_plot_layout.addWidget(self.ts_label)

        ########################################################
        plot_widget = pg.PlotWidget()
        plot_widget.setLimits(xMin=0, xMax=21, yMin=0, yMax=1.2)
        distinct_colors = get_distinct_colors(len(config_signal.indexpen_classes))
        plot_widget.addLegend()
        self.plots = [plot_widget.plot([], [], pen=pg.mkPen(color=color), name=c_name) for color, c_name in
                      zip(distinct_colors, config_signal.indexpen_classes)]
        [p.setDownsampling(auto=True, method='mean') for p in self.plots if p == PlotDataItem]
        [p.setClipToView(clip=True) for p in self.plots for p in self.plots if p == PlotDataItem]

        self.indexpeninference_plot_layout.addWidget(plot_widget)

        #########################################################
        barchart_widget = pg.PlotWidget()
        barchart_widget.setLimits(xMin=0, xMax=len(config_signal.indexpen_classes), yMin=0, yMax=1.1)
        label_x_axis = barchart_widget.getAxis('bottom')
        label_dict = dict(enumerate(config_signal.indexpen_classes)).items()
        label_x_axis.setTicks([label_dict])
        y1 = np.array([0] * len(config_signal.indexpen_classes))
        # create horizontal list
        x = np.arange(len(config_signal.indexpen_classes))
        self.indexpeninference_prob_bars = pg.BarGraphItem(x=x, height=y1, width=0.6, brush='r')
        barchart_widget.addItem(self.indexpeninference_prob_bars)
        self.indexpeninference_plot_layout.addWidget(barchart_widget)

        # reset inference text box button
        self.reset_indexpen_text_input_button = init_button(parent=self.indexpeninference_display_layout,
                                                   label='Reset Text Box', function=self.reset_indexpen_text_input_button_clicked)

        ###########################################################
        ###########################################################

        # Right objectName: indexpeninference_control_Widget layoutName: indexpeninference_control_vertical_layout
        # control pannel vertical layout
        self.indexpeninference_control_container, self.indexpeninference_control_layout = init_container \
            (parent=self.indexpeninference_control_vertical_layout, vertical=True,
             label='IndexPen Inference LSL Connection Control Panel')
        # check box hide realtime inference plot

        self.indexpeninference_modelselection_container, self.indexpeninference_modelselection_layout = init_container \
            (parent=self.indexpeninference_control_layout, vertical=True, label='IndexPen Inference Model')
        self.indexpeninference_modelpath_layout, self.indexpeninference_modelpath_input = init_inputBox(
            parent=self.indexpeninference_modelselection_layout,
            label='IndexPen Model Path :',
            default_input=config_path.indexpen_model_path)
        self.load_indexpen_model_btn = init_button(parent=self.indexpeninference_modelselection_layout,
                                                   label='Load IndexPen Model')

        self.indexpeninference_lslconnection_control_container, self.indexpeninference_lslconnection_control_layout = init_container \
            (parent=self.indexpeninference_control_layout, vertical=True, label='IndexPen Inference LSL control')
        self.indexpeninference_lslname_layout, self.indexpeninference_lslname_input = init_inputBox(
            parent=self.indexpeninference_lslconnection_control_layout,
            label='IndexPen mmWave LSL Outlet Name :',
            default_input=config_ui.mmWave_lsl_outlet_name_default)
        self.connect_mmwave_lsl_btn = init_button(parent=self.indexpeninference_lslconnection_control_layout,
                                                  label='Connect mmWave LSL')

        self.indexpeninference_plot_checkbox, self.indexpeninference_plot_checkbox = init_checkBox(
            parent=self.indexpeninference_control_layout, label='Inference Plot Hidden: ', default_checked=False)

        self.indexpeninference_start_stop_btn_container, self.indexpeninference_start_stop_btn_layout = init_container \
            (parent=self.indexpeninference_control_layout, vertical=False, label='')

        self.indexpeninference_start_btn = init_button(parent=self.indexpeninference_start_stop_btn_layout,
                                                       label='Start Inference')
        self.indexpeninference_stop_btn = init_button(parent=self.indexpeninference_start_stop_btn_layout,
                                                      label='Stop Inference')

        # button clicked
        self.load_indexpen_model_btn.clicked.connect(self.load_indexpen_model_btn_clicked)
        self.connect_mmwave_lsl_btn.clicked.connect(self.connect_mmwave_lsl_btn_clicked)
        self.indexpeninference_plot_checkbox.stateChanged.connect(self.indexpeninference_plot_checkbox_stateChange)
        self.indexpeninference_start_btn.clicked.connect(self.indexpeninference_start_btn_clicked)
        self.indexpeninference_stop_btn.clicked.connect(self.indexpeninference_stop_btn_clicked)

        self.interpreter = None
        self.mmWave_lsl_interface = None
        self.mmWave_inference_worker = {}
        self.worker_threads = {}

        # prediction parameters
        self.debouncer = np.zeros(31)
        self.relaxCounter = 0
        self.pred_prob_hist_buffer = deque(maxlen=config_signal.mmWave_fps * 20)
        self.pred_ts_hist_buffer = deque(maxlen=config_signal.mmWave_fps * 20)
        # # workers
        # self.worker_threads = {}
        # self.lsl_inference_workers = {}
        self.keyboard = Controller()

        # timer
        self.timer = QTimer()
        self.timer.setInterval(config.REFRESH_INTERVAL)  # for 1000 Hz refresh rate
        self.timer.timeout.connect(self.tick)
        self.timer.start()

        # # visualization timer
        # self.v_timer = QTimer()
        # self.v_timer.setInterval(config.VISUALIZATION_REFRESH_INTERVAL)  # for 15 Hz refresh rate
        # self.v_timer.timeout.connect(self.visualize_inference_result)
        # self.v_timer.start()

    def load_indexpen_model_btn_clicked(self):

        print('connect_mmwave_lsl_btn clicked')
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.indexpeninference_modelpath_input.text())
            self.interpreter.allocate_tensors()

            # interpreter.set_tensor(input1_index, np.expand_dims(np.array(X_mmw_rD_test[0]), axis=0).astype(np.float32))
            # interpreter.set_tensor(input2_index, np.expand_dims(np.array(X_mmw_rA_test[0]), axis=0).astype(np.float32))
            #
            # interpreter.invoke()
            # predictions = interpreter.get_tensor(output_index)
            #
            print('Successfully load IndexPen Model: ', self.indexpeninference_modelpath_input.text())
        except:
            print('File does not exist')
            raise ValueError

    def connect_mmwave_lsl_btn_clicked(self):
        try:
            if self.interpreter is None:
                raise AssertionError('Please Load the inference model(interpreter) first')
        except AssertionError as e:
            dialog_popup(str(e))
            return None

        # create mmWave lsl worker
        lsl_stream_name = self.indexpeninference_lslname_input.text()
        print(lsl_stream_name)
        try:
            self.mmWave_lsl_interface = LSLInletInterface(lsl_stream_name)
        except AttributeError:
            print('Cannot find LSL name')
            dialog_popup('Unable to find LSL Stream with given type {0}.'.format(lsl_stream_name))
            return None
        self.mmWave_inference_worker[lsl_stream_name] = workers.MmWaveLSLInletInferenceWorker(self.mmWave_lsl_interface,
                                                                                              indexpen_interpreter=self.interpreter)
        self.mmWave_inference_worker[lsl_stream_name].signal_data.connect(self.process_prediction_visualization)
        worker_thread = pg.QtCore.QThread(self)
        self.worker_threads[lsl_stream_name] = worker_thread
        self.mmWave_inference_worker[lsl_stream_name].moveToThread(self.worker_threads[lsl_stream_name])
        worker_thread.start()
        print('Inference worker created')

    def indexpeninference_plot_checkbox_stateChange(self):
        if self.indexpeninference_plot_checkbox.isChecked():
            self.indexpeninference_plot_container.hide()
        else:
            self.indexpeninference_plot_container.show()

    def indexpeninference_start_btn_clicked(self):
        print('indexpeninference_start_btn_clicked')
        try:
            self.mmWave_inference_worker[self.mmWave_lsl_interface.lsl_data_type].start_stream()
        except AttributeError:
            dialog_popup('please load the model and connect to lsl stream first')

    def indexpeninference_stop_btn_clicked(self):
        print('indexpeninference_stop_btn_clicked')

    def tick(self):
        """
        ticks every 'refresh' milliseconds
        """
        # pass
        [w.tick_signal.emit() for w in self.mmWave_inference_worker.values()]

    def process_prediction_visualization(self, data_dict):
        prediction_result = data_dict['prediction_result']
        prediction_timestamp = data_dict['prediction_timestamp']
        prediction_rate = data_dict['prediction_rate']
        self.pred_prob_hist_buffer.append(prediction_result)
        self.pred_ts_hist_buffer.append(prediction_timestamp)

        if self.relaxCounter == config_signal.relaxPeriod:
            breakIndices = np.argwhere(prediction_result >= config_signal.debouncerProbThreshold)

            for i, debouncer_value in enumerate(self.debouncer):
                if i in breakIndices:
                    self.debouncer[i] += 1
                else:
                    if self.debouncer[i] > 0:
                        self.debouncer[i] -= 1

            detects = np.argwhere(np.array(self.debouncer) >= config_signal.debouncerFrameThreshold)
            if len(detects) > 0:
                print(detects)
                detect_char = config_signal.indexpen_classes[detects[0][0]]
                print(detect_char)
                self.debouncer = np.zeros(31)
                self.relaxCounter = 0

                # GUI output char update invoke text input
                if detect_char == 'Nois':
                    print()
                    # self.keyboard.press(Key.enter)
                    # self.keyboard.release(Key.enter)
                elif detect_char == 'Act':
                    # toggle indexpen
                    self.indexpen_activated = not self.indexpen_activated
                    print('Activation: ', self.indexpen_activated)
                    typewrite('*')
                    # if self.indexpen_activated is True:
                    #     dih()
                    # else:
                    #     dah()

                    # self.keyboard.press(Key.enter)
                    # self.keyboard.release(Key.enter)
                elif detects == 'Ent':
                    typewrite('%')
                    # self.keyboard.press(Key.enter)
                    # self.keyboard.release(Key.enter)
                    # dih()

                elif detect_char == 'Spc':
                    self.keyboard.press(Key.space)
                    self.keyboard.release(Key.space)
                elif detect_char == 'Bspc':
                    self.keyboard.press(Key.backspace)
                    self.keyboard.release(Key.backspace)
                else:
                    self.keyboard.press(detect_char)
                    self.keyboard.release(detect_char)


        else:
            self.relaxCounter += 1

        # plot real time

        self.fs_label.setText('Prediction rate = ' + str(prediction_rate))
        self.ts_label.setText('Current Time Stamp = ' + str(prediction_timestamp))
        time_vector = np.array(self.pred_ts_hist_buffer) - self.pred_ts_hist_buffer[0]
        prediction_hist = np.array(self.pred_prob_hist_buffer)

        [plot.setData(time_vector, prediction_hist[:, i]) for i, plot in
         enumerate(self.plots)]

        self.indexpeninference_prob_bars.setOpts(x=np.arange(len(config_signal.indexpen_classes)),
                                                 height=prediction_result, width=0.6, brush='r')
        # print(np.arange(len(config_signal.indexpen_classes)), prediction_result)

        return None

    def reset_indexpen_text_input_button_clicked(self):
        self.indexpeninference_text_input.setText(config_ui.indexPen_text_input_default)
        self.indexpen_activated = False

    def calculate_levenshtein_ratio_and_distance_btn_clicked(self):
        true_string = self.task_dict[self.task_combo_box.currentIndex()]
        input_string = self.indexpeninference_text_input.toPlainText()
        print(true_string)
        print(input_string)
        levenshtein_ratio = levenshtein_ratio_and_distance(true_string, input_string, ratio_calc=True)
        print(levenshtein_ratio)
        levenshtein_distance = levenshtein_ratio_and_distance(true_string, input_string, ratio_calc=False)
        print(levenshtein_distance)
        self.levenshtein_distance_ration_label.setText('Levenshtein Distance Ratio: '+ str(levenshtein_ratio))
