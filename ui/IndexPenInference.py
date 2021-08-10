# This Python file uses the following encoding: utf-8
import pickle

from PyQt5 import QtCore, uic
from PyQt5 import QtWidgets

import numpy as np
from datetime import datetime
import pyqtgraph as pg
from utils.ui_utils import *
from config import config_path
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

        self.ui = uic.loadUi("ui/IndexPenInference.ui", self)

        # Left objectName: indexpeninference_display_Widget layoutName: indexpeninference_display_vertical_layout
        self.indexpeninference_display_container, self.indexpeninference_display_layout = init_container \
            (parent=self.indexpeninference_display_vertical_layout, vertical=True, label='IndexPen Real Time Inference')

        self.most_recent_detection_label = QLabel(text='Most Recent Detection:')
        self.indexpeninference_display_layout.addWidget(self.most_recent_detection_label)
        self.most_recent_detection_label.setAlignment(QtCore.Qt.AlignCenter)
        self.most_recent_detection_label.adjustSize()

        self.indexpeninference_text_layout, self.indexpeninference_text_input = init_textEditInputBox(
            parent=self.indexpeninference_display_layout,
            label='IndexPen Inference Text Input :',
            default_input=config_ui.indexPen_classes_default,
            vertical=True)

        # plotting container
        self.indexpeninference_plot_container, self.indexpeninference_plot_layout = init_container \
            (parent=self.indexpeninference_display_layout, vertical=True, label='IndexPen Inference argmax plot')
        # init inference plot block
        fs_label = QLabel(text='Sampling rate = ')
        ts_label = QLabel(text='Current Time Stamp = ')
        self.indexpeninference_plot_layout.addWidget(fs_label)
        self.indexpeninference_plot_layout.addWidget(ts_label)
        plot_widget = pg.PlotWidget()
        self.indexpeninference_plot_layout.addWidget(plot_widget)

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

        # button clicked
        self.load_indexpen_model_btn.clicked.connect(self.load_indexpen_model_btn_clicked)
        self.connect_mmwave_lsl_btn.clicked.connect(self.connect_mmwave_lsl_btn_clicked)

        self.interpreter = None

    def load_indexpen_model_btn_clicked(self):

        print('connect_mmwave_lsl_btn clicked')
        try:
            self.interpreter = tf.lite.Interpreter(model_path= self.indexpeninference_modelpath_input.text())
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
        print('load_indexpen_model_btn clicked')
        return
