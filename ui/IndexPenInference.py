# This Python file uses the following encoding: utf-8
import pickle

from PyQt5 import QtCore, uic
from PyQt5 import QtWidgets

import numpy as np
from datetime import datetime

from utils.ui_utils import *
from config import config_path
from utils.sound import *


class IndexPenInference(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()

        self.inference_state = 'idle'

        self.ui = uic.loadUi("ui/IndexPenInference.ui", self)


        # Left objectName: indexpeninference_display_Widget layoutName: indexpeninference_display_vertical_layout
        self.indexpeninference_display_container, self.indexpeninference_display_container = init_container \
            (parent=self.indexpeninference_display_vertical_layout, vertical=True, label='IndexPen Real Time Inference')
        self.indexpeninference_text_layout, self.indexpeninference_text_input = init_inputBox(parent=self.indexpeninference_display_container,
                                                                      label='IndexPen Inference Text Input :',
                                                                      default_input=config_ui.mmWave_lsl_outlet_name_default,
                                                                      vertical=True)




        # Right objectName: indexpeninference_control_Widget layoutName: indexpeninference_control_vertical_layout
        # control pannel vertical layout
        self.indexpeninference_control_container, self.indexpeninference_control_layout = init_container \
            (parent=self.indexpeninference_control_vertical_layout, vertical=True, label='IndexPen Inference LSL Connection Control Panel')
        # check box hide realtime inference plot

        self.indexpeninference_lslconnection_btns_container, self.indexpeninference_lslconnection_btns_layout = init_container \
            (parent=self.indexpeninference_control_vertical_layout, vertical=True, label='IndexPen Inference LSL control')
        self.label_list_layout, self.label_list_input = init_inputBox(parent=self.indexpeninference_lslconnection_btns_layout,
                                                                      label='IndexPen mmWave LSL Outlet :',
                                                                      default_input=config_ui.mmWave_lsl_outlet_name_default)

        self.connect_lsl = init_button(parent=self.indexpeninference_lslconnection_btns_layout, label='Start testing')




        self.indexpeninference_plot_checkbox, self.indexpeninference_plot_checkbox = init_checkBox(
            parent=self.indexpeninference_control_layout, label='Inference Plot : ', default_checked=False)

