from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox
from config import config

from ui.SettingsTab import SettingsTab
from ui.IndexPenInference import IndexPenInference

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, inference_interface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = uic.loadUi("ui/mainwindow.ui", self)
        self.setWindowTitle('Reality Navigation RealTime Prediction')


        self.settingTab = SettingsTab(self)
        self.settings_tab_vertical_layout.addWidget(self.settingTab)

        # SetingTab
        self.indexpenInference = IndexPenInference(self)
        self.indexpen_tab_horizontal_layout.addWidget(self.indexpenInference)



        self.app = app

        # # workers
        # self.worker_threads = {}
        # self.lsl_inference_workers = {}
        #
        # # data buffer
        # self.LSL_plots_fs_label_dict = {}
        #
        #
        # if inference_interface:
        #     self.inference_interface = inference_interface
        #
        # # timer
        # self.timer = QTimer()
        # self.timer.setInterval(config.REFRESH_INTERVAL)  # for 1000 Hz refresh rate
        # self.timer.timeout.connect(self.ticks)
        # self.timer.start()
        #
        # # visualization timer
        # self.v_timer = QTimer()
        # self.v_timer.setInterval(config.VISUALIZATION_REFRESH_INTERVAL)  # for 15 Hz refresh rate
        # self.v_timer.timeout.connect(self.visualize_inference_result)
        # self.v_timer.start()







    # def init_inference_lsl(self):
    #     data_lsl_stream_name = 'indexpen'
    #


    # def ticks(self):
    #     pass
    #
    # def visualize_inference_result(self):
    #     pass


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Exit Application?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:

            # stop inference

            event.accept()
            self.app.quit()
        else:
            event.ignore()