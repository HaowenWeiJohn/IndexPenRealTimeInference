from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
import datetime


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]

        self.graphWidget.setBackground('w')

        pen = pg.mkPen(color=(255, 0, 0))

        month_labels = [
            # Generate a list of tuples (x_value, x_label)
            (m, datetime.date(2020, m, 1).strftime('%B'))
            for m in months
        ]
        self.graphWidget.plot(months, temperature, pen=pen, labels=month_labels)

        ax = self.graphWidget.getAxis('bottom')
        # Pass the list in, *in* a list.
        ax.setTicks([month_labels])
        print(month_labels)


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()