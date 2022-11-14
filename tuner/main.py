import sys

#import os
#os.environ["QT_DEBUG_PLUGINS"] = "1"

import PyQt5.QtWidgets as wdg
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import time

import threading

import soundcard as sc
import util
import logging

logging.basicConfig(level=logging.DEBUG)

BG_FAIL_COLOR = "rgb(255, 100, 100)"


def round_to_sig_digets(x, n=3):
    f = 10 ** (n - int(np.log10(x)))
    x = round(x * f)
    return x / f


class App(wdg.QMainWindow):
    def __init__(self):

        # default values for the fundamental parameters
        self.ref_freq = 440
        self.key = "c"
        self.level = 0
        self.num_cyc = 15
        self.smpl_per_cyc = 12.3

        # some constants for the recoding buffer
        self.block_size = 512
        self.num_frames = 128

        # calculated on calling calculate_parameters
        self.rec_length = None
        self.sample_rate = None
        self.target_freq = None

        self.calculate_parameters()

        super().__init__()
        self.title = "Tuner"
        self.initUI()

        self.update_targetFreq()

        self.refFreqEdit.editingFinished.connect(self.update_targetFreq)
        self.keyEdit.editingFinished.connect(self.update_targetFreq)
        self.levelEdit.editingFinished.connect(self.update_targetFreq)

        self.numCycEdit.editingFinished.connect(self.update_num_cyc)
        self.smpl_per_cycEdit.editingFinished.connect(self.update_smpl_per_cyc)


        self.nextNoteBtn.clicked.connect(self.nextTargetFreq)
        self.prevNoteBtn.clicked.connect(self.prevTargetFreq)

    def calculate_parameters(self):
        """
        calculate parameters that follow from the fundamental parameters.

        based on the class properties

            self.refFreq
            self.key
            self.level
            self.num_cyc
            self.smpl_per_cyc

        calculate

            self.recLength
            self.samplerate
            self.targetFreq
        """
        self.target_freq = util.note_to_freq(
            note=self.key, level=self.level, freq_a=self.ref_freq
        )
        self.rec_length = self.num_cyc / self.target_freq
        self.sample_rate = int(self.smpl_per_cyc * self.target_freq)



    def initUI(self):

        ###########################
        #   Window
        ###########################
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, 1600, 1000)

        ###########################
        #   recording / sampling
        ###########################
        # settings on the right
        self.settingsGroup = wdg.QGroupBox(title="recording / sampling")
        self.settingsGroup.setMinimumWidth(300)
        self.settingsGroup.setMaximumWidth(300)
        self.settingsLayout = wdg.QVBoxLayout()

        # choose mic Label
        self.micDevLabel = wdg.QLabel("choose microphone")
        self.settingsLayout.addWidget(self.micDevLabel)

        # choose mic Combo
        self.micDevDropDown = wdg.QComboBox()
        all_mics = [str(d) for d in sc.all_microphones()]
        for d in all_mics:
            self.micDevDropDown.addItem(d)
        self.currentMic = sc.default_microphone()
        i = all_mics.index(str(self.currentMic))
        self.micDevDropDown.setCurrentIndex(i)
        self.settingsLayout.addWidget(self.micDevDropDown)

        # refFreq Label
        self.refFreqLabel = wdg.QLabel("reference freq. for 'a' (in Hz)")
        self.settingsLayout.addWidget(self.refFreqLabel)

        # refFreqLabel edit
        self.refFreqEdit = wdg.QLineEdit(str(self.ref_freq))
        self.settingsLayout.addWidget(self.refFreqEdit)

        # rec length Label
        self.recLegthLabel = wdg.QLabel("recoding length")
        self.settingsLayout.addWidget(self.recLegthLabel)

        # rec length Value Label
        self.recLengthValueLabel = wdg.QLabel()
        self.settingsLayout.addWidget(self.recLengthValueLabel)

        # samplerate Label
        self.samplerateLabel = wdg.QLabel("sampling rate")
        self.settingsLayout.addWidget(self.samplerateLabel)

        # samplerate Value edit
        self.samplerateValueLabel = wdg.QLabel("")
        self.settingsLayout.addWidget(self.samplerateValueLabel)

        # num_cyc Label
        self.numCycLabel = wdg.QLabel("number of cycles (wrt to target freq.)")
        self.settingsLayout.addWidget(self.numCycLabel)

        # num_cyc edit
        self.numCycEdit = wdg.QLineEdit(str(self.num_cyc))
        self.settingsLayout.addWidget(self.numCycEdit)

        # smpl_per_cyc Label
        self.smpl_per_cycLabel = wdg.QLabel("samples per cycles")
        self.settingsLayout.addWidget(self.smpl_per_cycLabel)

        # smpl_per_cyc edit
        self.smpl_per_cycEdit = wdg.QLineEdit(str(self.smpl_per_cyc))
        self.settingsLayout.addWidget(self.smpl_per_cycEdit)

        # put everything together
        self.settingsLayout.addStretch(1)
        self.settingsGroup.setLayout(self.settingsLayout)

        ###########################
        #   key selection
        ###########################

        # key selection
        self.keySelGroup = wdg.QGroupBox(title="select key")
        self.keySelGroup.setMinimumWidth(300)
        self.keySelGroup.setMaximumWidth(300)
        self.keySelLayout = wdg.QVBoxLayout()

        # key Label
        self.keyLabel = wdg.QLabel("key")
        self.keySelLayout.addWidget(self.keyLabel)

        # key edit
        self.keyEdit = wdg.QLineEdit(self.key)
        self.keySelLayout.addWidget(self.keyEdit)

        # level Label
        self.levelLabel = wdg.QLabel("level")
        self.keySelLayout.addWidget(self.levelLabel)

        # level edit
        self.levelEdit = wdg.QLineEdit(str(self.level))
        self.keySelLayout.addWidget(self.levelEdit)

        # target freq Label
        self.targetFreqLabel = wdg.QLabel("target frequency")
        self.keySelLayout.addWidget(self.targetFreqLabel)

        # target freq Label
        self.targetFreqShow = wdg.QLabel()
        self.keySelLayout.addWidget(self.targetFreqShow)

        self.prevNoteBtn = wdg.QPushButton("prev")
        self.prevNoteBtn.setMaximumWidth(50)
        self.nextNoteBtn = wdg.QPushButton("next")
        self.nextNoteBtn.setMaximumWidth(50)

        hl = wdg.QHBoxLayout()
        hl.addStretch(1)
        hl.addWidget(self.prevNoteBtn)
        hl.addWidget(self.nextNoteBtn)
        hl.addStretch(1)
        whl = wdg.QWidget()
        whl.setLayout(hl)

        self.keySelLayout.addWidget(whl)

        # put everything together
        self.keySelLayout.addStretch(1)
        self.keySelGroup.setLayout(self.keySelLayout)

        ###########################
        #   Graphs
        ###########################
        self.grWidget = pg.GraphicsLayoutWidget()
        self.plot_freq_base = self.grWidget.addPlot(row=0, col=0)
        p2 = self.grWidget.addPlot(row=0, col=1)
        p3 = self.grWidget.addPlot(row=0, col=2)
        self.plot_time = self.grWidget.addPlot(row=1, col=0, colspan=3)
        self.plot_time.setYRange(-1, 1, padding=0.05)

        self.mainLayout = wdg.QGridLayout()
        self.mainLayout.addWidget(self.grWidget, 0, 0, 2, 1)
        self.mainLayout.addWidget(self.settingsGroup, 0, 1, 1, 1)
        self.mainLayout.addWidget(self.keySelGroup, 1, 1, 1, 1)

        self.mainWidget = wdg.QWidget()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)

        self.buf = util.RecBuff(
            recDev=self.currentMic, samplerate=self.sample_rate, length=self.rec_length
        )
        self.buf.start_rec()

        self.micSignalPlotData = None
        self.plot_time_t = None
        self.plot_time_skip = None
        self.plot_time_n = 3000

        self.ft_plot_data = None

        self.init_draw()

        # self.plotCanvan.axes_time.set_xlim(0, self.buf.t[-1])
        # self.plotCanvan.axes_time.set_ylim(-1, 1)

        # self.plotThreadStopEvent = threading.Event()
        # self.plotThread = threading.Thread(target=self.draw)
        # self.plotThread.start()
        self.plottingtimer = QTimer()
        self.plottingtimer.timeout.connect(self.draw)
        #self.plottingtimer.start(5)

        self.show()

    def update_labels(self):
        """
        write sample_rate, rec_length and target_freq to the appropriate labels
        """
        self.recLengthValueLabel.setText("{:.3g}ms".format(self.rec_length*1000))
        self.samplerateValueLabel.setText("{}/s".format(self.sample_rate))
        self.targetFreqShow.setText("{:.2f} Hz".format(self.target_freq))


    def param_changed(self):
        self.calculate_parameters()
        self.update_labels()


    def update_targetFreq(self):
        s = self.keyEdit.text()
        if s in util.keys:
            self.key = s
            self.keyEdit.setStyleSheet("QLineEdit { background: rgb(255, 255, 255) }")
        else:
            self.keyEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False

        try:
            _level = float(self.levelEdit.text())
            if _level.is_integer():
                self.level = int(_level)
                self.levelEdit.setStyleSheet(
                    "QLineEdit { background: rgb(255, 255, 255) }"
                )
            else:
                raise ValueError("level is not an integer")
        except ValueError:
            self.levelEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False

        try:
            self.ref_freq = float(self.refFreqEdit.text())
            self.refFreqEdit.setStyleSheet(
                "QLineEdit { background: rgb(255, 255, 255) }"
            )
        except ValueError:
            self.refFreqEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False

        self.param_changed()
        logging.debug(
            "new input: target frequency set to {:.2f} HZ".format(self.target_freq)
        )

    def update_num_cyc(self):
        try:
            self.num_cyc = float(self.numCycEdit.text())
            self.numCycEdit.setStyleSheet(
                "QLineEdit { background: rgb(255, 255, 255) }"
            )
        except ValueError:
            self.numCycEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False
        self.param_changed()
        logging.debug("update num_cyc successfully -> {}".format(self.num_cyc))


    def update_smpl_per_cyc(self):
        try:
            self.smpl_per_cyc = float(self.smpl_per_cycEdit.text())
            self.smpl_per_cycEdit.setStyleSheet(
                "QLineEdit { background: rgb(255, 255, 255) }"
            )
        except ValueError:
            self.smpl_per_cycEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False
        self.param_changed()
        logging.debug(
            "update smpl_per_cyc successfully -> {}".format(self.smpl_per_cyc)
        )


    def nextTargetFreq(self):
        self.modTargetFreq(+1)

    def prevTargetFreq(self):
        self.modTargetFreq(-1)

    def modTargetFreq(self, step):
        i = util.keys_piano.index(self.key) + step
        self.level += i // 12
        self.key = util.keys_piano[i % 12]

        self.keyEdit.setText(self.key)
        self.levelEdit.setText(str(self.level))

        self.param_changed()
        logging.debug(
            "mod target freq (step {}): target frequency set to {:.2f} HZ".format(
                step, self.target_freq
            )
        )

    def restart_buff(self):
        self.stop_and_clean_draw()
        self.buf.stop_rec()
        del self.buf

        self.buf = util.RecBuff(
            recDev=self.currentMic,
            samplerate=self.sample_rate,
            length=self.rec_length,
            blocksize=self.block_size,
            numframes=self.num_frames,
        )
        self.buf.start_rec()
        self.init_draw()
        self.plottingtimer.start(5)



    def init_draw(self):
        self.plot_time_skip = max(self.buf.n // self.plot_time_n, 1)
        self.plot_time_t = self.buf.t[:: self.plot_time_skip]
        micSignal = self.buf.data[:: self.plot_time_skip]
        micSignal_full = self.buf.data
        self.full_time_t = self.buf.t
        self.full_time_dt = self.full_time_t[1]
        self.mid_time = self.full_time_t[-1] / 2
        self.ft_window_gauss_width = 0.5 * self.full_time_t[-1]

        if self.micSignalPlotData:
            self.micSignalPlotData.setData(
                self.plot_time_t, micSignal
            )  # overwrite existing data
        else:
            self.micSignalPlotData = self.plot_time.plot(
                self.plot_time_t, micSignal
            )  # create new data to plot

        self.ft_window_gauss = np.exp(
            -((self.full_time_t - self.mid_time) ** 2) / self.ft_window_gauss_width ** 2
        )
        self.ft_window = self.ft_window_gauss
        self.w_list_base = np.linspace(0.8 * self.target_freq, 1.2 * self.target_freq, 25)

        # self.abs_ft_base = self.abs_ft(
        #     self.w_list_base,
        #     self.full_time_t,
        #     self.full_time_dt,
        #     micSignal_full,
        #     self.ft_window,
        # )
        # if self.ft_plot_data:
        #     self.ft_plot_data.setData(self.w_list_base, self.abs_ft_base)
        # else:
        #     self.ft_plot_data = self.plot_freq_base.plot(
        #         self.w_list_base, self.abs_ft_base
        #     )

    def stop_and_clean_draw(self):
        self.plottingtimer.stop()
        self.plot_time.removeItem(self.micSignalPlotData)
        del self.micSignalPlotData
        self.micSignalPlotData = None

    def draw(self):
        self.micSignalPlotData.setData(
            self.plot_time_t, self.buf.data[:: self.plot_time_skip]
        )  # overwrite existing data
        self.abs_ft_base = self.abs_ft(
            self.w_list_base,
            self.full_time_t,
            self.full_time_dt,
            self.buf.data,
            self.ft_window,
        )
        self.ft_plot_data.setData(self.w_list_base, self.abs_ft_base)

    def closeEvent(self, event):
        logging.debug("QApp gets closeEvent")
        self.buf.stop_rec()
        self.stop_and_clean_draw()


class FreqCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_time = plt.subplot2grid(
            shape=(2, 3), loc=(1, 0), colspan=3, fig=self.fig
        )
        self.axes_freq_base = plt.subplot2grid(shape=(2, 3), loc=(0, 0), fig=self.fig)
        self.axes_freq_1 = plt.subplot2grid(shape=(2, 3), loc=(0, 1), fig=self.fig)
        self.axes_freq_2 = plt.subplot2grid(shape=(2, 3), loc=(0, 2), fig=self.fig)
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.98)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(
            self, wdg.QSizePolicy.Expanding, wdg.QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)


if __name__ == "__main__":
    app = wdg.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
