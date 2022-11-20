import sys

#import os
#os.environ["QT_DEBUG_PLUGINS"] = "1"

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import QTimer
import PyQt5.QtCore as QtCore
import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import time

import threading

import soundcard as sc
from . import util
from . import config

import logging
import warnings

logging.basicConfig(level=logging.DEBUG)

BG_FAIL_COLOR = "rgb(255, 100, 100)"


def round_to_sig_digets(x, n=3):
    f = 10 ** (n - int(np.log10(x)))
    x = round(x * f)
    return x / f


class TunerApp(QtWidgets.QMainWindow):
    def __init__(self):

        # default values for the fundamental parameters
        self.ref_freq = config.REFERNCE_FREQUENCY
        self.key = config.KEY
        self.level = config.LEVEL
        self.num_cyc = config.NUMBER_OF_CYCLES
        self.smpl_per_cyc = config.SAMPLES_PER_CYCLE

        # some constants for the recoding buffer
        self.block_size = config.BLOCK_SIZE
        self.num_frames = config.NUM_FRAMES

        # calculated on calling calculate_parameters
        self.rec_length = None
        self.sample_rate = None
        self.target_freq = None


        # init some plot data / parameters
        self.plot_refresh_time_in_ms = config.PLOT_REFRESH_TIME

        self.signal_plot_time_ms = None
        self.signal_plot_number_of_cycles = config.SIGNAL_PLOT_NUMBER_OF_CYCLES

        self.fourier_plot_w1 = None
        self.fourier_plot_w2 = None
        self.fourier_plot_w3 = None
        self.fourier_plot_dw = config.FOURIER_PLOT_DELTA_W_IN_PERCENT
        self.fourier_plot_n = config.FOURIER_PLOT_NUMBER_OF_DATA_POINTS
        self.fourier_freq_data = None
        self.fourier_freq_time = None

        self.sample_rate_fac = config.SAMPLE_RATE_FACTOR_FOR_SIGNAL_PLOT

        self.main_frequency_plot_memory_length = config.MAIN_FREQUENCY_PLOT_MEMORY_LENGTH
        self.main_frequency_plot_data_size = int(self.main_frequency_plot_memory_length / (self.plot_refresh_time_in_ms/1000))
        self.main_frequency_plot_time = np.linspace(-self.main_frequency_plot_memory_length, 0, self.main_frequency_plot_data_size)
        self.main_frequency_plot_levels = config.MAIN_FREQUENCY_PLOT_LEVELS

        if self.num_cyc <= self.signal_plot_number_of_cycles:
            raise ValueError(
                "the number of cycles to plot ({}) needs to be smaller ".format(self.signal_plot_number_of_cycles) +
                "than the number of cycles that are used for the frequency calculation ({})".format(self.num_cyc)
            )

        super().__init__()
        self.title = "Tuner"
        self.initUI()

        # triggers calculation of missing parameters and
        # set the appropriate labels
        self.calculate_parameters()
        self.update_labels()

        # connect button click events
        # key selection
        self.refFreqEdit.editingFinished.connect(self.update_targetFreq)
        self.keyEdit.editingFinished.connect(self.update_targetFreq)
        self.levelEdit.editingFinished.connect(self.update_targetFreq)
        self.nextNoteBtn.clicked.connect(self.nextTargetFreq)
        self.prevNoteBtn.clicked.connect(self.prevTargetFreq)
        # sampling parameters
        self.numCycEdit.editingFinished.connect(self.update_num_cyc)
        self.smpl_per_cycEdit.editingFinished.connect(self.update_smpl_per_cyc)

        try:
            self.DEBUG_buffer_from_file = config.DEBUG_BUFFER_FROM_FILE
        except AttributeError:
            self.DEBUG_buffer_from_file = None

        # setup and start the buffer which holds the microphone signal
        if self.DEBUG_buffer_from_file:
            self.buf = util.RecBuff_debug_from_file(self.DEBUG_buffer_from_file)
        else:
            self.buf = util.RecBuff(
                recDev=self.currentMic,
                samplerate=self.sample_rate_fac * self.sample_rate,
                length=self.rec_length
            )
            self.buf.start_rec()

        self.signal_plot_timer = QTimer()

        self.init_plot_data_items()
        self.signal_plot_timer.timeout.connect(self.update_signal_plot)

        # start plotting timers and show window
        self.signal_plot_timer.start(self.plot_refresh_time_in_ms)
        self.show()

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

        self.permanent_status_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.permanent_status_label)

        ###########################
        #   recording / sampling
        ###########################
        # settings on the right
        self.settingsGroup = QtWidgets.QGroupBox(title="recording / sampling")
        self.settingsGroup.setMinimumWidth(300)
        self.settingsGroup.setMaximumWidth(300)
        self.settingsLayout = QtWidgets.QVBoxLayout()

        # choose mic Label
        self.micDevLabel = QtWidgets.QLabel("choose microphone")
        self.settingsLayout.addWidget(self.micDevLabel)

        # choose mic Combo
        self.micDevDropDown = QtWidgets.QComboBox()
        all_mics = [str(d) for d in sc.all_microphones()]
        for d in all_mics:
            self.micDevDropDown.addItem(d)
        self.currentMic = sc.default_microphone()
        i = all_mics.index(str(self.currentMic))
        self.micDevDropDown.setCurrentIndex(i)
        self.settingsLayout.addWidget(self.micDevDropDown)

        # refFreq Label
        self.refFreqLabel = QtWidgets.QLabel("reference freq. for 'a' (in Hz)")
        self.settingsLayout.addWidget(self.refFreqLabel)

        # refFreqLabel edit
        self.refFreqEdit = QtWidgets.QLineEdit(str(self.ref_freq))
        self.settingsLayout.addWidget(self.refFreqEdit)

        # rec length Label
        self.recLegthLabel = QtWidgets.QLabel("recoding length")
        self.settingsLayout.addWidget(self.recLegthLabel)

        # rec length Value Label
        self.recLengthValueLabel = QtWidgets.QLabel()
        self.settingsLayout.addWidget(self.recLengthValueLabel)

        # samplerate Label
        self.samplerateLabel = QtWidgets.QLabel("sampling rate")
        self.settingsLayout.addWidget(self.samplerateLabel)

        # samplerate Value edit
        self.samplerateValueLabel = QtWidgets.QLabel("")
        self.settingsLayout.addWidget(self.samplerateValueLabel)

        # num_cyc Label
        self.numCycLabel = QtWidgets.QLabel("number of cycles (wrt to target freq.)")
        self.settingsLayout.addWidget(self.numCycLabel)

        # num_cyc edit
        self.numCycEdit = QtWidgets.QLineEdit(str(self.num_cyc))
        self.settingsLayout.addWidget(self.numCycEdit)

        # smpl_per_cyc Label
        self.smpl_per_cycLabel = QtWidgets.QLabel("samples per cycles")
        self.settingsLayout.addWidget(self.smpl_per_cycLabel)

        # smpl_per_cyc edit
        self.smpl_per_cycEdit = QtWidgets.QLineEdit(str(self.smpl_per_cyc))
        self.settingsLayout.addWidget(self.smpl_per_cycEdit)

        # put everything together
        self.settingsLayout.addStretch(1)
        self.settingsGroup.setLayout(self.settingsLayout)

        ###########################
        #   key selection
        ###########################

        # key selection
        self.keySelGroup = QtWidgets.QGroupBox(title="select key")
        self.keySelGroup.setMinimumWidth(300)
        self.keySelGroup.setMaximumWidth(300)
        self.keySelLayout = QtWidgets.QVBoxLayout()

        # key Label
        self.keyLabel = QtWidgets.QLabel("key")
        self.keySelLayout.addWidget(self.keyLabel)

        # key edit
        self.keyEdit = QtWidgets.QLineEdit(self.key)
        self.keySelLayout.addWidget(self.keyEdit)

        # level Label
        self.levelLabel = QtWidgets.QLabel("level")
        self.keySelLayout.addWidget(self.levelLabel)

        # level edit
        self.levelEdit = QtWidgets.QLineEdit(str(self.level))
        self.keySelLayout.addWidget(self.levelEdit)

        # target freq Label
        self.targetFreqLabel = QtWidgets.QLabel("target frequency")
        self.keySelLayout.addWidget(self.targetFreqLabel)

        # target freq Label
        self.targetFreqShow = QtWidgets.QLabel()
        self.keySelLayout.addWidget(self.targetFreqShow)

        self.prevNoteBtn = QtWidgets.QPushButton("prev")
        self.prevNoteBtn.setMaximumWidth(50)
        self.nextNoteBtn = QtWidgets.QPushButton("next")
        self.nextNoteBtn.setMaximumWidth(50)

        hl = QtWidgets.QHBoxLayout()
        hl.addStretch(1)
        hl.addWidget(self.prevNoteBtn)
        hl.addWidget(self.nextNoteBtn)
        hl.addStretch(1)
        whl = QtWidgets.QWidget()
        whl.setLayout(hl)

        self.keySelLayout.addWidget(whl)

        # put everything together
        self.keySelLayout.addStretch(1)
        self.keySelGroup.setLayout(self.keySelLayout)

        ###########################
        #   Graphs
        ###########################
        self.grWidget = pg.GraphicsLayoutWidget()

        # the plot of the Fourier transform data
        self.plot_fourier = self.grWidget.addPlot(row=0, col=0)
        # self.plot_fourier.setYRange(0, 1, padding=0.01)
        # the plot of the fitted base frequency over time
        self.plot_freq = self.grWidget.addPlot(row=0, col=1)
        self.plot_freq.plot(
            x=[-self.main_frequency_plot_memory_length, 0],
            y=[0, 0],
            pen={'style': QtCore.Qt.DotLine}
        )
        for l in self.main_frequency_plot_levels:
            self.plot_freq.plot(
                x=[-self.main_frequency_plot_memory_length, 0],
                y=[l, l],
                pen={'style': QtCore.Qt.DotLine}
            )
            self.plot_freq.plot(
                x=[-self.main_frequency_plot_memory_length, 0],
                y=[-l, -l],
                pen = {'style': QtCore.Qt.DotLine}
            )
        self.plot_freq.setYRange(-2*l, 2*l)
        # the plot of the microphone signal
        self.plot_signal = self.grWidget.addPlot(row=1, col=0, colspan=2)
        # self.plot_signal.setYRange(-1, 1, padding=0.01)

        ###########################
        #   put everything to the main window
        ###########################
        self.mainLayout = QtWidgets.QGridLayout()
        self.mainLayout.addWidget(self.grWidget, 0, 0, 2, 1)
        self.mainLayout.addWidget(self.settingsGroup, 0, 1, 1, 1)
        self.mainLayout.addWidget(self.keySelGroup, 1, 1, 1, 1)

        self.mainWidget = QtWidgets.QWidget()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)


    def update_labels(self):
        """
        write sample_rate, rec_length and target_freq to the appropriate labels
        """
        self.recLengthValueLabel.setText("{:.3g}ms".format(self.rec_length*1000))
        self.samplerateValueLabel.setText("{}/s".format(self.sample_rate))
        self.targetFreqShow.setText("{:.2f} Hz ({:.2f}ms)".format(self.target_freq, 1000/self.target_freq))

    def param_changed(self):
        self.stop_and_clean_draw()

        self.calculate_parameters()
        self.update_labels()
        self.restart_buff()

        self.init_plot_data_items()
        self.signal_plot_timer.start(self.plot_refresh_time_in_ms)

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
        if self.DEBUG_buffer_from_file is None:
            self.buf.stop_rec()
            del self.buf

            self.buf = util.RecBuff(
                recDev=self.currentMic,
                samplerate=self.sample_rate_fac * self.sample_rate,
                length=self.rec_length,
                blocksize=self.block_size,
                numframes=self.num_frames,
            )
            self.buf.start_rec()

    def stop_and_clean_draw(self):
        self.signal_plot_timer.stop()

        # clean up the microphone signal data item
        self.plot_signal.removeItem(self.plot_signal_plot_data_item)
        del self.plot_signal_plot_data_item
        self.plot_signal_plot_data_item = None

        if self.plot_signal_fit_plot_data_item:
            self.plot_signal.removeItem(self.plot_signal_fit_plot_data_item)
            del self.plot_signal_fit_plot_data_item
            self.plot_signal_fit_plot_data_item = None

        # clean up the Fourier plot data item
        self.plot_fourier.removeItem(self.plot_fourier_w1_data_item)
        del self.plot_fourier_w1_data_item
        self.plot_fourier_w1_data_item = None

        self.plot_fourier.removeItem(self.plot_fourier_v1_data_item)
        del self.plot_fourier_v1_data_item
        self.plot_fourier_v1_data_item = None

        # clean up the Fourier plot data item
        self.plot_fourier.removeItem(self.plot_fourier_w2_data_item)
        del self.plot_fourier_w2_data_item
        self.plot_fourier_w2_data_item = None

        self.plot_fourier.removeItem(self.plot_fourier_v2_data_item)
        del self.plot_fourier_v2_data_item
        self.plot_fourier_v2_data_item = None

        # clean up the Fourier plot data item
        self.plot_fourier.removeItem(self.plot_fourier_w3_data_item)
        del self.plot_fourier_w3_data_item
        self.plot_fourier_w3_data_item = None

        self.plot_fourier.removeItem(self.plot_fourier_v3_data_item)
        del self.plot_fourier_v3_data_item
        self.plot_fourier_v3_data_item = None

        self.plot_freq.removeItem(self.plot_main_frequency_data_item)
        del self.plot_main_frequency_data_item
        self.plot_main_frequency_data_item = None

    def init_plot_data_items(self):
        """
        Create the plot items so that they can be modified by the update routine triggered by the timer.
        Create some convenient data sets, so that they do not need to be recomputed by the update routine.
        """
        #######################
        #   the signal plot
        #######################
        # create x-data
        self.signal_plot_time_ms = np.arange(
            0,
            self.signal_plot_number_of_cycles / self.target_freq,   # T
            1/(self.sample_rate_fac * self.sample_rate)             # delta T
        ) * 1000
        # set the x range
        self.plot_signal.setXRange(0, self.signal_plot_time_ms[-1], padding=0.05)
        # init plot item with zeros as y-data
        self.plot_signal_plot_data_item = self.plot_signal.plot(
            x=self.signal_plot_time_ms,
            y=[0]*len(self.signal_plot_time_ms)
        )

        # init this plot data with None
        # because we will only draw the fit, if the least square fitting works
        # thus, the plot will be created / deleted dynamically
        self.plot_signal_fit_plot_data_item = None

        #######################
        #   the Fourier plot
        #######################
        # create x-data
        dw = self.fourier_plot_dw / 100 * self.target_freq
        self.fourier_plot_w1 = np.linspace(
            self.target_freq - dw,
            self.target_freq + dw,
            self.fourier_plot_n
        )

        self.fourier_plot_w2 = np.linspace(
            2*self.target_freq - dw,
            2*self.target_freq + dw,
            self.fourier_plot_n
        )

        self.fourier_plot_w3 = np.linspace(
            3*self.target_freq - dw,
            3*self.target_freq + dw,
            self.fourier_plot_n
        )

        self.fourier_plot_t = self.buf.t[::self.sample_rate_fac]
        self.fourier_plot_tmax = self.fourier_plot_t[-1]

        # init plot item with zeros as y-data
        self.plot_fourier_w1_data_item = self.plot_fourier.plot(
            x=self.fourier_plot_w1-self.target_freq,
            y=[0] * self.fourier_plot_n,
            pen={'color': '#F00', 'width': 3}
        )
        self.plot_fourier_v1_data_item = self.plot_fourier.plot(
            x=[0, 0],
            y=[0, 0],
            pen={'color': '#F00', 'width': 3}
        )

        self.plot_fourier_w2_data_item = self.plot_fourier.plot(
            x=self.fourier_plot_w2-2*self.target_freq,
            y=[0] * self.fourier_plot_n,
            pen={'color': '#0F0', 'width': 2}
        )
        self.plot_fourier_v2_data_item = self.plot_fourier.plot(
            x=[0, 0],
            y=[0, 0],
            pen={'color': '#0F0', 'width': 2}
        )
        self.plot_fourier_w3_data_item = self.plot_fourier.plot(
            x=self.fourier_plot_w3-3*self.target_freq,
            y=[0] * self.fourier_plot_n,
            pen={'width': 1}
        )
        self.plot_fourier_v3_data_item = self.plot_fourier.plot(
            x=[0, 0],
            y=[0, 0],
            pen={'width': 1}
        )

        #######################
        #   the main frequency plot
        #######################
        self.main_frequency_plot_data = np.empty(
            shape=self.main_frequency_plot_data_size
        )
        self.main_frequency_plot_data[:] = 0
        self.plot_main_frequency_data_item = self.plot_freq.plot(
            x=self.main_frequency_plot_time,
            y=self.main_frequency_plot_data,
            pen={'color': '#F00', 'width': 3}
        )

    def update_signal_plot(self):
        t0 = time.perf_counter_ns()

        raw_data = self.buf.copy_data()

        #######################
        #   the Fourier plot
        #######################
        data_for_FT = raw_data[::self.sample_rate_fac]
        ft_w1 = 1 / self.fourier_plot_tmax * util.fourier_int_array(data_for_FT, self.fourier_plot_t,
                                                                    2 * np.pi * self.fourier_plot_w1)
        ft_w2 = 1 / self.fourier_plot_tmax * util.fourier_int_array(data_for_FT, self.fourier_plot_t,
                                                                    2 * np.pi * self.fourier_plot_w2)
        ft_w3 = 1 / self.fourier_plot_tmax * util.fourier_int_array(data_for_FT, self.fourier_plot_t,
                                                                    2 * np.pi * self.fourier_plot_w3)

        ft_w1_abs = np.abs(ft_w1)
        ft_w2_abs = np.abs(ft_w2)
        ft_w3_abs = np.abs(ft_w3)
        sig_max = np.max(np.abs(data_for_FT))

        self.plot_fourier_w1_data_item.setData(
            x=self.fourier_plot_w1 - self.target_freq,
            y=ft_w1_abs / sig_max
        )
        self.plot_fourier_w2_data_item.setData(
            x=self.fourier_plot_w2 - 2 * self.target_freq,
            y=ft_w2_abs / sig_max
        )
        self.plot_fourier_w3_data_item.setData(
            x=self.fourier_plot_w3 - 3 * self.target_freq,
            y=ft_w3_abs / sig_max
        )

        idx1 = np.argmax(ft_w1_abs)
        idx2 = np.argmax(ft_w2_abs)
        idx3 = np.argmax(ft_w3_abs)
        scale = ft_w1_abs[idx1] * 1.1 / sig_max
        w1_max = self.fourier_plot_w1[idx1] - self.target_freq
        w2_max = self.fourier_plot_w2[idx2] - 2 * self.target_freq
        w3_max = self.fourier_plot_w3[idx3] - 3 * self.target_freq
        self.plot_fourier_v1_data_item.setData(
            x=[w1_max, w1_max],
            y=[0, scale]
        )
        self.plot_fourier_v2_data_item.setData(
            x=[w2_max, w2_max],
            y=[0, scale]
        )
        self.plot_fourier_v3_data_item.setData(
            x=[w3_max, w3_max],
            y=[0, scale]
        )


        try:
            omg_fit, amp, phi, r = util.fit_harmonic_function(
                signal_t=raw_data,
                t=self.buf.t,
                omg_guess=2*np.pi*self.fourier_plot_w1[idx1],
                N=3,
                kwargs_least_squares={'max_nfev': 15}
            )
            #print(r.optimality, r.nfev)
            phi0 = phi[0] % (2*np.pi)
            T = 2*np.pi / omg_fit
            t_sh = 3*T/4 - phi0 / omg_fit
            if t_sh < 0:
                t_sh += T

            idx = int(t_sh // self.buf.dt)
            t_sh_fine = t_sh - self.buf.t[idx]
        except RuntimeError:
            omg_fit = None
            idx = 0
            t_sh_fine = 0

        #######################
        #   the signal plot
        #######################

        data = raw_data[idx:len(self.signal_plot_time_ms)+idx]
        self.plot_signal_plot_data_item.setData(
            x=self.signal_plot_time_ms - t_sh_fine*1000,
            #x=self.signal_plot_time_ms,
            y=data
        )
        if omg_fit:
            self.main_frequency_plot_data[0:-1] = self.main_frequency_plot_data[1:]
            #self.main_frequency_plot_data[-1] = (omg_fit - self.target_freq) / self.target_freq
            self.main_frequency_plot_data[-1] = omg_fit
            sig_fit_t = util.n_harmonic_function(
                t=self.signal_plot_time_ms/1000,
                omg_base=omg_fit,
                amp=amp,
                phi=phi-phi0 - np.pi/2
            )
            if self.plot_signal_fit_plot_data_item is None:
                self.plot_signal_fit_plot_data_item = self.plot_signal.plot(
                    x=self.signal_plot_time_ms,
                    y=sig_fit_t,
                    pen={'color': '#F00', 'width': 3}
                )
            else:
                self.plot_signal_fit_plot_data_item.setData(
                    x=self.signal_plot_time_ms,
                    y=sig_fit_t
                )
        else:
            self.main_frequency_plot_data[0:-1] = self.main_frequency_plot_data[1:]
            self.main_frequency_plot_data[-1] = 0
            if self.plot_signal_fit_plot_data_item is not None:
                self.plot_signal.removeItem(self.plot_signal_fit_plot_data_item)
                del self.plot_signal_fit_plot_data_item
                self.plot_signal_fit_plot_data_item = None

        connect = self.main_frequency_plot_data != 0
        connect = np.asarray(np.logical_and(connect, np.roll(connect, -1)), dtype=np.int)
        self.plot_main_frequency_data_item.setData(
            x=self.main_frequency_plot_time,
            y=(self.main_frequency_plot_data/(2*np.pi)-self.target_freq) / self.target_freq,
            connect=connect
        )


        t1 = time.perf_counter_ns()
        dt = (t1 - t0) / 10**6
        self.permanent_status_label.setText("plotting takes {:.2f}ms".format(dt))
        if dt > self.plot_refresh_time_in_ms:
            warnings.warn("plotting takes longer that refresh time, timer too fast")



    def closeEvent(self, event):
        logging.debug("QApp gets closeEvent")
        self.buf.stop_rec()
        self.stop_and_clean_draw()

    def keyPressEvent(self, event):
        if event.key() == 83:
            fname = "{}_tuner_dump".format(time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))
            print("dump current buffer to {}.npy".format(fname))
            data = np.vstack((self.buf.t, self.buf.copy_data()))
            np.save(fname, data)
