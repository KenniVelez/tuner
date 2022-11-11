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
import tuner_util
import logging

logging.basicConfig(level=logging.DEBUG)

BG_FAIL_COLOR = "rgb(255, 100, 100)"


def round_to_sig_digets(x, n=3):
    f = 10 ** (n - int(np.log10(x)))
    x = round(x * f)
    return x / f


class App(wdg.QMainWindow):
    def __init__(self):
        self.refFreq = 440

        self.note = "c"
        self.level = 0
        self.targetFreq = tuner_util.note_to_freq(
            note=self.note, level=self.level, a=self.refFreq
        )

        self.num_cyc = 25
        self.smpl_per_cyc = 35

        self.recLength = self.num_cyc / self.targetFreq
        self.samplerate = int(self.smpl_per_cyc * self.targetFreq)
        self.blocksize = 512
        self.numframes = 128

        super().__init__()
        self.title = "Tuner"
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, 1600, 1000)

        # setting on the right
        self.settingsGroup = wdg.QGroupBox(title="settings")
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
        self.refFreqEdit = wdg.QLineEdit(str(self.refFreq))
        self.settingsLayout.addWidget(self.refFreqEdit)

        # rec length Label
        self.recLegthLabel = wdg.QLabel("recoding length (in s)")
        self.settingsLayout.addWidget(self.recLegthLabel)

        # rec length edit
        self.recLengthEdit = wdg.QLineEdit(str(round_to_sig_digets(self.recLength)))
        self.settingsLayout.addWidget(self.recLengthEdit)

        # samplerate Label
        self.samplerateLabel = wdg.QLabel("samplerate (in Hz)")
        self.settingsLayout.addWidget(self.samplerateLabel)

        # samplerate edit

        self.samplerateEdit = wdg.QLineEdit(str(self.samplerate))
        self.settingsLayout.addWidget(self.samplerateEdit)

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

        # blocksize Label
        self.blocksizeLabel = wdg.QLabel("blocksize")
        self.settingsLayout.addWidget(self.blocksizeLabel)

        # blocksize edit
        self.blocksizeEdit = wdg.QLineEdit(str(self.blocksize))
        self.settingsLayout.addWidget(self.blocksizeEdit)

        # numframes Label
        self.numframesLabel = wdg.QLabel("numframes")
        self.settingsLayout.addWidget(self.numframesLabel)

        # numframes edit
        self.numframesEdit = wdg.QLineEdit(str(self.numframes))
        self.settingsLayout.addWidget(self.numframesEdit)

        # put everything together
        self.settingsLayout.addStretch(1)
        self.settingsGroup.setLayout(self.settingsLayout)

        # note selection
        self.noteSelGroup = wdg.QGroupBox(title="select note")
        self.noteSelGroup.setMinimumWidth(300)
        self.noteSelGroup.setMaximumWidth(300)
        self.noteSelLayout = wdg.QVBoxLayout()

        # note Label
        self.noteLabel = wdg.QLabel("note")
        self.noteSelLayout.addWidget(self.noteLabel)

        # note edit
        self.noteEdit = wdg.QLineEdit(self.note)
        self.noteSelLayout.addWidget(self.noteEdit)

        # level Label
        self.levelLabel = wdg.QLabel("level")
        self.noteSelLayout.addWidget(self.levelLabel)

        # level edit
        self.levelEdit = wdg.QLineEdit(str(self.level))
        self.noteSelLayout.addWidget(self.levelEdit)

        # target freq Label
        self.targetFreqLabel = wdg.QLabel("target frequency")
        self.noteSelLayout.addWidget(self.targetFreqLabel)

        # target freq Label
        self.targetFreqShow = wdg.QLabel()
        self.noteSelLayout.addWidget(self.targetFreqShow)

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

        self.noteSelLayout.addWidget(whl)

        self.update_targetFreq()

        self.refFreqEdit.editingFinished.connect(self.update_targetFreq)
        self.noteEdit.editingFinished.connect(self.update_targetFreq)
        self.levelEdit.editingFinished.connect(self.update_targetFreq)

        self.plottingtimer = None
        self.recLengthEdit.editingFinished.connect(self.update_recLength)
        self.samplerateEdit.editingFinished.connect(self.update_samplingrate)
        self.numCycEdit.editingFinished.connect(self.update_num_cyc)
        self.smpl_per_cycEdit.editingFinished.connect(self.update_smpl_per_cyc)
        self.blocksizeEdit.editingFinished.connect(self.update_blocksize)
        self.numframesEdit.editingFinished.connect(self.update_numframes)


        self.nextNoteBtn.clicked.connect(self.nextTargetFreq)
        self.prevNoteBtn.clicked.connect(self.prevTargetFreq)

        # put everything together
        self.noteSelLayout.addStretch(1)
        self.noteSelGroup.setLayout(self.noteSelLayout)

        # the matplotlib canvas
        self.time_plot = None

        # self.plotCanvan = FreqCanvas(parent=None, width=5, height=4)
        # self.plotCanvan.move(0, 0)

        # self.scene = wdg.QGraphicsScene()
        # self.scene.addText("Hallo")
        # self.view = wdg.QGraphicsView(self.scene)

        self.grWidget = pg.GraphicsLayoutWidget()
        self.plot_freq_base = self.grWidget.addPlot(row=0, col=0)
        p2 = self.grWidget.addPlot(row=0, col=1)
        p3 = self.grWidget.addPlot(row=0, col=2)
        self.plot_time = self.grWidget.addPlot(row=1, col=0, colspan=3)
        self.plot_time.setYRange(-1, 1, padding=0.05)

        self.mainLayout = wdg.QGridLayout()
        self.mainLayout.addWidget(self.grWidget, 0, 0, 2, 1)
        self.mainLayout.addWidget(self.settingsGroup, 0, 1, 1, 1)
        self.mainLayout.addWidget(self.noteSelGroup, 1, 1, 1, 1)

        self.mainWidget = wdg.QWidget()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)

        self.buf = tuner_util.RecBuff(
            recDev=self.currentMic, samplerate=self.samplerate, length=self.recLength
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
        self.plottingtimer.start(5)

        self.show()

    def update_targetFreq(self):
        s = self.noteEdit.text()
        if s in tuner_util.notes:
            self.note = s
            self.noteEdit.setStyleSheet("QLineEdit { background: rgb(255, 255, 255) }")
        else:
            self.noteEdit.setStyleSheet(
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
            self.refFreq = float(self.refFreqEdit.text())
            self.refFreqEdit.setStyleSheet(
                "QLineEdit { background: rgb(255, 255, 255) }"
            )
        except ValueError:
            self.refFreqEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False

        self.targetFreq = tuner_util.note_to_freq(
            note=self.note, level=self.level, a=self.refFreq
        )
        self.targetFreqShow.setText("{:.2f} Hz".format(self.targetFreq))
        logging.debug(
            "new input: target frequency set to {:.2f} HZ".format(self.targetFreq)
        )

    def modTargetFreq(self, step):
        i = tuner_util.notes_piano.index(self.note) + step
        self.level += i // 12
        self.note = tuner_util.notes_piano[i % 12]

        self.noteEdit.setText(self.note)
        self.levelEdit.setText(str(self.level))

        self.targetFreq = tuner_util.note_to_freq(
            note=self.note, level=self.level, a=self.refFreq
        )
        self.targetFreqShow.setText("{:.2f} Hz".format(self.targetFreq))
        logging.debug(
            "mod target freq (step {}): target frequency set to {:.2f} HZ".format(
                step, self.targetFreq
            )
        )
        
    def process_new_TargetFreq(self):
        self.recLength = self.num_cyc / self.targetFreq
        self.samplerate = int(self.smpl_per_cyc * self.targetFreq)
        
        self.recLegthLabel.setText(str(round_to_sig_digets(self.recLength)))
        

    def update_recLength(self):
        if not self.plottingtimer:
            return
        try:
            self.recLength = float(self.recLengthEdit.text())
            self.recLengthEdit.setStyleSheet(
                "QLineEdit { background: rgb(255, 255, 255) }"
            )
        except ValueError:
            self.recLengthEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False

        self.num_cyc = self.recLength * self.targetFreq
        self.numCycEdit.setText(str(self.num_cyc))

        logging.debug("update recLength successfully -> {}".format(self.recLength))
        self.restart_buff()

    def update_samplingrate(self):
        if not self.plottingtimer:
            return
        try:
            self.samplerate = int(self.samplerateEdit.text())
            self.samplerateEdit.setStyleSheet(
                "QLineEdit { background: rgb(255, 255, 255) }"
            )
        except ValueError:
            self.samplerateEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False

        self.smpl_per_cyc = self.samplerate / self.targetFreq
        self.smpl_per_cycEdit.setText(str(self.smpl_per_cyc))

        logging.debug("update samplerate successfully -> {}".format(self.samplerate))
        self.restart_buff()

    def update_num_cyc(self):
        if not self.plottingtimer:
            return
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

        self.recLength = self.num_cyc / self.targetFreq
        self.recLengthEdit.setText(str(round_to_sig_digets(self.recLength)))

        logging.debug("update num_cyc successfully -> {}".format(self.num_cyc))
        self.restart_buff()

    def update_smpl_per_cyc(self):
        if not self.plottingtimer:
            return
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

        self.samplerate = int(self.smpl_per_cyc * self.targetFreq)
        self.samplerateEdit.setText(str(self.samplerate))

        logging.debug(
            "update smpl_per_cyc successfully -> {}".format(self.smpl_per_cyc)
        )
        self.restart_buff()

    def update_blocksize(self):
        if not self.plottingtimer:
            return

        try:
            _bs = float(self.blocksizeEdit.text())
            if _bs.is_integer():
                self.blocksize = int(_bs)
                self.blocksizeEdit.setStyleSheet(
                    "QLineEdit { background: rgb(255, 255, 255) }"
                )
            else:
                raise ValueError("blocksize in not an integer")
        except ValueError:
            self.blocksizeEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False

        logging.debug("update blocksize successfully -> {}".format(self.blocksize))
        self.restart_buff()

    def update_numframes(self):
        if not self.plottingtimer:
            return

        try:
            _nf = float(self.numframesEdit.text())
            if _nf.is_integer():
                self.numframes = int(_nf)
                self.numframesEdit.setStyleSheet(
                    "QLineEdit { background: rgb(255, 255, 255) }"
                )
            else:
                raise ValueError("numframes in not an integer")
        except ValueError:
            self.numframesEdit.setStyleSheet(
                "QLineEdit {{ background: {} }}".format(BG_FAIL_COLOR)
            )
            return False

        logging.debug("update numframes successfully -> {}".format(self.blocksize))
        self.restart_buff()

    def restart_buff(self):
        self.stop_and_clean_draw()
        self.buf.stop_rec()
        del self.buf

        self.buf = tuner_util.RecBuff(
            recDev=self.currentMic,
            samplerate=self.samplerate,
            length=self.recLength,
            blocksize=self.blocksize,
            numframes=self.numframes,
        )
        self.buf.start_rec()
        self.init_draw()
        self.plottingtimer.start(5)

    def nextTargetFreq(self):
        self.modTargetFreq(+1)

    def prevTargetFreq(self):
        self.modTargetFreq(-1)

    @staticmethod
    def abs_ft(w, t, dt, signal, window):
        ft = np.asarray(
            [dt * np.sum(signal * np.exp(1j * t * wi) * window) for wi in w]
        )
        return np.abs(ft)

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
        self.w_list_base = np.linspace(0.8 * self.targetFreq, 1.2 * self.targetFreq, 25)

        self.abs_ft_base = self.abs_ft(
            self.w_list_base,
            self.full_time_t,
            self.full_time_dt,
            micSignal_full,
            self.ft_window,
        )
        if self.ft_plot_data:
            self.ft_plot_data.setData(self.w_list_base, self.abs_ft_base)
        else:
            self.ft_plot_data = self.plot_freq_base.plot(
                self.w_list_base, self.abs_ft_base
            )

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
