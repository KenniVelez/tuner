import soundcard as sc
import numpy as np
import threading
import logging


notes = ['a', 'a#', 'h', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#']
notes_piano = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'h']


def freq_of_key_relative_to_a(i, freq_a=440):
    """
    Calculate the frequency of the key which is 'i' steps above the reference key a.
    The frequency of the reference key is given by 'freq_a' (default 440Hz).
    The calculation is based on twelve-tone equal temperament which is the standard for pianos.
    """
    return freq_a * 2**(i/12)


def freq_of_key_88(n, freq_a=440):
    """
    Calculate the frequency of the key 'n', where 'n' enumerates the keys on a standard 88 key piano
    from below (lowest key with n=1 is the a,,)

    (see freq_of_key_relative_to_a)
    """
    return freq_of_key_relative_to_a(i=n-49, freq_a=freq_a)


def note_to_key(note, level):
    """
    For a given note as string and the octave level, this function returns the key index relative to 'a'

    The string of a note can be 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#' or 'h'.
    """
    i = notes.index(note)
    return 12*level + i - 12*(i>2)


def note_to_freq(note, level, freq_a=440):
    """
    Returns the frequency of the key specified by the note as string and the octave level relative to 'a'.

    see note_to_key, freq_of_key_relative_to_a
    """
    return freq_of_key_relative_to_a(note_to_key(note, level), freq_a)


class RecBuff(object):
    def __init__(self, recDev, samplerate, length, blocksize=512, numframes=128):
        """
        holds the input buffer for a finite length

        :param recDev: the microphone (see soundcard module)
        :param samplerate: samplerate in Hz
        :param length: length of the buffer in s
        :param blocksize: see soundcard module
        :param numframes: see soundcard module
        """
        self.recDev = recDev
        self.samplerate = samplerate
        self.length = length
        self.blocksize = blocksize
        self.numframes = numframes
        self.n = max(int(samplerate*length)+1, numframes)
        self.data = np.zeros(shape=self.n, dtype=np.float64)
        self.t = np.linspace(0, length, self.n)
        self.stopEvent = threading.Event()
        self.recThread = None

        logging.debug("length       {}".format(length))
        logging.debug("samplingrate {}".format(samplerate))
        logging.debug("blocksize    {}".format(blocksize))
        logging.debug("numframes    {}".format(numframes))
        logging.debug("len(data)    {}".format(self.n))

    def _rec(self):
        logging.debug("_rec thread is running")
        with self.recDev.recorder(samplerate=self.samplerate, channels=1, blocksize=self.blocksize) as rec:
            while True:
                new_data = rec.record(numframes=self.numframes).flatten()
                if len(new_data) != self.numframes:
                    logging.debug("numframes {} but len(new_data) {}".format(self.numframes, len(new_data)))

                self.data[:-self.numframes] = self.data[self.numframes:]
                self.data[-self.numframes:] = new_data
                if self.stopEvent.is_set():
                    logging.debug("_rec thread notices stopEvent")
                    break

    def start_rec(self):
        logging.debug('start _rec thread')
        self.recThread = threading.Thread(target=self._rec)
        self.recThread.start()

    def stop_rec(self):
        if self.recThread:
            logging.debug('signal stopEvent')
            self.stopEvent.set()
            self.recThread.join()
            logging.debug('_rec thread has stopped')





# w = np.fft.fftfreq(n, t[1])
# print("dw", w[1])
# len_w = len(w)
# ln, = ax.plot(w, data[:len_w])
#
# timer_0 = time.perf_counter()
#
# with mic.recorder(samplerate=samplerate, channels=1, blocksize=blocksize) as rec:
#     while True:
#         new_data = rec.record(numframes=numframes).flatten()
#         data[:-numframes] = data[numframes:]
#         data[-numframes:] = new_data
#
#         timer_1 = time.perf_counter()
#         if timer_1-timer_0 > update_time:
#
#             spec = np.fft.fft(data)
#             ln.set_data(w, np.abs(spec[:len_w]))
#
#             ln.figure.canvas.flush_events()
#             timer_0 = timer_1
#
