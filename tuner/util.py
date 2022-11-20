import soundcard as sc
import numpy as np
from scipy.optimize import least_squares
import threading
import logging


keys = ['a', 'a#', 'h', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#']
keys_piano = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'h']


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
    i = keys.index(note)
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
        self.dt = 1/samplerate
        self.tmax = (self.n-1) * self.dt
        self.t = np.linspace(0, self.tmax, self.n)
        self.stopEvent = threading.Event()
        self.recThread = None
        self.data_access_lock = threading.Lock()

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

                with self.data_access_lock:
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

    def copy_data(self):
        with self.data_access_lock:
            return self.data.copy()


class RecBuff_debug_from_file(object):
    def __init__(self, fname):
        """
        read constant buffer from file, DEBUG
        """
        data = np.load(fname)
        t = data[0, :]
        s = data[1, :]
        self.recDev = None
        self.samplerate = None
        self.length = t[-1]
        self.blocksize = None
        self.numframes = None
        self.n = len(t)
        self.data = s
        self.dt = t[1] - t[0]
        self.tmax = t[-1]
        self.t = t
        self.stopEvent = None
        self.recThread = None
        self.data_access_lock = None

    def start_rec(self):
        pass

    def stop_rec(self):
        pass

    def copy_data(self):
        return self.data.copy()


def fourier_int(signal_t, t, w):
    """
    approximate the finite Fourier integral by discrete trapezoidal sum
    """
    return np.trapz(signal_t * np.exp(-1j*w*t), t)


def fourier_int_array(signal_t, t, w):
    """
    approximate the finite Fourier integral by discrete trapezoidal sum for an array of
    frequencies w
    """
    return np.trapz(signal_t * np.exp(-1j*w.reshape(-1, 1)*t), t, axis=1)


def n_harmonic_function(t, omg_base, amp, phi):
    """
    Calculate the N-harmonic function with base frequency 'omg_base'.

    'amp' and 'phi' are the lists (as numpy arrays) of amplitudes and phases of the (higher) harmonic(s), .i.e.
        s(t) = sum_n=1^N amp[n] cos(n * omg_base * t + phi[n]) .

    This function signature is intended to be used for curve_fit (leqast_squares)
    """
    n = np.arange(1, len(amp)+1)
    t = t.reshape(-1, 1)     # new axes for t, to use numpy sum for the summation over n
    return np.sum(amp * np.cos(n * omg_base * t + phi), axis=1)


def n_harmonic_residual(x, t, signal_t, N):
    omg_base = x[0]
    amp = x[1:N+1]
    phi = x[N+1:2*N+1]
    return n_harmonic_function(t, omg_base, amp, phi) - signal_t


def fit_harmonic_function(signal_t, t, omg_guess, N, kwargs_least_squares={}):
    """
    Fit the N-harmonic function to the signal data.

    :param signal_t: the signal data
    :param t: the time axes of the signal data
    :param omg_guess: an initial guess of the base frequency (in radians per sec)
    :param: the number of (higher) harmonics, N=2 means base frequency and the first higher harmonic.
    """

    n = np.arange(1, N+1)
    w = n*omg_guess
    S_wi = np.asarray([fourier_int(signal_t, t, wi) for wi in w])
    amp = np.abs(S_wi)
    phi = np.angle(S_wi)

    r = least_squares(
        fun=n_harmonic_residual,
        x0=np.concatenate(([omg_guess], amp, phi)),
        args=(t, signal_t, N),
        **kwargs_least_squares
    )

    if not r.success:
        raise RuntimeError("least_squares did not converge")

    omg_base = r.x[0]
    amp = r.x[1:N+1]
    phi = r.x[N+1:2*N+1]

    return omg_base, amp, phi, r









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
