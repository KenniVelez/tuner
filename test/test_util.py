from tuner import util

import numpy as np


def test_fft_trapz():
    t = np.linspace(0, 10, 150)
    signal = np.sin(t)

    w = np.linspace(0, 1, 15)

    ft = util.fourier_int_array(signal_t=signal, t=t, w=w)

    for i, wi in enumerate(w):
        ft1 = util.fourier_int(signal_t=signal, t=t, w=wi)
        ft2 = ft[i]
        assert abs(ft1 - ft2) < 1e-15



if __name__ == "__main__":
    test_fft_trapz()