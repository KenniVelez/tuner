# initial parameter values
REFERNCE_FREQUENCY = 440
KEY = "c"
LEVEL = 0
NUMBER_OF_CYCLES = 15
SAMPLES_PER_CYCLE = 12.3

# some constants for the recoding buffer
BLOCK_SIZE = 512
NUM_FRAMES = 128

# fitting parameters
NUMBER_OF_HIGHER_HARMONICS = 5
LEAST_SQUARES_MAX_NFEV = 5*NUMBER_OF_HIGHER_HARMONICS

# plotting parameters
SAMPLE_RATE_FACTOR_FOR_SIGNAL_PLOT = 2
PLOT_REFRESH_TIME = 100 # ms

SIGNAL_PLOT_NUMBER_OF_CYCLES = 8
SIGNAL_PLOT_STYLE_SIGNAL = {'color': '#bbb', 'width': 1}
SIGNAL_PLOT_STYLE_FIT = {'color': '#e41a1c', 'width': 2}

FOURIER_PLOT_DELTA_W_IN_PERCENT = 8
FOURIER_PLOT_NUMBER_OF_DATA_POINTS = 150
FOURIER_PLOT_NUMBER_OF_HIGHER_HARMONICS = NUMBER_OF_HIGHER_HARMONICS

# taken from https://matplotlib.org/stable/gallery/color/colormap_reference.html
# using the colormap Set2
FOURIER_PLOT_HIGHER_HARMONICS_STYLES = [
    {'color': '#e41a1c', 'width': 4},
    {'color': '#377eb8', 'width': 2},
    {'color': '#4daf4a', 'width': 1},
    {'color': '#984ea3', 'width': 1},
    {'color': '#ff7f00', 'width': 1},
    {'color': '#ffff33', 'width': 1},
]

MAIN_FREQUENCY_PLOT_MEMORY_LENGTH = 10 #sec
MAIN_FREQUENCY_PLOT_AVERAGING_LENGTH = 1 #sec
MAIN_FREQUENCY_PLOT_LEVELS = [0.01, 0.05]
MAIN_FREQUENCY_PLOT_STYLE = {'color': '#e41a1c', 'width': 2}

#DEBUG_BUFFER_FROM_FILE = "./dev/2022_11_18__14_37_19_Gitarre E tief.npy"