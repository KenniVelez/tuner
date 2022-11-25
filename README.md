# Tuner
by Richard Hartmann

Tuner - frequency analysis of the microphone input suitable to tune instruments.

## Install and Run

if you have not, install `poetry`, dependency management and packaging in Python.  

Use `poetry install` to install dependencies.

Run the app by `poetry run python start_tuner.py`

## Usage

0) Make sure the desired microphone is selected.
   If you do not want to tune to a at 440Hz, change the value
1) Select the key you want to tune to. Use the 'next' and 'prev' to see how the
   inputs 'key' and 'level' work. It also shows the target frequency of that key.
2) Play a tone and tune such that the relative difference (upper right panel) 
   is well below 1% (0.01) for some time. 
   You can use the mouse (drag and scroll) to adjust the axes.

## Donation

If you like the tuner-App, feel free to 
[donate with PayPal](https://www.paypal.com/donate/?hosted_button_id=E8LH2WMYGQCGG).
If you don't like PayPal, contact me at richard_hartmann(at)gmx.&#8574;&#8519;.


## Basic Idea of the Algorithm

1) Perform Fourier integral (finite integral over several oscillations 
   of the recoded microphone signal) for frequencies in the vicinity of a given
   target frequency &Omega;, and multiples of that (higher harmonics contributions).
2) Plot such spectra and shift the frequency axis by -n&Omega; 
   (different shift for each higher harmonic!).
   In that way, if the base frequency of the recoded signal &omega;<sub>1</sub> 
   is resonant with &Omega;, all maxima should align at &omega;-n&Omega; = 0.
   In practice the maxima of the spectra vary much over time, 
   so their alignment is not well suited to judge the resonance, 
   i.e. the tuning to the target frequency &Omega;.
   (upper left plot)
3) Use the amplitude and phase information from the Fourier integral as
   initial condition to fit a harmonic model signal with base frequency 
   &omega;<sub>1</sub> and its higher harmonics n&omega;<sub>1</sub> to the 
   recoded input. 
   (lower plot)
4) Show the relative difference of the optimal value &omega;<sub>1</sub> with 
   respect to the target frequency &Omega;.
   Keep track of its value over some time.
   Record no value if the least square fitting does not converge sufficiently fast.
   (upper right plot)


