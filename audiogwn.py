import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
from scipy import signal



song=AudioSegment.from_wav('bells.wav')
play(song)


samplerate,data= sf.read('bells.wav')

N = len(samplerate)
slength = N/data
t1 = np.linspace(0, N/data, N)
plt.figure(1)
plt.plot(t1, samplerate,'orchid',alpha=0.85)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()

sampling_rate = 11025
Nsamples = 65740 
noiseSigma = 0.1
noise = np.random.normal(0, noiseSigma, Nsamples)

noisy_signal = samplerate + noise

sf.write('noisy.wav', noisy_signal, sampling_rate)

song_after=AudioSegment.from_wav('noisy.wav')
play(song_after)

f,Fs= sf.read('noisy.wav')
N = len(f)
slength = N/Fs
t2 = np.linspace(0, N/Fs, N)
plt.figure(2)
plt.plot(t2, f,'teal',alpha=0.85)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()

h=np.array([-0.015,0.058,-0.350,1.000,-0.350,0.058,-0.005])
x=noisy_signal
y = signal.lfilter(h,1,x)

plt.figure(3)
plt.plot(t2, y, 'navy', alpha=0.85)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()
plt.show()










