#!/usr/bin/env python3
import pyaudio
import wave
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import read
from tts import tts

def recordAudio(filename, recordlength=5, rate=44100, channels=2, chunksize=1024, format=pyaudio.paInt32):
    CHUNK = chunksize
    FORMAT = format
    CHANNELS = channels
    RATE = rate
    RECORD_SECONDS = recordlength
    WAVE_OUTPUT_FILENAME = sys.argv[1] + filename + ".wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def plotFFTMag(fy, sogf_mag, plotTitle="Frequency Response"):
    sns.set_style('darkgrid')
    plt.figure()
    plt.plot(fy, sog0f_mag)
    # label the axes
    plt.ylabel("|Freq(Signal)|")
    plt.xlabel("Freq [Hz]")
    plt.title(plotTitle) # label the title
    plt.grid()
    plt.show() # display the plot

def computeFFTMag(sig0, fs, plot=False, debug=0):
    ''' Given a signal and its sampling rate, plot the one-sided freq response
    In: sig0: 1D array in time domain
    In: fs: sampling rate of sig0

    Out: Mag response and corresponding freq
    '''
    n = len(sig0) # length
    sig0f = np.fft.rfft(sig0) # fft
    sig0f = sig0f/n # normalize
    sig0f1 = sig0f[range(n//2)] # 1-sided
    sog0f_mag = abs(sig0f1) # Magnitude response
    fy = np.linspace(0, fs//(2), n//2) # freq axis
    if plot: plotFFTMag(fy, sogf_mag)
    if debug >= 1: print("Most energy detected at",fy[np.argmax(sog0f_mag)] ,"Hz") # print the Frequency where the max mag response occurs
    return fy, sog0f_mag # returns the frquency and the magnitude response at each frequency


def getNoiseStatistics():
    filenames = sys.path[0] +"/AudioData/NoiseReduction/"
    statistics = {}
    for i,file in enumerate(os.listdir(filenames)):
        audioData = read(filenames+file)
        samplingRate = audioData[0]
        leftMono = audioData[1][0:,0]
        rightMono = audioData[1][0:,1]
        signal = audioData[1]#np.vstack([leftMono, rightMono])
        frequency, magnitude = computeFFTMag(signal, samplingRate)
        statistics["maxFrequency"] = frequency[np.where(frequency==np.argmax(magnitude))]
        statistics["meanAmplitude"] = np.mean(signal.flatten())
        statistics["meanFrequency"] = frequency[np.where(frequency == np.mean(magnitude))]
    return statistics

def 
if __name__ == "__main__":
    noiseStats = getNoiseStatistics()
