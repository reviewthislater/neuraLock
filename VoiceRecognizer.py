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
from scipy import stats
from scipy.signal import lfilter
from sklearn.preprocessing import normalize


def recordAudio(filename, recordlength=5, rate=44100, channels=1, chunksize=1024, format=pyaudio.paInt32):
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

def plotFFTMag(frequency, sigMag, plotTitle="Frequency Response"):
    sns.set_style('darkgrid')
    plt.figure()
    plt.plot(frequency, sigMag)
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
    if isinstance(sig0, list):
        n = len(sig0)
        sig0f = np.fft.rfft(sig0)
    else:
        n = sig0.size # length
        sig0f = np.fft.rfft(sig0)[0] # fft
    sig0f = sig0f/n # normalize
    sig0f1 = sig0f[range(n//2)] # 1-sided
    sigMag= np.abs(sig0f1) # Magnitude response
    fy = np.linspace(0, fs//(2), n//2) # freq axis
    if plot: plotFFTMag(fy, sigMag)
    if debug >= 1: print("Most energy detected at",fy[np.argmax(sigMag)] ,"Hz") # print the Frequency where the max mag response occurs
    return fy, sigMag # returns the frquency and the magnitude response at each frequency


def getNoiseStatistics():
    filenames = sys.path[0] +"/AudioData/NoiseReduction/"
    statistics = {}
    for i,file in enumerate(os.listdir(filenames)):
        audioData = read(filenames+file)
        samplingRate = audioData[0]
        leftMono = audioData[1][0:,0]
        rightMono = audioData[1][0:,1]
        signal = audioData[1]#np.vstack([leftMono, rightMono])
        frequency, magnitude = computeFFTMag(signal, samplingRate, plot=False)
        statistics["maxFrequency"] = frequency[np.where(frequency == np.argmax(magnitude))]
        statistics["meanAmplitude"] = np.mean(signal.flatten())
        statistics["meanFrequency"] = frequency[np.where(frequency == np.mean(magnitude))]
    return statistics

def filterNoise(audioSignals):
    n = 15  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    for i, audio in enumerate(audioSignals): audioSignals[i] = lfilter(b,a,audio)
    return audioSignals

def normalizeData(audioData):
    for i, audio in enumerate(audioData): audioData[i] = normalize(audio.reshape(1,-1), norm="l1")
    return audioData

def convertToMono(audioSignals):
    if audioSignals[0].shape[1] == 1: return audioSignals # Data is already mono so lets just return it
    elif audioSignals[0].shape[1] == 2:
        for i, audio in enumerate(audioSignals): audioSignals[i] = (audio.sum(axis=1) / 2) # Convert data to mono and return it
        return audioSignals
    else: raise Exception("Whoah you discovered %d dimensional audio" % audioSignals[0].shape[1])

def clipAudio(audioSignals):
    for i, audio in enumerate(audioSignals):
        amplitudeThreshold = np.percentile(audio, 0.3)
        leftIndex = np.where(audio>=amplitudeThreshold)[0][0] # Grab the first index above this threshold
        rightIndex = np.where(audio>=amplitudeThreshold)[0][-1] # Grab the last index above this threshold
        audioSignals[i] = np.resize(audioSignals[i], [1,rightIndex-leftIndex]) # Update the previous array in place with a decreased size of the original
    return audioSignals

def preprocessAudio(audioFilePaths):
     audioData = [read(audioFile) for audioFile in audioFilePaths] # Read in the audioData
     audioRates = np.array([audio[0] for audio in audioData]) # Grab the sampling rate
     audioSignals = np.array([audio[1] for audio in audioData]) # Grab the signals themselves
     # Take the audio, convert it to mono, then filter it and then clip it
     for preprocess in [convertToMono, filterNoise, clipAudio]: audioSignals = preprocess(audioSignals)
     return audioRates, audioSignals

def frequencyTransform(audioSignals, audioRates):
    frequencyData = [ [] , [] ]
    for i, audio in enumerate(audioSignals):
        freqs, mag = computeFFTMag(audioSignals[i], audioRates[i])
        frequencyData[0].append(freqs)
        frequencyData[1].append(mag)
    frequencies = np.array([np.array(freqs) for freqs in frequencyData[0]])
    frequencyMags = np.array([np.array(mags) for mags in frequencyData[1]])
    return frequencies, frequencyMags

def extractFeatures(audioSignals, audioRates):
    audioFrequencies, audioMagnitudes = frequencyTransform(audioSignals, audioRates)
    meanAmps = []
    stdAmps = []
    audioLengths = []
    lowestFrequencies = []
    highestFrequencies = []
    meanMags = []
    stdMags = []
    skewMags = []
    kurtosisMags = []
    for i, audio in enumerate(audioSignals):
        meanAmps.append(np.mean(audio))
        stdAmps.append(np.std(audio))
        audioLengths.append(audio.size/audioRates[i])
    for i, audioMag in enumerate(audioMagnitudes):
        magThreshold = np.percentile(audioMag, 0.3)
        leftIndex = np.where(audioMag>=magThreshold)[0][0] # Grab the first index above this threshold
        rightIndex = np.where(audioMag>=magThreshold)[0][-1] # Grab the last index above this threshold
        lowestFrequencies.append(audioFrequencies[i][leftIndex])
        highestFrequencies.append(audioFrequencies[i][rightIndex])
        meanMags.append(np.mean(audioMag))
        stdMags.append(np.std(audioMag))
        skewMags.append(stats.skew(audioMag))
        kurtosisMags.append(stats.kurtosis(audioMag))
    featureData = np.array(meanAmps).reshape(-1,1)
    for feature in [stdAmps, audioLengths, lowestFrequencies, highestFrequencies,
                    meanMags, stdMags, skewMags, kurtosisMags]:
                    featureData = np.hstack([featureData, np.array(feature).reshape(-1,1)])
    return featureData



if __name__ == "__main__":
    audioFilePaths = [sys.path[0] + "/AudioData/Alex/" + filename for filename in os.listdir(sys.path[0] +"/AudioData/Alex/")]
    audioRates, audioSignals = preprocessAudio(audioFilePaths)
    audioFeatures = extractFeatures(audioSignals, audioRates)
