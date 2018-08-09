import sys
import pyaudio
import seaborn as sns
import numpy as np
import pyttsx3
import wave
import os
from scipy.signal import welch
import warnings
import cv2

class utilities(object):

    @staticmethod
    def speak(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.setProperty('rate',120)  #120 words per minute
        engine.setProperty('volume',0.9)
        engine.runAndWait()
        engine.stop()

    @staticmethod
    def recordAudioToFile(filename, recordlength=3, rate=44100, channels=2, chunksize=1024, format=pyaudio.paInt32):
        CHUNK = chunksize
        FORMAT = format
        CHANNELS = channels
        RATE = rate
        RECORD_SECONDS = recordlength
        WAVE_OUTPUT_FILENAME = filename + ".wav"
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("[INFO] recording")
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        print("[INFO] done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    @staticmethod
    def getImageData(rootDirectory=""):
        detector = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        images = [] # images with exactly one face
        imagePaths = [] # paths to all the images
        validPaths = [] # paths to images with only 1 face detected
        imageDirectory = rootDirectory # looks for the root directory relative to where the script was run because python needs absolute paths (add to sys.path for relative paths)
		# Grab image paths
        for label in sorted(os.listdir(imageDirectory)): # for each folder in the image directory
            for image in os.listdir(imageDirectory + "/" + label): # for each file in each folder
                imagePaths.append(imageDirectory + "/" + label + "/" + image) #add to the image pathh
		# Determine valid images (exactly one face needs to be detected to be a valid image)
        for imagePath in imagePaths:
            original = cv2.imread(imagePath)
            image = original #if opencv else original
            faces = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # change scale factor here if faces aren't being detected, it relates to how far the person is from the camera
            if len(faces) != 1:
                warnings.warn("Not using an image from the train data set because exactly one face was not detected")
            else: validPaths.append(imagePath)
        return validPaths


    @staticmethod
    def frequencyTransform(audioSignals, audioRates):
        """
        input audioSignals: numpy array of 1d numpy arrays containing sound data
        input audioRates: numpy array of 1d scalars with the rates of the corresponding audio
        output frequencies: numpy array of 1d numpy arrays containing frequency data
        output frequencyMags: numpy array of 1d numpy arrays containing frequency magnitude data

        Create list of two lists
        Go through each signal and compute the fft mag
        Append the data to the corresponding inner list
        Mend result to output data type
        """
        # List containing two python lists, frequencyData[0] is the frequency, frequencyData[1] is the magnitude
        frequencyData = [ [] , [] ]
        for i, audio in enumerate(audioSignals): # audioSignals can contain many samples so we need to iterate
            freqs, mag = welch(audioSignals[i], fs=audioRates[i], nperseg=(1024+512)) # fft and magnitude
            frequencyData[0].append(freqs)
            frequencyData[1].append(mag)
        frequencies = np.array([freqs for freqs in frequencyData[0]]) # For each sample make a numpy array of the frequencies and wrap it in a numpy array
        frequencyMags = np.array([mags for mags in frequencyData[1]])#  For each sample make a numpy array of the magnitudes and wrap it in a numpy array
        return frequencies, frequencyMags


    @staticmethod
    def getFilePaths(directory=None):
        filePaths = []
        labels = []
        names = []
        for i, label in enumerate(sorted(os.listdir(directory))): # for each folder in the audio directory, need to sort because diff platforms return in diff order
            names.append(label) # grab the names of the people since all scikit models classify on numbers not strings
            for filename in os.listdir(directory + "/" + label): # for each file in each folder
                filePaths.append(directory + "/" + label + "/" + filename) #add to the audio paths
                labels.append(i) # append the integer label
        return filePaths, labels, names

    @staticmethod
    def convertToMono(audioSignals):
        for i, audio in enumerate(audioSignals):
            if len(audioSignals[i].shape) != 1:
                if audioSignals.shape[0] == 1: return np.array((audio.sum(axis=1) / 2).reshape(1,-1))
                audioSignals[i] = (audio.sum(axis=1) / 2) # Convert data to mono and return it
        return audioSignals
