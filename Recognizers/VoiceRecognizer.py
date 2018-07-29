#!/usr/bin/env python3
import numpy as np
import wave
import sys
import os
from scipy.io.wavfile import read
from scipy import stats
from scipy.signal import lfilter
from sklearn.preprocessing import normalize
import speech_recognition as sr
import pickle
from array import array
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from .utilities import utilities


class VoiceRecognizer(object):

    def __init__(self, model=None, names=[], unlockPhrase="unlock", savedModelPath="voiceModel.bin", audioDirectory="AudioData"):
        self.model = model
        self.names = names
        self.unlockPhrase = unlockPhrase
        self.savedModelPath = savedModelPath
        self.audioDirectory = audioDirectory

    def predict(self, signal, rate):
        if self.model is None: raise ValueError("The model has not been fit yet.")
        signal = [np.array(self.__preprocessAudio(None,signal)[1])]
        signalFeatures = self.__extractFeatures(signal, rate)
        predictionProbabilities = self.model.predict_proba(signalFeatures)
        confidence = np.amax(predictionProbabilities)
        prediction = np.argmax(predictionProbabilities)
        return prediction, confidence

    def __getAudioFilePaths(self, audioDirectory=None):
        if audioDirectory is None: audioDirectory = self.audioDirectory
        audioFilePaths = []
        labels = []
        for i, label in enumerate(os.listdir(audioDirectory)): # for each folder in the audio directory
            self.names.append(label) # grab the names of the people since all scikit models classify on numbers not strings
            for audio in os.listdir(audioDirectory + "/" + label): # for each file in each folder
                audioFilePaths.append(audioDirectory + "/" + label + "/" + audio) #add to the audio paths
                labels.append(i) # append the integer label
        return audioFilePaths, labels

    def train(self):
        self.names = []
        audioFilePaths, targets = self.__getAudioFilePaths()
        audioRates, audioSignals = self.__preprocessAudio(audioFilePaths)
        audioFeatures = self.__extractFeatures(audioSignals, audioRates)
        self.model = KNeighborsClassifier(n_jobs=-1, n_neighbors=3)
        self.model.fit(audioFeatures, targets)
        self.saveModel()

    def saveModel(self):
        pickleFile = open(self.savedModelPath, "wb")
        data = {"model": self.model, "names": self.names, "unlockPhrase": self.unlockPhrase}
        pickleFile.write(pickle.dumps(data))
        pickleFile.close()

    def loadModel(self):
        pickleFile = open(self.savedModelPath, "rb")
        data = pickle.loads(pickleFile.read())
        self.model = data["model"]
        self.names = data["names"]
        self.unlockPhrase = data["unlockPhrase"]
        pickleFile.close()


    def __filterNoise(self, audioSignals):
        n = 15  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        for i, audio in enumerate(audioSignals): audioSignals[i] = lfilter(b,a,audio)
        return audioSignals

    def __convertToMono(self, audioSignals):
        if audioSignals[0].shape[1] == 1: return audioSignals # Data is already mono so lets just return it
        elif audioSignals[0].shape[1] == 2:
            for i, audio in enumerate(audioSignals):
                if audioSignals.shape[0] == 1: return np.array((audio.sum(axis=1) / 2).reshape(1,-1))
                audioSignals[i] = (audio.sum(axis=1) / 2) # Convert data to mono and return it
            return audioSignals
        else: raise Exception("Whoah you discovered %d dimensional audio" % audioSignals[0].shape[1])

    def __clipAudio(self, audioSignals):
        for i, audio in enumerate(audioSignals):
            amplitudeThreshold = np.percentile(audio, 0.3)
            ampCondition =  np.where(audio>=amplitudeThreshold)[0] # Going to be used twice so store it for now
            leftIndex = ampCondition[0] # Grab the first index above this threshold
            rightIndex = ampCondition[-1] # Grab the last index above this threshold
            if audioSignals.shape[0] == 1: return np.array(audioSignals[i][leftIndex:rightIndex])
            audioSignals[i] = audioSignals[i][leftIndex:rightIndex] # Update the previous array in place with a decreased size of the original

        return audioSignals

    def __preprocessAudio(self, audioFilePaths=None, audioSignals=None):
         audioRates = [] # This will remain empty if we pass in the signal because we shoudl already know the rate
         if audioSignals is None:
             audioData = [read(audioFile) for audioFile in audioFilePaths] # Read in the audioData
             audioRates = np.array([audio[0] for audio in audioData]) # Grab the sampling rate
             audioSignals = np.array([audio[1] for audio in audioData]) # Grab the signals themselves
         else: audioSignals = np.array([audioSignals]) # preprocessing is built to take in an input of a numpy array containing numpy arrays
         # Take the audio, convert it to mono, then filter it and then clip it
         audioSignals = self.__convertToMono(audioSignals)
         audioSignals = self.__filterNoise(audioSignals)
         audioSignals = self.__clipAudio(audioSignals)
         return audioRates, audioSignals



    def __extractFeatures(self,audioSignals, audioRates):
        """
        input audioSignals: numpy array of 1d numpy arrays containing sound data
        input audioRates: numpy array of 1d scalars with the rates of the corresponding audio
        output featureData: a numpy array of shape (samples, numFeatures) containing data for a model to fit to

        Extracts features from the sound data
        Transforms the signals to the  domain
        Extracts features from the frequency domain sound data
        Returns the extracted features for each sample as a numpy matrix
        """
        audioFrequencies, audioMagnitudes = utilities.frequencyTransform(audioSignals, audioRates)
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
            audioLengths.append(audio.size/audioRates[i]) # Useful if implementing unlock keyphrase
        for i, audioMag in enumerate(audioMagnitudes):
            magThreshold = np.percentile(audioMag, 0.3) # Grab the nth percentile for threshold
            magCondition = np.where(audioMag>=magThreshold)[0] #
            leftIndex = magCondition[0] # Grab the first index above this threshold
            rightIndex = magCondition[-1] # Grab the last index above this threshold
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

    def recognizeWord(self, filename):
        r = sr.Recognizer()
        with sr.AudioFile(sys.path[0] + "/" + filename + ".wav") as source:
            audio = r.record(source)
        try:
            return r.recognize_sphinx(audio)
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
            return ""
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))
            return ""
