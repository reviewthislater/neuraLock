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
import speech_recognition as sr
import pickle
from array import array
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from matplotlib.colors import ListedColormap
from sklearn import neighbors

class VoiceRecognizer(object):

    def __init__(self, model=None, names=None, unlockPhrase="unlock", savedModelName="model.bin"):
        self.model = model
        self.names = []
        self.unlockPhrase = unlockPhrase
        self.savedModelName = savedModelName

    def predict(self, signal, rate):
        if self.model is None: raise ValueError("The model has not been fit yet.")
        signal = [np.array(self.__preprocessAudio(None,signal)[1])]
        signalFeatures = self.__extractFeatures(signal, rate)
        predictionProbabilities = self.model.predict_proba(signalFeatures)
        confidence = np.amax(predictionProbabilities)
        prediction = np.argmax(predictionProbabilities)
        return prediction, confidence

    def __getAudioFilePaths(self, audioDirectory="AudioData"):
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
        audioFilePaths, targets = self.__getAudioFilePaths(audioDirectory="AudioData")
        audioRates, audioSignals = self.__preprocessAudio(audioFilePaths)
        audioFeatures = self.__extractFeatures(audioSignals, audioRates)
        VoiceRecognizer.plotkNN(audioFeatures[:,3:5], targets, 3)
        exit()
        self.model = KNeighborsClassifier(n_jobs=-1, n_neighbors=3)
        self.model.fit(audioFeatures, targets)
        self.saveModel()
        """X_train, X_test, y_train, y_test = train_test_split(audioFeatures, targets, test_size=0.2, random_state=42)

        confidence = np.amax(self.model.predict_proba(X_test))
        predictions = self.model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print("Accuracy Score: ", acc)
        print(confidence)
        print(predictions)
        """

    @staticmethod
    def plotkNN(X, y, n_neighbors = 15):
        #Stolen from:
        # http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
        h = .1  # step size in the mesh

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X, y)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure(figsize=(50,100))
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                        edgecolor='k', s=80)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xlabel('Lowest Frequency')
            plt.ylabel('HighestFrequency')
            plt.title("3-Class classification (k = %i, weights = '%s')"
                      % (n_neighbors, weights))
        plt.show()

    def saveModel(self):
        pickleFile = open(self.savedModelName, "wb")
        pickle.dump(self.model, pickleFile)
        pickleFile.close()


    def loadModel(self):
        pickleFile = open(self.savedModelName, "rb")
        self.model = pickle.load(pickleFile)
        pickleFile.close()

    def recordAudioToFile(self,filename, recordlength=5, rate=44100, channels=1, chunksize=1024, format=pyaudio.paFloat32):
        CHUNK = chunksize
        FORMAT = format
        CHANNELS = channels
        RATE = rate
        RECORD_SECONDS = recordlength
        WAVE_OUTPUT_FILENAME = sys.path[0] + "/" + filename + ".wav"
        print(WAVE_OUTPUT_FILENAME)
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("* recording")
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
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

    @staticmethod
    def __plotFFTMag(frequency, sigMag, plotTitle="Frequency Response"):
        sns.set_style('darkgrid')
        plt.figure()
        plt.plot(frequency, sigMag)
        # label the axes
        plt.ylabel("|Freq(Signal)|")
        plt.xlabel("Freq [Hz]")
        plt.title(plotTitle) # label the title
        plt.grid()
        plt.show() # display the plot

    @staticmethod
    def __computeFFTMag(sig0, fs, plot=False, debug=0):
        ''' Given a signal and its sampling rate, plot the one-sided freq response
        In: sig0: 1D array in time domain
        In: fs: sampling rate of sig0

        Out: Mag response and corresponding freq
        '''
        n = len(sig0) if isinstance(sig0, list) else sig0.size
        sig0f = np.fft.rfft(sig0) # fft
        sig0f = sig0f/n # normalize
        sig0f1 = sig0f[range(n//2)] # 1-sided
        sigMag= np.abs(sig0f1) # Magnitude response
        fy = np.linspace(0, fs//(2), n//2) # freq axis
        if plot: VoiceRecognizer.__plotFFTMag(fy, sigMag)
        if debug >= 1: print("Most energy detected at",fy[np.argmax(sigMag)] ,"Hz") # print the Frequency where the max mag response occurs
        return fy, sigMag # returns the frquency and the magnitude response at each frequency


    def __getNoiseStatistics(self):
        filenames = sys.path[0] +"/AudioData/NoiseReduction/"
        statistics = {}
        for i,file in enumerate(os.listdir(filenames)):
            audioData = read(filenames+file)
            samplingRate = audioData[0]
            leftMono = audioData[1][0:,0]
            rightMono = audioData[1][0:,1]
            signal = audioData[1]#np.vstack([leftMono, rightMono])
            frequency, magnitude = VoiceRecognizer.__computeFFTMag(signal, samplingRate, plot=False)
            statistics["maxFrequency"] = frequency[np.where(frequency == np.argmax(magnitude))]
            statistics["meanAmplitude"] = np.mean(signal.flatten())
            statistics["meanFrequency"] = frequency[np.where(frequency == np.mean(magnitude))]
        return statistics

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

    def __frequencyTransform(self, audioSignals, audioRates):
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
            freqs, mag = self.__computeFFTMag(audioSignals[i], audioRates[i]) # fft and magnitude
            frequencyData[0].append(freqs)
            frequencyData[1].append(mag)
        frequencies = np.array([np.array(freqs) for freqs in frequencyData[0]]) # For each sample make a numpy array of the frequencies and wrap it in a numpy array
        frequencyMags = np.array([np.array(mags) for mags in frequencyData[1]])#  For each sample make a numpy array of the magnitudes and wrap it in a numpy array
        return frequencies, frequencyMags

    def __extractFeatures(self,audioSignals, audioRates):
        """
        input audioSignals: numpy array of 1d numpy arrays containing sound data
        input audioRates: numpy array of 1d scalars with the rates of the corresponding audio
        output featureData: a numpy array of shape (samples, numFeatures) containing data for a model to fit to

        Extracts features from the sound data
        Transforms the signals to the frequency domain
        Extracts features from the frequency domain sound data
        Returns the extracted features for each sample as a numpy matrix
        """
        audioFrequencies, audioMagnitudes = self.__frequencyTransform(audioSignals, audioRates)
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


if __name__ == "__main__":
    retrain = True
    vr = VoiceRecognizer()
    if retrain: vr.train()
    else: vr.loadModel()
    #self.recordAudioToFile("recorded", recordlength=3, rate=44100, channels=2, chunksize=1024, format=pyaudio.paFloat32)
    """r = sr.Recognizer()
    with sr.AudioFile(sys.path[0] + "/recorded.wav") as source:
        audio = r.record(source)

    try:
        print("Sphinx thinks you said " + r.recognize_sphinx(audio))
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))
        """
    testSignal = read(sys.path[0] + "/Clip-3.wav")[1]
    testRate = [44100]
    prediction, confidence = vr.predict(testSignal, testRate)
    print(vr.names, prediction)
    print("Prediction: " + vr.names[prediction] + "    Confidence: ", confidence)
    testSignal = read(sys.path[0] + "/tim.wav")[1]
    testRate = [44100]
    prediction, confidence = vr.predict(testSignal, testRate)
    print(vr.names, prediction)
    print("Prediction: " + vr.names[prediction] + "    Confidence: ", confidence)
