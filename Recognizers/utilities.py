import sys
import pyaudio
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import pyttsx3
import wave

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
    def recordAudioToFile(filename, recordlength=5, rate=44100, channels=2, chunksize=1024, format=pyaudio.paFloat32):
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
            data = stream.read(CHUNK)
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


    @staticmethod
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

    @staticmethod
    def computeFFTMag(sig0, fs, plot=False, debug=0):
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
            freqs, mag = utilities.computeFFTMag(audioSignals[i], audioRates[i]) # fft and magnitude
            frequencyData[0].append(freqs)
            frequencyData[1].append(mag)
        frequencies = np.array([np.array(freqs) for freqs in frequencyData[0]]) # For each sample make a numpy array of the frequencies and wrap it in a numpy array
        frequencyMags = np.array([np.array(mags) for mags in frequencyData[1]])#  For each sample make a numpy array of the magnitudes and wrap it in a numpy array
        return frequencies, frequencyMags
