import sys
import pyaudio
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
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
    def computeSteps(x, y=None, z=None, fs=50, plot=False):
        """ Estimate the number of steps taken using the accel data
            In:
            'x': x data
            'y': optional y data
            'z': optional z data
            'fs': sample frequency
            'plot': boolean to indicate whether the function should plot the results
            Out: 'steps': number of steps/peaks taken in the signal """
        secsbetweensteps = 0.1
        t = np.array([0.025 * x for x in range(x.size)])
        mag = (x**2+y**2+z**2)**(1/2) if y is not None and z is not None else np.abs(x)
        magdt = mag - np.mean(mag)
        pd = secsbetweensteps*fs // 1
        ph = np.std(magdt)
        peaks = utilities.detect_peaks(magdt, mph=ph, mpd=pd, edge='rising')
        steps = magdt[peaks.tolist()]
        if plot:
            plt.plot(t, magdt, label="magdt")
            plt.plot(t[peaks.tolist()], steps, linestyle='none', marker = 'o', label="steps")
            plt.legend(loc=1)
            plt.xlabel("Time (s)")
            plt.ylabel("Mag (Acceleration) (ms/s^2)")
            plt.title("# of steps found is %s" % len(steps))
            plt.show()
        return len(steps)

    @staticmethod
    def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):

        """Detect peaks in data based on their amplitude and other features.

        Parameters
        ----------
        x : 1D array_like
            data.
        mph : {None, number}, optional (default = None)
            detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            detect peaks that are at least separated by minimum peak distance (in
            number of data).
        threshold : positive number, optional (default = 0)
            detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
            for a flat peak, keep only the rising edge ('rising'), only the
            falling edge ('falling'), both edges ('both'), or don't detect a
            flat peak (None).
        kpsh : bool, optional (default = False)
            keep peaks with same height even if they are closer than `mpd`.
        valley : bool, optional (default = False)
            if True (1), detect valleys (local minima) instead of peaks.

        Returns
        -------
        ind : 1D array_like
            indeces of the peaks in `x`.

        Notes
        -----
        The detection of valleys instead of peaks is performed internally by simply
        negating the data: `ind_valleys = detect_peaks(-x)`

        The function can handle NaN's

        See this IPython Notebook [1]_.

        References
        ----------
        .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

        Examples
        --------
        >>> from detect_peaks import detect_peaks
        >>> x = np.random.randn(100)
        >>> x[60:81] = np.nan
        >>> # detect all peaks and plot data
        >>> ind = detect_peaks(x, show=True)
        >>> print(ind)

        >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
        >>> # set minimum peak height = 0 and minimum peak distance = 20
        >>> detect_peaks(x, mph=0, mpd=20, show=True)

        >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
        >>> # set minimum peak distance = 2
        >>> detect_peaks(x, mpd=2, show=True)

        >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
        >>> # detection of valleys instead of peaks
        >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

        >>> x = [0, 1, 1, 0, 1, 1, 0]
        >>> # detect both edges
        >>> detect_peaks(x, edge='both', show=True)

        >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
        >>> # set threshold = 2
        >>> detect_peaks(x, threshold = 2, show=True)
        """

        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])
        return ind

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
        if plot: utilities.__plotFFTMag(fy, sigMag)
        if debug >= 1: print("Most energy detected at",fy[np.argmax(sigMag)] ,"Hz") # print the Frequency where the max mag response occurs
        return fy, sigMag # returns the frquency and the magnitude response at each frequency

    @staticmethod
    def getImageData(rootDirectory=""):
        detector = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        images = [] # images with exactly one face
        imagePaths = [] # paths to all the images
        validPaths = [] # paths to images with only 1 face detected
        imageDirectory = rootDirectory # looks for the root directory relative to where the script was run because python needs absolute paths (add to sys.path for relative paths)
		# Grab image paths
        for label in os.listdir(imageDirectory): # for each folder in the image directory
            for image in os.listdir(imageDirectory + "/" + label): # for each file in each folder
                imagePaths.append(imageDirectory + "/" + label + "/" + image) #add to the image pathh
		# Determine valid images (exactly one face needs to be detected to be a valid image)
        for imagePath in imagePaths:
            original = cv2.imread(imagePath)
            image = original #if opencv else original
            faces = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # change scale factor here if faces aren't being detected, it relates to how far the person is from the camera
            if len(faces) != 1: warnings.warn("Not using an image from the train data set because exactly one face was not detected")
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
