#!/usr/bin/env python3
import numpy as np
import sys
import os
from scipy.io.wavfile import read
import speech_recognition as sr
from .utilities import utilities
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils
torch.manual_seed(1) # Keep randomness consistent through different program executions (see how modifications alter the results without worrying about how randomness played a role)

# https://pytorch.org/docs/stable/nn.html
class VoiceRecognizer(nn.Module):

    def __init__(self, numClasses:int, validUsers:list, names:list=[], optimizer:None=None, criterion:nn.Module=nn.CrossEntropyLoss(),
                    savedModelPath:str="voiceModel.bin", audioDirectory:str="AudioData", unlockPhrase:str=""):
        """
        input numClasses: integer for the number of classes for classification
        input validUsers: list of strings of valid users for unlocking the safe
        input optimizer: optimizer used for training (don't set this here set
                         this after creating the object and before calling trainNetwork)
        input criterion: criterion for the loss function
        input savedModelPath: path where to save/load the model to/from
        input audioDirectory: path where the folders containing training audio can be found
        input unlockPhrase: recognized word must match this phrase, if the unlockPhrase is not set the
                            recognizer will not utilize this authentication method
        """
        super(VoiceRecognizer, self).__init__() # calling nn.module.__init__()
        self.dtype = torch.float
        self.device = torch.device("cpu")
        self.savedModelPath = savedModelPath
        self.names = names
        self.criterion = criterion
        self.optimizer = None # optimizer generally needs VoiceCNNInstance.parameters to initialize
        self.unlockPhrase = unlockPhrase
        self.validUsers = validUsers

        self.layer1 = nn.Sequential(
            nn.Conv1d(1,10,kernel_size=3,stride=1,padding=2),
            nn.InstanceNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(10,70,kernel_size=3,stride=1,padding=2),
            nn.BatchNorm1d(70),
            nn.Softshrink(),
            nn.Dropout(),
            nn.AvgPool1d(kernel_size=2,stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(70,3,kernel_size=5,stride=1,padding=2),
            nn.InstanceNorm1d(3),
            nn.PReLU(),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size=4,stride=2)
        )
        self.fc = nn.Linear(285, numClasses)

    def forward(self, x):
        """
        input x: single or multiple signals in a torch Tensor

        output out: torchTensor containing index and prediction information
                    of varying size depending on how many signals were input
        Pass the signal through the network and return the output
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def loadModel(self):
        """
        Loads the model weights from the savedModelPath
        """
        self.load_state_dict(torch.load(self.savedModelPath))

    def saveModel(self):
        """
        Saves the model weights to the savedModelPath
        """
        torch.save(self.state_dict(), self.savedModelPath)

    def __log(self, logger, loss, epoch, i, accuracy):
        """
        input logger: logger object
        input loss: loss from the criterion of the training
        input epoch: number for current epoch of training
        input i: Step number for this epoch
        input accuracy: Accuracy of last batch size
        Logs parameters during a training run for use with tensorboard
        """
        # 1. Log scalar values (scalar summary)
        info = { 'loss': loss.item(), 'accuracy': accuracy.item() }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch*i+i+1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch*i+i+1)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch*i+i+1)


    def trainNetwork(self, trainLoader=None, epochs=1000, debug=True, log=False):
        """
        input trainLoader: DataLoader object either manually created or from using createDataLoader for training set
        input epochs: number of times to pass over the dataset
        input debug: boolean for debug output

        Trains the network and saves the model to savedModelPath.
        If debug is set to true than it will output progress and also perform logging for viewing in tensorboardX
        """
        if log:
            from logger import Logger # only import here since it wont be used anywhere else and only under debug condition
            logger = Logger('./logs') # create the logger object

        self.train = True # enter training mode, this changes how certain modules will behave in the layers ex. batchnorm1d
        totalSteps = len(trainLoader) # Determine total number of training steps in the batch
        for epoch in range(epochs):
                for i, (signals, labels) in enumerate(trainLoader):
                    if i == 0 and log: logger.logNetwork(self) #
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    signals = signals.unsqueeze_(1) # 1 input channel (frequency Mags)

                    # Forward pass
                    outputs = self.forward(signals)
                    loss = self.criterion(outputs,labels.to(torch.long))
                    # Bakcwards and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if debug and (i+1) % 5 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                                .format(epoch+1, epochs, i+1, totalSteps, loss.item()))
                    if log:
                        # Compute accuracy
                        _, argmax = torch.max(outputs, 1)
                        accuracy = (labels.to(torch.long) == argmax.squeeze()).float().mean()
                        self.__log(logger,loss, epoch, i, accuracy)

        self.saveModel()

    def test(self, testLoader):
        """
        input testLoader: Dataloader object containing the test audio and targets

        Goes through the test set and counts number of correct predictions
        Prints accuracy to terminal
        """
        self.train = False
        with torch.no_grad():
            correct = 0
            total = 0
            for signals, labels in testLoader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                signals.unsqueeze_(1)
                outputs = self.forward(signals)
                _, predicted = torch.max(outputs.data, 1)
                labels = labels.to(torch.long)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
            print('Test Accuracy of the model on the {0} test signals: {1} %'.format(total, 100 * correct / total))


    def predict(self, signal):
        """
        input signal: torch tensor of shape (batchsize, numFrequencyMags)

        Take in the signal and add a dummy axis
        Run the signal through the network
        Grab the two highest predictions
        Grab confidence level for highest prediction
        If predicted user is valid user and above confidence threshold return users name and confidence
        Else return "Unknown" and confidence
        """
        self.train = False # set training mode to false (will change behavior of some parts of the network like BatchNorm1d and Dropout)
        signal = torch.Tensor(signal).unsqueeze(1) # add the dummy axis
        signal = signal.to(self.device) # put the signal onto the device
        with torch.no_grad(): # don't autograd when predicting
            outputs = self.forward(signal)
            prediction = torch.sort(outputs.data)
            if prediction[0][0][0].item() >= 0: norm = -prediction[0][0][0].item() # figure out how to make smallest num 0
            else: norm = prediction[0][0][0].item()
            highestPrediction = prediction[0][0][-1].item() + norm # grab highest prediction and normalize
            secHighestPrediction = prediction[0][0][-2].item() + norm # grab second highest prediction and normalize
            confidence = (1-(secHighestPrediction/highestPrediction))*100 # using 1 because the highest prediction should be normalized to 1
            prediction = prediction[1][0][-1].item() # grab the target number for the most confident prediction
        if confidence >= 90 and self.names[prediction] in self.validUsers:
            return self.names[prediction], confidence
        else: return  "Unknown", confidence


    def run(self):
        """
        output: string, predicted name

        Attempt 3 times:
            Records audio to a temporary file
            Reads in that audio
            Process the audio
            Predict on the audio
            Return the prediction if its above the confidence level set in predict
        Return "Unknown"
        """
        audioRate = np.array([48000]) # we will use this data rate for audio recording
        for i in range(3):
            wordRec = False # set word wecognized to false at the start of each run
            # Turn on red led
            utilities.recordAudioToFile(".tempRecording", recordlength=5, rate=audioRate.item(), chunksize=512) # record the audio
            # Turn on orange led
            signal = np.array([read(".tempRecording.wav")[1]]) # read in the data
            if self.unlockPhrase != "": wordRec = self.recognizeWord() # if an unlock phrase has been set check if recognized word matches unlockPhrase
            os.remove(".tempRecording.wav") # this will prevent replaying of previous audio data
            freqSignal = self.preprocessAudio(signal, audioRate) # clean audio and convert to frequency domain
            prediction, confidence = self.predict(freqSignal) # predict on the frequency magnitudes of the audio
            if prediction in self.validUsers and self.unlockPhrase == "": return prediction
            elif prediction in self.validUsers and self.unlockPhrase != "" and wordRec: return prediction
        #turn off orange led and turn on red led
        return "Unknown"

    def recognizeWord(self):
        """
        output isWord: true if recognized word matches unlockPhrase otherwise false

        Create recognizer object
        Load in the audio from the temp file path
        Recognize audio and return isWord
        """
        r = sr.Recognizer()
        with sr.AudioFile(".tempRecording.wav") as source: audio = r.record(source) # grab the audio
        try:
            return r.recognize_sphinx(audio) == self.unlockPhrase # recognize audio and compare to unlockPhrase
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
            return False
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))
            return False


    def preprocessAudio(self, signals, rates, loader=False, targets=None):
        """
        input signals: 1d numpy array of 1d numpy arrays containing the signals
        input rates: 1d numpy array of 1d numpy arrays containing the signal rates
        input loader: boolean as to whether the output should be a DataLoader of the processed audio
        input targets: 1d np array of targets. Only needed if loader is true
        """
        signals = utilities.convertToMono(signals) # All signals are converted to mono to speed up CNN
        frequencies, frequencyMags = utilities.frequencyTransform(signals, rates) # CNN recieves frequency domain signals
        frequencyMags.tolist() # Create data loader takes in a list (not sure if this was actually doing anything because it shouldn't be in place, test later to see if it can be removed)
        if loader: # create a dataloader if a targets array was passed to method else just return the freqMags
            if targets is not None: return self.createDataLoader(frequencyMags, targets)
            else: raise Exception("Cannot create DataLoader object without the target vector")
        else: return frequencyMags


    def createDataLoader(self, signals, targets):
        """
        input signals: 1d numpy array of 1d numpy arrays containing the preprocessed signals
        input targets: 1d numpy array of scalar values containing the respective targets

        output: Dataloader object containing the signals and targets information with batchSize of 5

        Manipulate targets datatype
        Create tensors of signals and targets data
        Create tensor dataset with these signals and targets
        Create a dataloader with the resulting tensor dataset
        """
        npTargets = map(np.array, targets) # convert inner target scalars to single element numpy arrays
        signalTensor = torch.stack([torch.Tensor(i) for i in signals]) # create a torch tensor from numpy array of numpy arrays
        targetTensor = torch.stack([torch.Tensor(i) for i in npTargets]) # create a torch tensor from the numpy array of numpy arrays
        signalDataset = utils.TensorDataset(signalTensor, targetTensor)
        return utils.DataLoader(dataset=signalDataset, batch_size=5)
