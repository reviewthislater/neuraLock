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
torch.manual_seed(1)


# https://pytorch.org/docs/stable/nn.html
class VoiceRecognizer(nn.Module):

    def __init__(self, numClasses, validUsers, names=[], optimizer=None, criterion=nn.CrossEntropyLoss(),
                    savedModelPath="voiceModel.bin",
                    audioDirectory="AudioData", unlockPhrase=""):
        super(VoiceRecognizer, self).__init__() # calling nn.module.__init__()
        self.dtype = torch.float
        self.device = torch.device("cpu")
        self.savedModelPath = savedModelPath
        self.names = names
        self.criterion = criterion
        self.optimizer = None #optimizer generally needs VoiceCNNInstance.parameters to initialize
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
            nn.MaxPool1d(kernel_size=4,stride=2)
        )
        self.fc = nn.Linear(285, numClasses)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def loadModel(self): self.load_state_dict(torch.load(self.savedModelPath))

    def saveModel(self): torch.save(self.state_dict(), self.savedModelPath)

    def train(self, trainLoader=None, epochs=1000, debug=True):
        self.train = True
        totalSteps = len(trainLoader)
        for epoch in range(epochs):
                for i, (signals, labels) in enumerate(trainLoader):
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
        self.saveModel()

    def test(self, testLoader):
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
        self.train = False
        signal = torch.Tensor(signal).unsqueeze(1)
        signal = signal.to(self.device)
        with torch.no_grad():
            outputs = self.forward(signal)
            prediction = torch.sort(outputs.data)
            highestPrediction = prediction[0][0][-1].item()
            secHighestPrediction = prediction[0][0][-2].item()
            if highestPrediction > 0 and secHighestPrediction <= 0: confidence = 100
            elif highestPrediction > 0 and secHighestPrediction > 0: confidence = (1-(secHighestPrediction/highestPrediction))*100
            else: confidence = 0
            prediction = prediction[1][0][-1].item()
        return self.names[prediction] if confidence >= 70 else "Unknown", confidence


    def run(self):
        """
        output: string, predicted name

        Records audio to a temporary file
        Reads in that audio
        Process the audio
        Predict on the audio
        Return the prediction if its above the confidence level set in predict otherwise return unknown
        """
        audioRate = np.array([44100])
        for i in range(3):
            wordRec = False
            # Turn on red led
            utilities.recordAudioToFile(".tempRecording", recordlength=5, rate=audioRate.item(), chunksize=1024)
            # Turn on orange led
            signal = np.array([read(".tempRecording.wav")[1]])
            if self.unlockPhrase != "": wordRec = self.recognizeWord() == self.unlockPhrase
            os.remove(".tempRecording") # this will prevent of previous audio data
            freqSignal = self.preprocessAudio(signal, audioRate)
            prediction, confidence = self.predict(freqSignal)
            if prediction in self.validUsers and self.unlockPhrase == "": return prediction
            elif prediction in self.validUsers and self.unlockPhrase != "" and wordRec: return prediction
        #turn off orange led and turn on red led
        return "Unknown"

    def recognizeWord(self):
        r = sr.Recognizer()
        with sr.AudioFile(".tempRecording.wav") as source: audio = r.record(source)
        try:
            return r.recognize_sphinx(audio) == self.unlockPhrase
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
            return False
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))
            return False


    def preprocessAudio(self, signals, rates, loader=False, targets=None):
        signals = utilities.convertToMono(signals)
        frequencies, frequencyMags = utilities.frequencyTransform(signals, rates)
        frequencyMags.tolist()
        if loader:
            if targets is not None: return self.createDataLoader(frequencyMags, targets)
            else: raise Exception("Cannot create DataLoader object without the target vector")
        else: return frequencyMags


    def createDataLoader(self, signals, targets):
        npTargets = []
        for target in targets: npTargets.append(np.array(target, dtype="long"))
        signalTensor = torch.stack([torch.Tensor(i) for i in signals])
        targetTensor = torch.stack([torch.Tensor(i) for i in npTargets])
        signalDataset = utils.TensorDataset(signalTensor, targetTensor)
        return utils.DataLoader(dataset=signalDataset, batch_size=5)
