#!/usr/bin/env python3
from Recognizers.VoiceRecognizer import VoiceRecognizer
from Recognizers.FaceRecognizer import FaceRecognizer
from Recognizers.utilities import utilities
from scipy.io.wavfile import read
from time import sleep
import numpy as np
import sys
import argparse
import torch.optim as optim
from gpiozero import Servo, Button

servo = Servo(17, frame_width=20/100)
lockButton = Button(18)
servo.value = -1
sleep(1)
servo.value = None
sleep(1)

def unlock(state):
	servo.value = 1
	sleep(2)
	servo.value=None
	state = "unlocked"
	print("State: ", state)
	return state

def lock(state):
	servo.value = -1
	sleep(2)
	servo.value = None
	state = "locked"
	print("State: ",state)
	return state


def driver():
	ap = argparse.ArgumentParser(
	prog="neuraLock",
	description="DIY safe utilizing facial and vocal recognition",
	epilog="By: Alex Epstein https://github.com/alexanderepstein\n&\nDave Wigley https://github.com/davidwigley")
	ap.add_argument("-ad", "--audio_dataset", help="path to audio training data", required=True)
	ap.add_argument("-fd", "--face_dataset",  help="path to face training data", required=True)
	ap.add_argument("-vm", "--voice_model", help="path to serialized voice model", required=True)
	ap.add_argument("-fm", "--face_model", help="path to serialized face model", required=True)
	ap.add_argument("-lv", "--load_voice_model", help="boolean for loading voice model", default=False, action="store_true")
	ap.add_argument("-lf", "--load_face_model", help="boolean for loading face model", default=False, action="store_true")
	ap.add_argument("-u", "--unlock_phrase", help="unlock phrase", default="")
	ap.add_argument("-am", "--authentication_model", choices=["both", "face", "voice"], default="both", help="choose the authentication model")
	ap.add_argument('-vu', '--valid_users', nargs='+', help="users the safe should open for", required=True)
	args = ap.parse_args()

	validUsers = args.valid_users
	state = "locked"

	if args.authentication_model == "both" or args.authentication_model == "face":
		fr = FaceRecognizer(validUsers ,detectionMethod="hog", encodings=args.face_model, imageDirectory=args.face_dataset)
		if args.load_face_model: fr.loadEncodings()
		else args.load_face_model: fr.encodeImages()
		vs, fps = fr.setup()

	if args.authentication_model == "both" or args.authentication_model == "voice":
		_,_, trainNames = utilities.getFilePaths(args.audio_dataset)
		vr = VoiceRecognizer(len(trainNames), validUsers, unlockPhrase=args.unlock_phrase,
							savedModelPath=args.voice_model, audioDirectory=args.audio_dataset, names=trainNames)
		vr.optimizer = optim.Adam(vr.parameters(), lr=0.001)
		if args.load_voice_model: vr.loadModel()
		else:
			trainPaths, trainLabels, trainNames = utilities.getFilePaths(directory=args.audio_dataset)
			trainData = [read(audioFile) for audioFile in trainPaths] # Read in the audioData
			trainRates = np.array([audio[0] for audio in trainData]) # Grab the sampling rate
			trainSignals = np.array([audio[1] for audio in trainData]) # Grab the signals themselves
			trainLoader = vr.preprocessAudio(trainSignals, trainRates, loader=True, targets=trainLabels)
			vr.trainNetwork(trainLoader, epochs=25, debug=True)

	if args.authentication_model == "both":
		bothAuthenticated = False
		while True:
			while not bothAuthenticated:
                # Turn on orange led for video
				recognizedFaces = fr.run(vs, fps, debug=False)
                # Turn on green led for video
				recognizedVoice = [vr.run()]
                #if recognizedVoice != "Unknown": Turn on green led for audio
                #else: continue
				userSet = set(recognizedFaces) & set(recognizedVoice) & set(validUsers) # make sure we are returning the same valid user
				if len(userSet) == 1:
					# status light goes green
					bothAuthenticated = True
					state = unlock(state)

			while state == "unlocked":
				if lockButton.is_pressed:
					# status light goes red
					state = lock(state)
					bothAuthenticated = False
					vs, fps = fr.setup()
					sleep(4)

	elif args.authentication_model == "voice":
		# all face leds go off
		voiceAuthenticated = False
		while True:
			while not voiceAuthenticated:
				recognizedVoice = [vr.run()]
				userSet = set(recognizedVoice) & set(validUsers)

				if len(userSet) == 1:
					#green led for voice goes on
					# status light goes green
					voiceAuthenticated = True
					state = unlock(state)


			while state == "unlocked":
				if lockButton.is_pressed:
					# status light goes red
					state = lock(state)
					voiceAuthenticated = False


	elif args.authentication_model == "face":
		# all voice leds go off
		faceAuthenticated = False
		while True:
			while not faceAuthenticated:
                # Turn on orange led for video
				recognizedFaces = fr.run(vs, fps, debug=False)
                # Turn on green led for video
				if len(recognizedFaces) == 1:
					# status light goes green
					faceAuthenticated = True
					state = unlock(state)

			while state == "unlocked":
				if lockButton.is_pressed:
					state = lock(state)
					faceAuthenticated = False
					vs, fps = fr.setup()
					sleep(4)

if __name__ == "__main__": driver()

#./controller.py -ad "AudioData" -fd "ImageData" -vm "voiceModel.bin" -fm "imageEncodings.bin" -lv -am both -vu Alex Dave
