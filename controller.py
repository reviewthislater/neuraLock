#!/usr/bin/env python3
from Recognizers.VoiceRecognizer import VoiceRecognizer
from Recognizers.FaceRecognizer import FaceRecognizer
from Recognizers.utilities import utilities
from scipy.io.wavfile import read
import numpy as np
import sys
import argparse

lockAngle = 180
unlockAngle = 90
state = "unlocked"

def unlock():
#    if state == "locked": moveServoToUnlockAngle
    state = "unlocked"
    print("State: ", state)
    return

def lock():
#    if state == "unlocked": moveServoToLockAngle
    state = "locked"
    print("State: ",state)
    return

def testVoiceRecognizer(vr):
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

def driver():
	ap = argparse.ArgumentParser(
	prog="neuraLock",
	description="DIY safe utilizing facial and vocal recognition",
	epilog="By: Alex Epstein https://github.com/alexanderepstein\n&\nDave Wigley https://github.com/davidwigley")
	ap.add_argument("-ad", "--audio_dataset", help="path to audio training data", required=True)
	ap.add_argument("-fd", "--face_dataset",  help="path to face training data", required=True)
	ap.add_argument("-vm", "--voice_model", help="path to serialized voice model", required=True)
	ap.add_argument("-fm", "--face_model", help="path to serialized face model", required=True)
	ap.add_argument("-lv", "--load_voice_model", help="boolean for loading voice model", action="store_true",)
	ap.add_argument("-lf", "--load_face_model", help="boolean for loading face model", action="store_true",)
	ap.add_argument("-u", "--unlock_phrase", help="unlock phrase", default="")
	ap.add_argument("-am", "--authentication_model", choices=["both", "face", "voice"], default="both", help="choose the authentication model")
	ap.add_argument('-vu', '--valid_users', nargs='+', help="users the safe should open for", required=True)
	args = ap.parse_args()

	validUsers = args.valid_users


	if args.authentication_model == "both" or args.authentication_model == "face":
		fr = FaceRecognizer(validUsers ,detectionMethod="hog", encodings=args.face_model)
		if not args.load_face_model: fr.encodeImages()
		data, vs, fps= fr.setup()

	if args.authentication_model == "both" or args.authentication_model == "voice":
		vr = VoiceRecognizer(unlockPhrase=args.unlock_phrase, savedModelPath=args.voice_model,
		audioDirectory=args.audio_dataset)
		if args.load_voice_model: vr.loadModel()
		else: vr.train()

	if args.authentication_model == "both":
		bothAuthenticated = False
		while True:
			while not bothAuthenticated:
                # Turn on orange led for video
				recognizedFaces = fr.run(data, vs, fps, debug=False)
                # Turn on green led for video
				recognizedVoice = [vr.run()]
                #if recognizedVoice != "Unknown": Turn on green led for audio
                #else: continue
				if args.unlock_phrase != "" and not vr.recognizeWord(): continue
				userSet = set(recognizedFaces) & set(recognizedVoice) & set(validUsers)
				if len(userSet) == 1:
					bothAuthenticated = True
					unlock()
			"""
            while state == "unlocked":
                if lock button pressed:
                    lock()
                    bothAuthenticated = False
			"""
	elif args.authentication_model == "voice":
		voiceAuthenticated = False
		while True:
			while not voiceAuthenticated:
				recognizedVoice = [vr.run()]
                #if recognizedVoice != "Unknown": Turn on green led for audio
                #else: continue
				if args.unlock_phrase != "" and not vr.recognizeWord(): continue
				userSet = set(recognizedVoice) & set(validUsers)
				if len(userSet) == 1:
					voiceAuthenticated = True
					unlock()
			"""
            while state == "unlocked":
                if lock button pressed:
                    lock()
                    voiceAuthenticated = False
			"""
	elif args.authentication_model == "face":
		faceAuthenticated = False
		while True:
			while not faceAuthenticated:
                # Turn on orange led for video
				recognizedFaces = fr.run(data, vs, fps, debug=False)
                # Turn on green led for video
				print("recognizedFaces: ", recognizedFaces)
				userSet = set(recognizedFaces) & set(validUsers)
				print("UserSet: ", userSet)
				if len(userSet) == 1:
					faceAuthenticated = True
					unlock()
			"""
            while state == "unlocked":
                if lock button pressed:
                    lock()
                    faceAuthenticated = False
			"""
if __name__ == "__main__": driver()

#./controller.py -ad "AudioData" -fd "ImageData" -vm "voiceModel.bin" -fm "imageEncodings.bin" -lv -am both -vu Alex Dave
