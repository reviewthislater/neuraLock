#!/usr/bin/env python3
from Recognizers.VoiceRecognizer import VoiceRecognizer#FaceRecognizer, VoiceRecognizer
from Recognizers.utilities import utilities
from scipy.io.wavfile import read
import numpy as np
import sys
import argparse

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
    epilog="By: Alex Epstein https://github.com/alexanderepstein\n&\nDave Wigley https://github.com/davidwigley"
    )
    ap.add_argument("-ad", "--audio_dataset", help="path to audio training data", required=True)
    ap.add_argument("-fd", "--face_dataset",  help="path to face training data", required=True)
    ap.add_argument("-vm", "--voice_model", help="path to serialized voice model", required=True)
    ap.add_argument("-fm", "--face_model", help="path to serialized face model", required=True)
    ap.add_argument("-lv", "--load_voice_model", help="boolean for loading voice model", action="store_true",)
    ap.add_argument("-lf", "--load_face_model", help="boolean for loading face model", action="store_true",)
    ap.add_argument("-u", "--unlock_phrase", help="unlock phrase", default="unlock")
    args = ap.parse_args()

    vr = VoiceRecognizer(unlockPhrase=args.unlock_phrase, savedModelPath=args.voice_model,
    audioDirectory=args.audio_dataset)
    if args.load_voice_model: vr.loadModel()
    else: vr.train()
    testVoiceRecognizer(vr)

if __name__ == "__main__": driver()

#./controller.py -ad "AudioData" -fd "ImageData" -vm "voiceModel.bin" -fm "imageEncodings.bin" -lv
