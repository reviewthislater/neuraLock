#!/usr/bin/env python3
import cv2
import sys
import os
import numpy as np
from PIL import Image
import warnings

model = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
videoCapture = cv2.VideoCapture(0)
names = []
confidenceThreshold = 40


def getImageData(rootDirectory="ImageData"):
    allLabels = [] # holds the labels for every image in the train data set
    validLabels = [] # only holds the labels for images that detected exactly one face
    images = [] # images with exactly one face
    imagePaths = [] # paths to the images
    imageDirectory = sys.path[0] + "/" + rootDirectory # looks for the root directory relative to where the script was run because python needs absolute paths (add to sys.path for relative paths)

    #imagePaths = [imageDirectory + "/" + folder + "/" + file for file in os.listdir(imageDirectory + "/" +folder) for folder in os.listdir(imageDirectory)]
    # should be able to replace the following loop with soemthing similar to the above code (its still looping anyway)

    # Grab image paths
    for i, label in enumerate(os.listdir(imageDirectory)): # for each folder in the image directory
        names.insert(0,label) # grab the names of the people since opencv predicts to integer labels we will convert back to these names later
        for image in os.listdir(imageDirectory + "/" + label): # for each file in each folder
           imagePaths.append(imageDirectory + "/" + label + "/" + image) #add to the image paths
           allLabels.append(i) # append the integer label

    # Determine valid images (exactly one face needs to be detected to be a valid image)
    for imagePath in imagePaths:
        original = cv2.imread(imagePath)
        grayImage = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # change scale factor here if faces aren't being detected, it relates to how far the person is from the camera
        if len(faces) != 1: warnings.warn("Not using an image from the train data set because exactly one face was not detected")
        else:
            validLabels.append(allLabels[i])
            for (x, y, w, h) in faces: images.append(grayImage[y:y+h, x:x+w])

    return images, validLabels



def train(rootDirectory="ImageData"):
    images, labels = getImageData(rootDirectory) # grab the data
    model.train(images, np.array(labels)) # train on the data


# Dynamic predictions from the video camera
def predict():
    while True:
        _, frame = videoCapture.read() # read in the fram we don't care about the return code for continous frames
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
        faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # detect faces using the cascade
        for (x, y, w, h) in faces:
            face = grayFrame[y: y + h, x: x + w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Draw a rectangle around the faces
            label, confidence = model.predict(face)
            text = names[label] if confidence <= confidenceThreshold else "Unknown"
            cv2.putText(frame, text, (int(x+w/3),int(y+1.1*h)), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,255,255), 2)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


# Think of the following as main in C, its the entry point of the program (for now until implement setup.py)
if __name__ == "__main__":
    train()
    predict()
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
