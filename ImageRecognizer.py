#!/usr/bin/env python3
import cv2
import sys
import os
import numpy as np
import warnings
from sklearn.linear_model import SGDClassifier

faceCascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
#faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
videoCapture = cv2.VideoCapture(0)
names = []
confidenceThreshold = 65


def getImageData(rootDirectory="ImageData", opencv=True):
    allLabels = [] # holds the labels for every image in the train data set
    validLabels = [] # only holds the labels for images that detected exactly one face
    images = [] # images with exactly one face
    imagePaths = [] # paths to the images
    imageDirectory = sys.path[0] + "/" + rootDirectory # looks for the root directory relative to where the script was run because python needs absolute paths (add to sys.path for relative paths)

    #imagePaths = [imageDirectory + "/" + folder + "/" + file for file in os.listdir(imageDirectory + "/" +folder) for folder in os.listdir(imageDirectory)]
    # should be able to replace the following loop with soemthing similar to the above code (its still looping anyway)

    # Grab image paths
    for i, label in enumerate(os.listdir(imageDirectory)): # for each folder in the image directory
        #names.insert(0,label) # grab the names of the people since opencv predicts to integer labels we will convert back to these names later
        names.append(label)
        for image in os.listdir(imageDirectory + "/" + label): # for each file in each folder
           imagePaths.append(imageDirectory + "/" + label + "/" + image) #add to the image paths
           allLabels.append(i) # append the integer label

    # Determine valid images (exactly one face needs to be detected to be a valid image)
    for i, imagePath in enumerate(imagePaths):
        original = cv2.imread(imagePath)
        image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) #if opencv else original


        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # change scale factor here if faces aren't being detected, it relates to how far the person is from the camera
        if len(faces) != 1: warnings.warn("Not using an image from the train data set because exactly one face was not detected")
        else:
            validLabels.append(allLabels[i])
            for (x, y, w, h) in faces:
                images.append(cv2.resize(image[y:y+h, x:x+w],(50,50))) if not opencv else images.append(image[y:y+h, x:x+w])
                if not opencv: assert  image.size == 50*50

    assert len(validLabels) == len(images)
    return images, validLabels


def loadModel(rootDirectory="ImageData", opencv=True):
    filename = "ocvmodel" if opencv else "svcmodel"
    filepath = rootDirectory+ "/" + filename
    if os.path.isfile(filepath):
        model = pickle.load(filepath)
    else:

        train(model, opencv, rootDirectory)
        pm = pickle.dumps(model, -1)
        pickle.dump(pm, filepath)
    return model


def train(model, opencv=True, rootDirectory="ImageData"):
    images, labels = getImageData(rootDirectory) # grab the data
    if opencv: model.train(images, np.array(labels)) # train on the data
    else:
        sciImages = np.empty(shape=(1,2500))
        for i, image in enumerate(images):
            image = cv2.resize(image, (50,50)).ravel()
            if i == 0: sciImages[0, :] = image
            else: sciImages = np.vstack([sciImages, image])
        model.fit(sciImages, np.array(labels))


# Dynamic predictions from the video camera
def predict(model, opencv=True):
    print(names)
    while True:
        _, frame = videoCapture.read() # read in the fram we don't care about the return code for continous frames
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
        faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # detect faces using the cascade
        for (x, y, w, h) in faces:
            face = grayFrame[y: y + h, x: x + w] if opencv else cv2.resize(grayFrame[y: y + h, x: x + w],(50,50))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Draw a rectangle around the faces
            if opencv:
                label, confidence = model.predict(face)
                confidence = 100 - confidence
            else:
                label =  model.predict(face.ravel().reshape(1,-1)).item()
                confidence = model.predict_proba(face.ravel().reshape(1,-1))[0]
                print(label)
                print(confidence)
                break
            text = names[label] if confidence >= confidenceThreshold else "Unknown"
            cv2.putText(frame, text, (int(x+w/5),int(y+1.1*h)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 2)
            cv2.putText(frame," Confidence: " + str(confidence), (int(x+w/3),int(y+1.2*h)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 2)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


# Think of the following as main in C, its the entry point of the program (for now until implement setup.py)
if __name__ == "__main__":
    opencv = True
    model = cv2.face.LBPHFaceRecognizer_create() if opencv else SGDClassifier(loss="modified_huber")
    train(model, rootDirectory="ImageData", opencv=opencv)
    predict(model, opencv)
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
