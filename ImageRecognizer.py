#!/usr/bin/env python3
import cv2
import sys
import os
import numpy as np
import warnings
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
import tts
from sklearn.decomposition import KernelPCA, PCA, NMF, FastICA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

faceCascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
#faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
videoCapture = cv2.VideoCapture(0)
names = []
confidenceThreshold = 60

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
        names.append(label) # grab the names of the people since opencv predicts to integer labels we will convert back to these names later
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
                image = np.array(image)
                cv2.equalizeHist(image, image) # Uniformly distribute lighting
                #image = cv2.bilateralFilter(image, 15, 10, 10) # Smooth sharp features like a blur
                #image = cv2.fastNlMeansDenoising(image,None,4,7,21) # Remove noise from image
                images.append(cv2.resize(image[y:y+h, x:x+w],(50,50))) if not opencv else images.append(image[y:y+h, x:x+w])
                if not opencv: assert len(images[-1]) == 50*50

    assert len(validLabels) == len(images)
    return images, validLabels


def loadModel(rootDirectory="ImageData", opencv=True):
    filename = "ocvmodel" if opencv else "svcmodel"
    filepath = rootDirectory+ "/" + filename + ".yml"
    if opencv:
        if os.path.isfile(filepath) and False:
            model = cv2.face.LBPHFaceRecognizer_create()
            model.load(filepath)
        else:
            model = cv2.face.LBPHFaceRecognizer_create()
            train(model, opencv, rootDirectory)
            model.save(filepath)
    return model


def train(model, modelType="OpenCV", rootDirectory="ImageData", decomposers=[]):
    images, labels = getImageData(rootDirectory) # grab the data
    labels = np.array(labels)

    if modelType == "OpenCV" or modelType == "Combined":
        model[0].train(images, labels) # train on the data opencv model found at index 0
    if modelType == "SVC" or modelType == "Combined":
        sciImages = np.empty(shape=(1,2500))
        for i, image in enumerate(images):
            image = cv2.resize(image, (50,50)).ravel()
            if i == 0: sciImages[0, :] = image
            else: sciImages = np.vstack([sciImages, image])
        for decomposer in decomposers: sciImages = decomposer.fit_transform(sciImages,  labels)
        model[1].fit(sciImages, labels) if modelType == "Combined" else model[0].fit(sciImages, labels)



# Dynamic predictions from the video camera
def predict(model, modelType="OpenCV", decomposers=[]):
    lastLabel = 0
    continousPredictions = 0
    while True:
        _, frame = videoCapture.read() # read in the fram we don't care about the return code for continous frames
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
        faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # detect faces using the cascade
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Draw a rectangle around the faces
            if modelType == "OpenCV":
                assert len(model) == 1
                face = grayFrame[y: y + h, x: x + w]
                cv2.equalizeHist(face, face)
                label, confidence = model[0].predict(face)
                confidence = 100 - confidence # confidence in opencv comes out as 100-confidence
            elif modelType == "SVC":
                assert len(model) == 1
                face = grayFrame[y: y + h, x: x + w]
                cv2.equalizeHist(face, face)
                face = cv2.resize(face,(50,50))
                face= face.ravel().reshape(1,-1)
                for decomposer in decomposers: face = decomposer.transform(face)
                predictions = model[0].predict_proba(face)[0]
                print(predictions)
                label = np.argmax(predictions)
                confidence = predictions[label]*100
            else:
                assert len(model)==2
                # opencv first
                face = grayFrame[y: y + h, x: x + w]
                cv2.equalizeHist(face, face)
                cvModel = model[0]
                svcModel = model[1]
                cvLabel, cvConfidence = cvModel.predict(face)
                cvConfidence = 100 - cvConfidence
                # now svc
                face = cv2.resize(face,(50,50))
                face = face.ravel().reshape(1,-1)
                for decomposer in decomposers: face = decomposer.transform(face)
                svcPredictions = svcModel.predict_proba(face)[0]
                svcLabel = np.argmax(svcPredictions)
                svcConfidence = svcPredictions[svcLabel]*100
                if cvLabel == svcLabel:
                    label = cvLabel
                    confidence = (svcConfidence+cvConfidence)/2
                else:
                    label = cvLabel if cvConfidence >= svcConfidence else svcLabel
                    confidence = max(cvConfidence, svcConfidence)-20 # 20 percent penatly for the fact they dont agree

            if confidence >= confidenceThreshold:
                if lastLabel == label: continousPredictions += 1
                else: continousPredictions = 0
            lastLabel = label
            #if opencv and continousPredictions >= 1:
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
    modelTypes = ["OpenCV","SVC", "Combined"]
    modelType=modelTypes[1]
    if modelType == "OpenCV": model = [cv2.face.LBPHFaceRecognizer_create()]
    elif modelType == "SVC": model = [SVC(probability=True, kernel="poly", degree=4)]#SVR(kernel="poly", degree=5)
    else: model = [cv2.face.LBPHFaceRecognizer_create(), SVC(probability=True, kernel="poly", degree=4)]
    decomposers = [LinearDiscriminantAnalysis(), KernelPCA(kernel="poly", degree= 8)]#KernelPCA(kernel="rbf", degree=5)] # PCA()
    train(model=model, modelType=modelType, rootDirectory="ImageData", decomposers=decomposers)
    predict(model=model, modelType=modelType, decomposers=decomposers)
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
