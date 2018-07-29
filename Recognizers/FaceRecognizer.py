#!/usr/bin/env python3
# import the necessary packages
from imutils.video import VideoStream, FPS
import face_recognition as fr
import imutils
import pickle
import time
import cv2
import os
import sys
import warnings

class FaceRecognizer(object):

	def __init__(self, validUsers, detectionMethod="hog", encodings="imageEncodings.bin", detector=cv2.CascadeClassifier(sys.path[0] + "/lbpcascade_frontalface.xml")):
		self.detectionMethod = detectionMethod # Default to hog which is 128d feature extraction and linear svm
		self.encodings = encodings #pickled encodings, should only need to encode once
		self.detector = detector # lbp is faster than haar cascade
		assert type(validUsers) == type([])
		self.validUsers = validUsers # list of validUsers

	def getImageData(self, rootDirectory="ImageData"):
		images = [] # images with exactly one face
		imagePaths = [] # paths to all the images
		validPaths = [] # paths to images with only 1 face detected
		imageDirectory = sys.path[0] + "/" + rootDirectory # looks for the root directory relative to where the script was run because python needs absolute paths (add to sys.path for relative paths)
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


	def encodeImages(self):
	    	print("[INFO] quantifying faces...")
	    	imagePaths, labels = getImageData() #  grab the paths to the input images in our dataset
	    	knownEncodings = []
	    	knownNames = []

	    	for (i, imagePath) in enumerate(imagePaths): # loop over the image paths
	    		print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	    		name = imagePath.split(os.path.sep)[-2] # extract the person name from the image path
	    		image = cv2.imread(imagePath) # load image
	    		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to rgb
	    		faces = fr.face_locations(rgb, model=detectionMethod) # detect box around face
	    		encodings = fr.face_encodings(rgb, faces) # compute the facial embedding for the face
	    		for encoding in encodings: # loop over the encodings
	    			knownEncodings.append(encoding) # add each encoding to knownEncodings
	    			knownNames.append(name) # add each name to knownNames
	    	print("[INFO] serializing encodings...")
	    	data = {"encodings": knownEncodings, "names": knownNames}
	    	f = open(self.encodings, "wb")
	    	f.write(pickle.dumps(data)) # dump the facial encodings + names to disk
	    	f.close()

	def setup(self):
		print("[INFO] loading encodings + face detector...")
		data = pickle.loads(open(self.encodings, "rb").read()) # load the known faces and embeddings along with lbp for face detection
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start() # initialize the video stream and allow the camera sensor to warm up
		time.sleep(2.0)
		fps = FPS().start() # start the FPS counter
		return data, vs, fps

	def run(self, data, vs, fps, debug=True):
		continousPredictions = 0
	    # loop over frames from the video file stream
		while continousPredictions <=3:
			frame = vs.read() # grab the frame from the threaded video stream
			frame = imutils.resize(frame, width=500) # resize to 500px (to speedup processing)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the input frame from (1) BGR to grayscale (for face detection)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # (2) from BGR to RGB (for face recognition)

			rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 	# detect faces in the grayscale frame
	    		# OpenCV returns bounding box coordinates in (x, y, w, h) order
	    		# but we need them in (top, right, bottom, left) order, so we
	    		# need to do a bit of reordering
			boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
			encodings = fr.face_encodings(rgb, boxes) # compute the facial embeddings for each face bounding box
			names = []
	    		# loop over the facial embeddings
			for encoding in encodings:
				matches = fr.compare_faces(data["encodings"], encoding) # attempt to match each face in the input image to our known encodings
				name = "Unknown"

	    			# check to see if we have found a match
				if True in matches:
	    				# find the indexes of all matched faces then initialize a
	    				# dictionary to count the total number of times each face
	    				# was matched
					matchedIdxs = [i for (i, b) in enumerate(matches) if b]
					counts = {}

	    				# loop over the matched indexes and maintain a count for
	    				# each recognized face face
					for i in matchedIdxs:
						name = data["names"][i]
						counts[name] = counts.get(name, 0) + 1

	    				# determine the recognized face with the largest number
	    				# of votes (note: in the event of an unlikely tie Python
	    				# will select first entry in the dictionary)
					name = max(counts, key=counts.get)
	    			# update the list of names
				names.append(name)
	    			# loop over the recognized faces
			if any(face in self.validUsers for face in names): continousPredictions += 1
			else: continousPredictions = 0

			if debug:
				for ((top, right, bottom, left), name) in zip(boxes, names):
					# draw the predicted face name on the image
					cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
					y = top - 15 if top - 15 > 15 else top + 15
					cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
					# display the image to our screen
					cv2.imshow("Frame", frame)
					key = cv2.waitKey(1) & 0xFF
					if key == ord("q"): break # if the `q` key was pressed, break from the loop
					fps.update() # update the FPS counter
		if debug:
			 # stop the timer and display FPS information
			fps.stop()
			print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
			# do a bit of cleanup
			cv2.destroyAllWindows()
			vs.stop()

		return set.union(set(names), set(self.validUsers))# detected users
