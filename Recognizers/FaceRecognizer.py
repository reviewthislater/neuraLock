#!/usr/bin/env python3
# import the necessary packages
from imutils.video import VideoStream, FPS
from imutils import resize
import face_recognition as fr
import pickle
from  time import sleep
import cv2
import os
from .utilities import utilities

class FaceRecognizer(object):

	def __init__(self, validUsers, detectionMethod="hog", imageDirectory=None, encodings="imageEncodings.bin", detector=cv2.CascadeClassifier("lbpcascade_frontalface.xml")):
		self.detectionMethod = detectionMethod # Default to hog which is 128d feature extraction and linear svm
		self.encodings = encodings #pickled encodings, should only need to encode once
		self.detector = detector # lbp is faster than haar cascade
		assert type(validUsers) == type([])
		self.validUsers = validUsers # list of validUsers
		self.imageDirectory = imageDirectory
		self.encodingData = None

	def encodeImages(self):
		"""
		Get image data
		Perform feature	extraction on the data		if debug:
			# stop the timer and display FPS information
			print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
			# do a bit of cleanup
			cv2.destroyAllWindows()
		Save the model of the extracted features and their respective targets
		"""
		print("[INFO] quantifying faces...")
		imagePaths = utilities.getImageData(self.imageDirectory) #  grab the paths to the input images in our dataset
		knownEncodings = []
		knownNames = []
		for (i, imagePath) in enumerate(imagePaths): # loop over the image paths
			print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
			name = imagePath.split(os.path.sep)[-2] # extract the person name from the image path
			image = cv2.imread(imagePath) # load image
			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to rgb
			faces = fr.face_locations(rgb, model=self.detectionMethod) # detect box around face
			encodings = fr.face_encodings(rgb, faces) # compute the facial embedding for the face
			for encoding in encodings: # loop over the encodings
				knownEncodings.append(encoding) # add each encoding to knownEncodings
				knownNames.append(name) # add each name to knownNames
		print("[INFO] serializing encodings...")
		self.encodingData = {"encodings": knownEncodings, "names": knownNames}
		f = open(self.encodings, "wb")
		f.write(pickle.dumps(self.encodingData)) # dump the facial encodings + names to disk
		f.close()

	def loadEncodings(self):
		"""
		Load the encodings for the model
		"""
		print("[INFO] loading encodings + face detector...")
		self.encodingData = pickle.loads(open(self.encodings, "rb").read()) # load the known faces and embeddings along with lbp for face detection


	def setup(self):
		"""
		output vs: video stream object that has been started and warmed up
		output fps: fps counting object

		Request the VideoStream resource and start it
		Sleep to allow the camera to warm up
		Create the fps counter and start it
		Return the video stream object and the fps counter object
		"""
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start() # initialize the video stream
		sleep(2.0) # sleep to let camera warm up
		fps = FPS().start() # start the FPS counter
		return vs, fps

	def run(self, vs, fps=None, debug=False):
		"""
		input vs: video stream object for capturing images
		input fps: fps counting object for debug use (only needed when debug is True)
		input debug: boolean for printing information to the terminal

		output validNameSet: a set of the names of recognized faces that are in the valid user set

		Read in the frame
		Detect the faces
		Encode the faces
		Compare the faces
		Predict the name of the person
		Check any person in image is in valid users
			if true: Increment continuous predictious
			else: Set continuous predictions to 0
		Run the above lines in loop until continous predictions is 3
		Return the valid user set intersected with the recognized user set
		"""
		continousPredictions = 0
	    # loop over frames from the video file stream
		while continousPredictions <= 3:
			frame = vs.read() # grab the frame from the threaded video stream
			frame = resize(frame, width=500) # resize to 500px (to speedup processing)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the input frame from (1) BGR to grayscale (for face detection)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # (2) from BGR to RGB (for face recognition)
			rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # detect faces in the grayscale frame
            # OpenCV returns bounding box coordinates in (x, y, w, h) order
	    	# but we need them in (top, right, bottom, left) order, so we
	    	# need to do a bit of reordering
			boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
			encodings = fr.face_encodings(rgb, boxes) # compute the facial embeddings for each face bounding box
			names = []
			for encoding in encodings: 	# loop over the facial embeddings
				matches = fr.compare_faces(self.encodingData["encodings"], encoding, tolerance=0.45) # attempt to match each face in the input image to our known encodings
				name = "Unknown"
				if True in matches: # check to see if we have found a match
					matchedIdxs = [i for (i, b) in enumerate(matches) if b] # find the indexes of all matched faces
					counts = {} # dictionary to count the total number of times each face was matched
					for i in matchedIdxs: 	# loop over the matched indexes
						name = self.encodingData["names"][i]
						counts[name] = counts.get(name, 0) + 1 # maintain a count for each recognized face
					name = max(counts, key=counts.get) # face with largest number of votes
				names.append(name) 	# update the list of names
			if any(face in self.validUsers for face in names): continousPredictions += 1
			else: continousPredictions = 0

			if debug:
				for ((top, right, bottom, left), name) in zip(boxes, names):
					# draw the predicted face name on the image
					cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
					y = top - 15 if top - 15 > 15 else top + 15
					cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
					key = cv2.waitKey(1) & 0xFF
					if key == ord("q"): break # if the `q` key was pressed, break from the loop
				cv2.imshow("Frame", frame) # display the image to our screen
				fps.update() # update the FPS counter

		if debug:
			# stop the timer and display FPS information
			fps.stop()
			print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
			cv2.destroyAllWindows() # do a bit of cleanup

		vs.stop() # stops the video stream thread
		sleep(1) # Give the thread some time to stop
		vs.stream.release() # releases the camera resource
		sleep(1) # Give the camera some time to release
		return set(names) & set(self.validUsers) # detected users
