class lockMechanism():
    imageRecognized = False
    voiceRecognized = False
    lockAngle = 180
    unlockAngle = 90

    @staticmethod
    def unlock():
        if imageRecognized and voiceRecognized:
            #move servo to unlock angle

    @staticmethod
    def lock():
        #move servo to lock angle
