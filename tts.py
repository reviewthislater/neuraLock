import pyttsx3
import espeak

espeak.
class tts():


    def speak(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.setProperty('rate',120)  #120 words per minute
        engine.setProperty('volume',0.9)
        engine.runAndWait()
        engine.stop()
