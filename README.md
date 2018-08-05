<div align="center">

# neuraLock

<img src="http://cosmonio.com/Research/Deep-Learning/files/small_1420.png" width=300px height=300px>

#### A proof of concept safe utilizing deep learning for facial, vocal, and word recognition.
#### Contains code and STLS to build your own nueraLock safe

![platform](https://img.shields.io/badge/platform-Raspberry%20Pi-red.svg) [![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=plastic)]()

</div>


## Hardware
[Raspberry Pi 3 Model B](https://www.amazon.com/gp/product/B01LPLPBS8/ref=oh_aui_detailpage_o00_s00?ie=UTF8&psc=1): $35.99
[Samsung Evo Micro SD](https://www.amazon.com/Samsung-MicroSDXC-Adapter-MB-ME64GA-AM/dp/B06XX29S9Q/ref=sr_1_3?s=electronics&ie=UTF8&qid=1533447856&sr=1-3&keywords=sd+card+samsung&dpID=41pa5T0NGKL&preST=_SX300_QL70_&dpSrc=srch): $17.99
[Keyboard w/ trackpad](https://www.amazon.com/Rii-Wireless-Keyboard-Touchpad-Control/dp/B00I5SW8MC/ref=sr_1_5?s=electronics&ie=UTF8&qid=1533447912&sr=1-5&keywords=mini+wireless+keyboard&dpID=41qFpeX3-cL&preST=_SY300_QL70_&dpSrc=srch): $15.95
[ELP Mini USB Camera](https://www.amazon.com/gp/product/B01DRJXDEA/ref=oh_aui_detailpage_o01_s00?ie=UTF8&psc=1): $29.99
[Lavalier Mini USB Microphone](https://www.amazon.com/gp/product/B074BLM973/ref=oh_aui_detailpage_o01_s00?ie=UTF8&psc=1): $14.99
[5000 mAh External Battery](https://www.amazon.com/gp/product/B00MWU1GGI/ref=oh_aui_detailpage_o01_s00?ie=UTF8&psc=1): $11.99
[Assorted LEDS](https://www.amazon.com/Lights-Emitting-Assortment-Arduino-300-Pack/dp/B00UWBJM0Q/ref=sr_1_4?s=hi&ie=UTF8&qid=1533447484&sr=1-4&keywords=leds&th=1): $9.99
[Tactile Switches](https://www.amazon.com/Uxcell-a15111200ux1613-Momentary-Tactile-Terminal/dp/B019DCWTSQ/ref=sr_1_17?s=hi&ie=UTF8&qid=1533447743&sr=1-17&keywords=tactile+switch): $7.46
3d printer and filament: $??.??

#### Total Cost: 144.35 + ??.??

All of these parts except for the raspberry pi 3 can be exchanged with similar parts, the provided links provide either the exact or similar hardware I used.


## Setup & Install
### Clone the repository
```bash
cd ~
git clone https://github.com/alexanderepstein/neuraLock
```
### Enter the repository
```bash
cd neuraLock
```
### Run environment build script
```
./env.sh
```
The script will ask for sudo privileges.
### Build training dataset
For both image and audio data the folder structure should look like
```bash
├── AudioData
│   ├── Subject1
│   │   ├── file1.wav
│   │   ├── file2.wav
│   │   ├── file3.wav
│   ├── Subject2
│   │   ├── file1.wav
│   │   ├── file2.wav
│   ├── Subject3
│   │   ├── file1.wav
│   │   ├── file2.wav
│   │   ├── file3.wav
```

### Gathering Face Training Data
I used about 8 subjects from the Yale Faces Extended B training set, the larger the number of subjects used the more robust the classifier will be.

### Gathering Audio Training Data
I used about 5 subjects from the [VoxForge dataset](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/), it is useful to utilize training data with the same sampling rate as your microphone. It is also useful to get training data from the same individual recorded at different times, something like [Angus-CJG](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/48kHz_16bit/Angus-20080320-cjg.tgz)
and [Angus-NDD](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/48kHz_16bit/Angus-20080320-ndd.tgz) the wav files from both of these archives would go under one folder named Angus in the AudioData root directory.

### Wiring The Hardware
![WiringDiagram](https://github.com/alexanderepstein/neuraLock/blob/master/Wiring.png)

The pin layout will be treated as BCM, to see what that looks like visit [here](https://gpiozero.readthedocs.io/en/stable/recipes.html#pin-numbering)

|     Component    | GPIO |
|:----------------:|------|
| Servo            |  17  |
| Push Button      |  18  |
| Safe Green LED   |  21  |
| Safe Orange LED  |  20  |
| Safe Red LED     |  16  |
| Face Green LED   |  13  |
| Face Orange LED  |  19  |
| Face Red LED     |  26  |
| Voice Green LED  |   5  |
| Voice Orange LED |   6  |
| Voice Red LED    |  12  |

### Executing the program

#### Command line arguments
```bash
-ad AUDIO_DATASET, --audio_dataset AUDIO_DATASET
                      path to audio training data
 -fd FACE_DATASET, --face_dataset FACE_DATASET
                       path to face training data
 -vm VOICE_MODEL, --voice_model VOICE_MODEL
                       path to serialized voice model
 -fm FACE_MODEL, --face_model FACE_MODEL
                       path to serialized face model
 -lv, --load_voice_model
                       boolean for loading voice model
 -lf, --load_face_model
                       boolean for loading face model
 -u UNLOCK_PHRASE, --unlock_phrase UNLOCK_PHRASE
                       unlock phrase
 -am {both,face,voice}, --authentication_model {both,face,voice}
                       choose the authentication model
 -vu VALID_USERS [VALID_USERS ...], --valid_users VALID_USERS [VALID_USERS ...]
                       users the safe should open for
```
A valid execution of the program might look like
```bash
./controller.py -ad "AudioData" -fd "ImageData" -vm "voiceModel.bin" -fm "imageEncodings.bin" -lv -am both -vu Alex
# Execute the program, audio data is located in AudioData, image data is contained in "ImageData", the voice model should be read to/written from "voiceModel.bin", the face encodings should be read to/written from "imageEncodings.bin", load the voice model, authenticate using both voice and face, set Alex as a valid user
```

### Issues
For any issues please open a new issue in the issue tracker [here](https://github.com/alexanderepstein/neuraLock/issues/new)

## License

MIT License

Copyright (c) 2018 Alex Epstein

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
