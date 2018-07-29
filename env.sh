#! /usr/bin/env bash
# Raspberry Pi neuraLock environment build script
# By Alex Epstein https://github.com/alexanderepstein
echo "Updating and Upgrading"
sudo apt-get update && sudo apt-get upgrade -y
echo "Removing wolfram-engine to save space"
sudo apt-get purge wolfram-engine -y
echo "Installing build tools"
sudo apt-get install build-essential cmake pkg-config -y
echo "Install audio and video codecs"
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev python3-pyaudio python3-espeak libhdf5-dev libhdf5-serial-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt-get install libxvidcore-dev libx264-dev -y
echo "Install GUI libraries"
sudo apt-get install libqtgui4 libqt4-test libgtk2.0-dev -y
echo "Install optimization libraries"
sudo apt-get install libatlas-base-dev gfortran swig -y
echo "Cleaning unneeded packages"
sudo apt autoremove -y
echo "Install python libraries"
sudo pip3 install -U numpy pandas scikit-learn matplotlib seaborn pyttsx3
echo "Install OpenCV"
sudo pip3 install opencv-python opencv-contrib-python
echo "Validating OpenCV install"
python3 -c "import cv2; print(cv2.__version__)" || { echo "OpenCV install failed"; exit 1; }
echo "Install dlib and face_recognition"
sudo pip3 install dlib face_recognition
echo "Setting matplotlibrc"
touch ~/.config/matplotlib/matplotlibrc
echo "backend: TkAgg"> ~/.config/matplotlib/matplotlibrc
echo "Validating dlib and face_recognition install"
python3 -c "import face_recognition" || echo "Face recognition install failed"
echo "Finished install successfully"
exit 0
