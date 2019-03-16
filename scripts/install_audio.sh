#!/bin/sh
# Install audio packages for Ubuntu which is needed for pyaudio.

sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools

pip install pyaudio