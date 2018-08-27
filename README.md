# movehack_2018

This repository contains demo code for Outliers on Indian Roads(Artificial Intelligence for Indian Transport Infrastructure).

## Installation on ubuntu 16.04
1. Tensorflow https://www.tensorflow.org/install/
2. run below commands on terminal.

    `pip install -r requirements.txt`

    `sudo apt-get install protobuf-compiler`

    `protoc object_detection/protos/*.proto --python_out=.`

    `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`

2. OpenCV, numpy
   `sudo apt-get install python-opencv numpy`

## Steps to Run

1. Place PNG image(s) in input folder.
2. run `python run.py`
3. find results in output folder
