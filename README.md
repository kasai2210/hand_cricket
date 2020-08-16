# Hand Cricket
Play the all time school classic Hand Cricket a.k.a. ODD-EVE

## Requirements
1) Python 3
2) Keras 
3) Tensorflow
4) PIL
5) Numpy
6) Matplotlib
7) OpenCV

## Setup Instructions
1) Clone the repository
```bash
$ git clone https://github.com/kasai2210/hand_cricket.git
$ cd hand_cricket
```
2) Install the dependencies
```bash
$ pip install -r requirements.txt
```
3) Gather images for training
```bash
$ python gather_images.py <label_name> <total_training_examples>
```
Use the command in the following way, suppose you want to collect 200 examples for label = 1, then use - 
```bash
$ python gather_images.py 1 200
```
Likewise do this for all the labels, i.e. label = 0, 1, 2, 3, 4, 5

4) Train your model
```bash
$ python train.py
```
5) Now comes the fun part, play the game 

For bowling - 
```bash
$ python bowl.py
```
For batting - 
```bash
$ python bat.py
```

## ENJOY!!
