## Image Captioning - Team SaaS
A repository to collaborate on the Course Project for ECE285 course "Machine Learning for Image Processing" taken at UCSD in the Fall Quarter of 2019

The project is a collaborative project that was worked on by [Anirudh](https://github.com/Anirudh-Swaminathan), [Aparna](https://github.com/aparna9625), [Savitha](https://github.com/savitha0602) and [Sidharth](https://github.com/Sidharth2905) from team **Saas**

## Software Requirements
To work on the project, please use Python 3.
Install the required packages from [requirements.txt](./requirements.txt)
```bash
pip install --user -r requirements.txt
```
NOTE - These were the packages that were default present in the UCSD DSMLP cluster. They have been uploaded here so as to ensure compatibility for runs.

## Collaboration
Contribution guidelines are given [here](./CONTRIBUTING.md)

## Dataset Preparation Instructions

To work on the dataset, we have used the official repositories provided for COCO Dataset.
The first repository that is utilized as a submodule is [cocoapi](https://github.com/cocodataset/cocoapi/tree/636becdc73d54283b3aac6d4ec363cffbb6f9b20). This provides a wrapper to load captions that correspond to images.
The second repository that is utilized as a submodule is [coco-caption](https://github.com/Anirudh-Swaminathan/coco-caption/tree/c5ffd796caca13de757967317ba364d7d91e3a2f). This provides a wrapper to evaluate our resultant captions by providing implementations of BLEU scores.
#### NOTE:
This repository is my fork of the original repository [here](https://github.com/tylin/coco-caption). 

### Dataset Annotations (Captions) Download
These are the steps to set up the dataset:-
1. Just use the images of the dataset given in the DSMLP cluster in /datasets/COCO-2015/
2. Create a sub-directory in the project root named datasets/
3. Create another sub-directory within it named as COCO/
4. For the captions, download the annotations from the MS COCO website.
5. Download the training set annotations as a zip file from [here](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
6. Inflate inside the ./datasets/COCO/
7. Similary, download the zip file for 2014 testing image information from [here](http://images.cocodataset.org/annotations/image_info_test2014.zip) and follow step 6.
8. Similarly, download the zip file for the 2015 testing image information from [here](http://images.cocodataset.org/annotations/image_info_test2015.zip) and follow step 6.

### Image loader
These are the steps to get your COCO dataset image loader up and running :-

1. Clone this repo recursively.
To do this, run
```bash
cd src/
git submodule update --init --recursive
```
2. Build the submodule by running the following
```bash
cd src/cocoapi/PythonAPI/
make
```
3. Additionally symlink the pycocotools in the cocoapi's PythonAPI directory into src/
This can be done by the following
```bash
cd src/
ln -s ./cocoapi/PythonAPI/pycocotools ./
``` 

## Trained Models Loading
The trained models for the project can be found in this Google Drive [link](https://drive.google.com/drive/folders/1OgL4AOoS6XyzSQr6W_Gu6uV0BVRVWatS?usp=sharing).

Please download each of the folders and place that folder in the [outputs](./outputs) folder.

## Code Organization
| File | Function |
| ---- | -------- |
| src/alex\_adam.py | Trains for 1 epoch with alexnet as encoder and standard LSTM as decoder |
| src/alexnet\_adam.ipynb | Loads the trained and evaluated model of Alexnet encoder and graphs the losses, BLEU scores and prints loss on validation set |
| src/demo.ipynb | Runs a demo of our code to produce caption for 1 random image |
| src/experiment\_flip.ipynb | Runs the experiment to randomly flip an image horizontally and vertically and display the captions |
| src/framework.ipynb | Our initial try for a framework |
| src/framework\_final.ipynb | A framework that is duplicated to be changed to run specific experiments |
| src/framework\_final.py | A framework that trains and evaluates a specific model |
| src/nntools\_modified.py | A python file provided to us in ECE285 that is modified to be specific to our needs |
| src/vgg16\_adam.ipynb | Loads the trained and evaluated model of VGG16 network with Adam Optimizer and graphs the losses, BLEU scores and prints loss on validation set |
| src/vgg16\_sgd.ipynb | Loads the trained and evaluated model of VGG16 network with SGD Optimizer, 0.9 momentum and graphs the losses, BLEU scores and prints loss on validation set |
| src/vgg16\_sgd\_nesterov.ipynb | Loads the trained and evaluated model of VGG16 network with SGD Optimizer, 0.9 momentum, Nesterov acceleration and graphs the losses, BLEU scores and prints loss on validation set |
| src/vgg16\_sgd\_zero\_mom.ipynb | Loads the trained and evaluated model of VGG16 network with SGD Optimizer, 0 momentum and graphs the losses, BLEU scores and prints loss on validation set |
| src/vgg16\_adam.py | Trains network and evaluates on validation set with VGG16 encoder and Adam Optimizer for 5 epochs |
| src/vgg16\_sgd.py |  Trains network and evaluates on validation set with VGG16 encoder and SGD Optimizer, 0.9 momentum for 1 epochs |
| src/vgg16\_sgd\_nesterov.py |  Trains network and evaluates on validation set with VGG16 encoder and SGD Optimizer, 0.9 momentum, Nesterov Acceleration for 4 epochs |
| src/vgg16\_sgd\_zero\_mom.py |  Trains network and evaluates on validation set with VGG16 encoder and SGD Optimizer with 0 momentum for 1 epochs |
| src/vocab\_creator.py | Parses the training dataset of all the captions and generates vocabulary used in all experiments |


