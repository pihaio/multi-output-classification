# USAGE
# python train.py --dataset dataset --model output/fashion.model \
#	--categorybin output/category_lb.pickle --colorbin output/color_lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data, clothing category labels (i.e., shirts, jeans,
# dresses, etc.) along with the color labels (i.e., red, blue, etc.)
data = []
categoryLabels = []
colorLabels = []

# loop over the input images
for imagePath in imagePaths:
	# update the respective lists
	(color, cat) = imagePath.split(os.path.sep)[-2].split("_")
        # print imagePath
        # print color, cat
	categoryLabels.append(cat)
	colorLabels.append(color)

# convert the label lists to NumPy arrays prior to binarization
categoryLabels = np.array(categoryLabels)
colorLabels = np.array(colorLabels)
print categoryLabels
print colorLabels

# binarize both sets of labels
print("[INFO] binarizing labels...")
categoryLB = LabelBinarizer()
colorLB = LabelBinarizer()
categoryLabels = categoryLB.fit_transform(categoryLabels)
colorLabels = colorLB.fit_transform(colorLabels)
print categoryLabels
print colorLabels

split = train_test_split(imagePaths, categoryLabels, colorLabels,
	test_size=0.2, random_state=42)
(trainX, testX, trainCategoryY, testCategoryY,
	trainColorY, testColorY) = split

# np.savetxt("categoryLabels.txt", categoryLabels, fmt="%d %d %d %d", delimiter="\n")
# np.savetxt("colorLabels.txt", colorLabels, fmt="%d %d %d", delimiter="\n")
# np.savetxt("data.txt", imagePaths, fmt="%s", delimiter="\n")

np.savetxt("data/trainX.txt", trainX, fmt="%s", delimiter="\n")
np.savetxt("data/testX.txt", testX, fmt="%s", delimiter="\n")
np.savetxt("data/trainCategoryY.txt", trainCategoryY, fmt="%d %d %d %d", delimiter="\n")
np.savetxt("data/testCategoryY.txt", testCategoryY, fmt="%d %d %d %d", delimiter="\n")
np.savetxt("data/trainColorY.txt", trainColorY, fmt="%d %d %d", delimiter="\n")
np.savetxt("data/testColorY.txt", testColorY, fmt="%d %d %d", delimiter="\n")

