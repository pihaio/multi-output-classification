# USAGE
# python classify.py --model output/fashion.model --categorybin output/category_lb.pickle --colorbin output/color_lb.pickle --image examples/black_dress.jpg

# import the necessary packages
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

import numpy as np
import argparse
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--categorybin", required=True,
	help="path to output category label binarizer")
ap.add_argument("-c", "--colorbin", required=True,
	help="path to output color label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

transformations = transforms.Compose([
    transforms.Resize((96, 96)),
    #transforms.Normalize(),
    transforms.ToTensor()])

category_dict = ['dress', 'jeans', 'shirt', 'shoes']
color_dict = ['black', 'blue', 'red']

# load the image
image = Image.open(args["image"])

# pre-process the image for classification
image = image.convert('RGB')
gray = image.convert('L')
image = transformations(image)
gray = transformations(gray)
image.unsqueeze_(dim=0)
gray.unsqueeze_(dim=0)

#image = Variable(image)
#gray = Variable(gray)

# load the trained convolutional neural network from disk, followed
# by the category and color label binarizers, respectively
print("[INFO] loading network...")
categoryLB = torch.load(args["categorybin"])
categoryLB.eval()
colorLB = torch.load(args["colorbin"])
colorLB.eval()

# classify the input image using Keras' multi-output functionality
print("[INFO] classifying image...")
categoryProba = categoryLB(gray)
colorProba = colorLB(image)

# find indexes of both the category and color outputs with the
# largest probabilities, then determine the corresponding class
# labels
categoryIdx = categoryProba[0].argmax()
colorIdx = colorProba[0].argmax()

categoryLabel = category_dict[categoryIdx]
colorLabel = color_dict[colorIdx]

print categoryProba, categoryIdx
print colorProba, colorIdx

# draw the category label and color label on the image
categoryText = "category: {} ({:.2f}%)".format(categoryLabel,
	categoryProba[0][categoryIdx] * 100)
colorText = "color: {} ({:.2f}%)".format(colorLabel,
	colorProba[0][colorIdx] * 100)

print("[INFO] {}".format(categoryText))
print("[INFO] {}".format(colorText))
