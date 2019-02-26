import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from imutils import paths
import numpy as np
import random
import math

class FashionDataset(Dataset):
    def __init__(self, path, file_name, category_label_name, color_label_name, transform = None):
        self.transform = transform

        file_path = os.path.join(path, file_name)
        category_label_path = os.path.join(path, category_label_name)
        color_label_path = os.path.join(path, color_label_name)
        self.image_path = np.loadtxt(file_path, dtype=str)
        category_label = np.loadtxt(category_label_path, dtype=int, comments='#')
        color_label = np.loadtxt(color_label_path, dtype=int, comments='#')
        self.category_label = []
        self.color_label = []
        for i in range(len(category_label)):
            sub_category_label = category_label[i]
            value = 0
            '''
            for j in range(len(sub_category_label)):
                value = value + sub_category_label[j] * math.pow(2, j)
            value = int(value)
            '''
            for j in range(len(sub_category_label)):
                if sub_category_label[j] == 1:
                    value = j
                    break
            value = int(value)
            self.category_label.append(value)
        
        for i in range(len(color_label)):
            sub_color_label = color_label[i]
            value = 0
            '''
            for j in range(len(sub_color_label)):
                value = value + sub_color_label[j] * math.pow(2, j)
            value = int(value)
            '''
            for j in range(len(sub_color_label)):
                if sub_color_label[j] == 1:
                    value = j
                    break
            value = int(value)
            self.color_label.append(value)


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]) 
        image = image.convert('RGB')
        gray = image.convert('L')
        category_label = self.category_label[idx]
        color_label = self.color_label[idx]
        if self.transform is not None:
            image = self.transform(image)
            gray = self.transform(gray)
        
        sample = {'gray': gray, 'image': image, 'category': category_label, 'color': color_label}

        return sample

def main():
    path = './data'
    file_name = 'trainX.txt'
    category_label_name = 'trainCategoryY.txt'
    color_label_name = 'trainColorY.txt'
    transformations = transforms.Compose([
    transforms.Resize((96, 96)),
    #transforms.Normalize(),
    transforms.ToTensor()])

    fashion_dataset = FashionDataset(path, file_name, category_label_name, color_label_name, transform=transformations)
    for i in range(len(fashion_dataset)):
        sample = fashion_dataset[i]
        print(sample['gray'].size(), sample['image'].size())

if __name__ == '__main__':
    main()

