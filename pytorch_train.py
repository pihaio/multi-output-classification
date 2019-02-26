from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

import FashionDataset 

from models.fashionnet import CategoryNet, ColorNet 


path = './data'
file_name = 'trainX.txt'
category_label_name = 'trainCategoryY.txt'
color_label_name = 'trainColorY.txt'

test_file_name = 'testX.txt'
test_category_label_name = 'testCategoryY.txt'
test_color_label_name = 'testColorY.txt'

batch_size = 8
epochs = 6
category_labels = 4
color_labels = 3

transformations = transforms.Compose([
    transforms.Resize((96, 96)),
    #transforms.Normalize(),
    transforms.ToTensor()])

fashion_dataset = FashionDataset.FashionDataset(path, file_name, category_label_name, color_label_name, transform = transformations)
test_fashion_dataset = FashionDataset.FashionDataset(path, test_file_name, test_category_label_name, test_color_label_name, transform = transformations)

train_loader = DataLoader(fashion_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )

test_loader = DataLoader(test_fashion_dataset,
                         batch_size=100,
                         shuffle=False,
                         num_workers=4
                         )

color_model = ColorNet(color_labels)
category_model = CategoryNet(category_labels)

category_optimizer = optim.Adam(category_model.parameters(), lr=0.001)
category_criterion = nn.CrossEntropyLoss()
color_optimizer = optim.Adam(color_model.parameters(), lr=0.001)
color_criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for iter, traindata in enumerate(train_loader):
        gray, image, category, color = Variable(traindata['gray']), Variable(traindata['image']), Variable(traindata['category']), Variable(traindata['color'])

        category_train_outputs = category_model(gray)
        category_loss = category_criterion(category_train_outputs, category)
        category_optimizer.zero_grad()

        color_train_outputs = color_model(image)
        color_loss = color_criterion(color_train_outputs, color)
        color_optimizer.zero_grad()

        loss = color_loss + category_loss

        loss.backward()

        category_optimizer.step()
        color_optimizer.step()


        if iter % 100 == 0:
            for test_step, test_x in enumerate(test_loader):
                test_gray, test_image, test_category, test_color = Variable(test_x['gray']), Variable(test_x['image']), Variable(test_x['category']), Variable(test_x['color'])

                test_output = category_model(test_gray)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = torch.sum(pred_y == test_category).data.float() / float(test_category.size(0))

                color_test_output = color_model(test_image)
                color_pred_y = torch.max(color_test_output, 1)[1].data.squeeze()
                color_accuracy = torch.sum(color_pred_y == test_color).data.float() / float(test_color.size(0))

                if iter == 200:
                    print(pred_y[:10], 'category prediction number')
                    print(test_category[:10].numpy(), 'category real number')
                    print(color_pred_y[:10], 'color prediction number')
                    print(test_color[:10].numpy(), 'color real number')
                print('Epoch: ', epoch, 'step: ', iter, '| category loss: %.4f' % category_loss.data[0], '| category acc: %.2f' % accuracy, '| color loss: %.4f' % color_loss.data[0], '| color acc: %.2f' % color_accuracy)
                break


torch.save(category_model, './output/category_net.pkl')  # save entire net
torch.save(color_model, './output/color_net.pkl')  # save entire net
