import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms

from collections import OrderedDict
import cv2
import numpy as np

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2 ,return_indices=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2 ,return_indices=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2 ,return_indices=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2 ,return_indices=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,1000),
            nn.Softmax(dim=1)
        )

        self.feature_maps = OrderedDict()
        self.pool_switch = OrderedDict()

        self.init_weight()

    def init_weight(self):
        vgg16 = models.vgg16(pretrained=True)
        pre_features = vgg16.features
        for i,layer in enumerate(pre_features):
            if isinstance(layer, nn.Conv2d):
                self.features[i].weight.data = layer.weight.data
                self.features[i].bias.data = layer.bias.data

        pre_classifier = vgg16.classifier
        for i,layer in enumerate(pre_classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[i].weight.data = layer.weight.data
                self.classifier[i].bias.data = layer.bias.data

    def forward(self, x):
        for idx,layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, max_loc = layer(x)
            else:
                x = layer(x)

        x = x.view(1, -1)
        output = self.classifier(x)
        return output


if __name__ == '__main__':
    img = cv2.imread('../data/deconv/pug.jpg')
    img = cv2.resize(img, (224,224))
    print(img.shape)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    img = transform(img)
    img = img.unsqueeze_(0)
    print(img.shape)

    vgg16 = VGG16()
    output = vgg16(img)
    print(torch.argmax(output[0]))
