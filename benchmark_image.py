import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torch.nn as nn

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torchvision.models as models
import matplotlib.pyplot as plt

#transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#Transformation for AlexNet
transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=True)


#model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
#model = models.resnet34(weights="ResNet34_Weights.DEFAULT")
#model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
#model = models.resnet101(weights="ResNet101_Weights.DEFAULT")
#model = models.resnet152(weights="ResNet152_Weights.DEFAULT")

model = models.alexnet(weights="AlexNet_Weights.DEFAULT")

#model = models.squeezenet1_0(weights="SqueezeNet1_0_Weights.DEFAULT")
#model = models.squeezenet1_1(weights="SqueezeNet1_1_Weights.DEFAULT")

#model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
#model = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")
#model = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")

#model = models.vgg11(weights="VGG11_Weights.DEFAULT")
#model = models.vgg11_bn(weights="VGG11_BN_Weights.DEFAULT")
#model = models.vgg13(weights="VGG13_Weights.DEFAULT")
#model = models.vgg13_bn(weights="VGG13_BN_Weights.DEFAULT")
#model = models.vgg16(weights="VGG16_Weights.DEFAULT")
#model = models.vgg16_bn(weights="VGG16_BN_Weights.DEFAULT")
#model = models.vgg19(weights="VGG19_Weights.DEFAULT")
#model = models.vgg19_bn(weights="VGG19_BN_Weights.DEFAULT")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("The model will be running on", device, "device")

model.eval() 


with torch.no_grad():
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)
    start_time = time.time()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    end_time = time.time()

print(end_time-start_time, " seconds for 1 image")    
