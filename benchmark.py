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

transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

batch_size = [16,32,64,128,256,512,1024,2048,4096,8192]
inference_time = []
for i in batch_size : 
    test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)
    test_loader = DataLoader(test_set, batch_size=i, shuffle=False, num_workers=0)

    #model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    #model = models.alexnet(weights="AlexNet_Weights.DEFAULT")
    #model = models.squeezenet1_0(weights="SqueezeNet1_0_Weights.DEFAULT")
    #model = models.googlenet(weights="GoogLeNet_Weights.DEFAULT")
    #model = models.vgg16(weights="VGG16_Weights.DEFAULT")
    model = models.vgg19(pretrained=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("The model will be running on", device, "device")

    start_time = time.time()
    total = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
    end_time = time.time()
    inference_time.append(end_time-start_time)
    print(end_time-start_time, " seconds in average for ", total, " samples of testing data, for batch_size = ", i)    
    
print(inference_time)
plt.plot(batch_size,inference_time)
