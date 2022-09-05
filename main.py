import io
import os
import torchvision.transforms as transforms
from PIL import Image
from classes import AlexNet
import torch
import torch.nn as nn

model = AlexNet()
model = torch.load('melanoma_CNN27.pt', map_location=torch.device('cpu'))
model.eval()

def transform_image():
    input_transforms= [transforms.Resize((227, 227)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(r'./uploaded/image.jpg')
    timg = my_transforms(image)
    timg.unsqueeze_(0)
    return timg


def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction



input_tensor = transform_image()
prediction = get_prediction(input_tensor)

print(prediction)
