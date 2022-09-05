import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset

transform = transforms.Compose([transforms.Resize((227, 227)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

batch_size = 64

train_set = torchvision.datasets.ImageFolder('./melanoma_cancer_dataset/train', 
                                            transform=transform)

test_set = torchvision.datasets.ImageFolder('./melanoma_cancer_dataset/test', 
                                            transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                          shuffle=True, num_workers=2)

classes = train_set.classes
