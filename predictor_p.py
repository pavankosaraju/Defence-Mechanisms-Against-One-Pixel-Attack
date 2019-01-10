import sys
import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.enhanced_resnet import EnhancedResnet
from resnetmodel import ResNet
#import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt

dt="test"
idx = 99
pixel=None

model = ResNet()
checkpoint = torch.load('./resnet56.th')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

mym = EnhancedResnet()   
dnl = torch.load('./utils/logs/denoiser.pth')
mym.denoised_layer.load_state_dict(dnl['model'])
mym.denoised_layer.eval()

classes = {0:"Airplane",1:"Automobile",2:"Bird",3:"Cat",4:"Deer",5:"Dog",6:"Frog",7:"Horse",8:"Ship",9:"Truck"}
if dt == "train":
    imgs = datasets.CIFAR10('./data', train=True, download=True)
    tr = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    tnsr,lb = tr.__getitem__(idx)
    img = imgs.__getitem__(idx)[0]
else:
    imgs = datasets.CIFAR10('./data', train=False, download=True)
    tr = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    tnsr,lb = tr.__getitem__(idx)
    img = imgs.__getitem__(idx)[0]
	
if pixel is not None:	
    x = pixel[0]
    y = pixel[1]
    tnsr[0][x][y] = pixel[2]/255
    tnsr[1][x][y] = pixel[3]/255
    tnsr[2][x][y] = pixel[4]/255

tt = torch.reshape(tnsr,(1,3,32,32))		
pilTrans = transforms.ToPILImage()
pilImg = pilTrans(tnsr)

f = plt.figure(figsize=(6,3))
plt.axis('off')
plt.imshow(pilImg)
plt.show()
	
wo = model(tt)
wovals = wo.detach().numpy()
wolabel = np.argmax(wovals)
wolevel = (np.amax(wovals))*100
	
prdct = mym.denoised_layer(tt)
prdct = model(prdct)
npvals = prdct.detach().numpy()
label = np.argmax(npvals)
level = (np.amax(npvals))*100
	

print("\nResnet RESULT - Actual label is "+classes[lb]+" and model predicted it as "+classes[wolabel]+" with confidence {0: .2f}".format(wolevel))   
print("\nEnhancedResnet RESULT - Actual label is "+classes[lb]+" and model predicted it as "+classes[label]+" with confidence {0: .2f}".format(level))

