import sys
import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.enhanced_resnet import EnhancedResnet
#import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
#from data.data_utils import get_data

def plot_figs(img1,img2,img3=None, tot=2):
    f = plt.figure(figsize=(6,3))
    plt.axis('off')
    f.add_subplot(1,tot, 1)
    plt.imshow(img1)
    f.add_subplot(1,tot, 2)
    plt.imshow(img2)
    if tot>3:
        f.add_subplot(1,tot, 3)
        plt.imshow(img3)
    plt.show()

def imgdenoise(dt,idx,pixel=None):
    model = EnhancedResnet()
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))])
    
    if dt == "train":
        #imgs = datasets.CIFAR10('./data', train=True, download=True)
        tr = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        tnsr,lb = tr.__getitem__(idx)
        #img = imgs.__getitem__(id)[0]
    else:
        #imgs = datasets.CIFAR10('./data', train=False, download=True)
        tr = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
        tnsr,lb = tr.__getitem__(idx)
        #img = imgs.__getitem__(id)[0]
	
    if pixel is not None:	
        x = pixel[0]
        y = pixel[1]
        tnsr[0][x][y] = pixel[2]/255
        tnsr[1][x][y] = pixel[3]/255
        tnsr[2][x][y] = pixel[4]/255
    
    tt = torch.reshape(tnsr,(1,3,32,32))
    dnl = torch.load('./utils/logs/denoiser.pth')
    model.denoised_layer.load_state_dict(dnl['model'])
    model.denoised_layer.eval()
    
    wo = model.denoised_layer(tt)
    wo = torch.reshape(wo,(3,32,32))
    
    pilTrans = transforms.ToPILImage()
    pilImg1 = pilTrans(tnsr)
    pilImg2 = pilTrans(wo)
    print("1. Original Image 2. Denoise Image")
    plot_figs(pilImg1, pilImg2)

def predict(dt,idx,pixel=None):
    
    
    classes = {0:"Airplane",1:"Automobile",2:"Bird",3:"Cat",4:"Deer",5:"Dog",6:"Frog",7:"Horse",8:"Ship",9:"Truck"}
    model = EnhancedResnet()
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
		
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(tnsr)
	
    rsl = torch.load('./utils/logs/resnet.pth')    
    dnl = torch.load('./utils/logs/denoiser.pth')
    model.denoised_layer.load_state_dict(rsl['model'])
    model.residualnet.load_state_dict(dnl['model'])
    model.eval()
	
    wo = model.residualnet(tnsr)
    wovals = wo.detach().numpy()
    wolabel = np.argmax(wovals)
    wolevel = (np.amax(wovals))*100
	
    prdct = model(tnsr)
    npvals = prdct.detach().numpy()
    label = np.argmax(npvals)
    level = (np.amax(npvals))*100
	

    print("\nResnet RESULT - Actual label is "+classes[lb]+" and model predicted it as "+classes[wolabel]+" with confidence {0: .2f}".format(wolevel))   
    print("\nEnhancedResnet RESULT - Actual label is "+classes[lb]+" and model predicted it as "+classes[label]+" with confidence {0: .2f}".format(level))
    print("1. Original Image 2. Perturbed Image")
    plot_figs(img, pilImg)
	
if __name__ == '__main__':

    # Predict class of 99th image in CIFAR10 test set
    # If pixel = (x,y,c) is given then it will predict class of that preturbed image
    #predict("test",id = 99,pixel=[16,16,255,255,0])
    imgdenoise("test",58,[16,16,255,255,0])
    
		
        		
	
	
