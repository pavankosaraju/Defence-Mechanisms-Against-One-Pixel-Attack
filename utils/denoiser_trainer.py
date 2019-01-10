#import sys
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim


def corrupt_pixel(imgs):

    for p in range(imgs.shape[0]):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247, 0.243, 0.261]
        c = random.randint(0,2)
        x = random.randint(0,31)
        y = random.randint(0,31)
        noise = random.random()
        noise = (noise - mean[c])/std[c]
        imgs[p][c][x][y] = noise
		
    return imgs
	
	
def train_denoiser(model,eps,bsize,trset,ttset):
        
  
	
    trainloader = torch.utils.data.DataLoader(trset, batch_size=bsize, shuffle=True)
    testloader = torch.utils.data.DataLoader(ttset, batch_size=bsize, shuffle=True)
    last_epoch = 0
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
		
    if os.path.isfile("./utils/logs/denoiser.pth"):
        print("Loading last checkpoint ...")
        checkpoint = torch.load('./utils/logs/denoiser.pth')
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint['epoch']
        print("Last epoch: %d "%(last_epoch+1))
        #if last_epoch<eps:
        #    eps = eps - last_epoch
        #elif last_epoch == eps:
        #   eps = eps - last_epoch
        #accuracy = checkpoint['acc']
        model.eval()
		
    if torch.cuda.is_available():
        model = model.cuda()
        
    for epoch in range(eps):
        print('\nTraining denoiser layers for epoch %d \n' %(epoch+1))
        for i, trdata in enumerate(trainloader, 0):
            trinputs, trlabels = trdata
            cor_inputs = trinputs
            if i%2 == 0:
                cor_inputs = corrupt_pixel(cor_inputs)
            if torch.cuda.is_available():
                cor_inputs = cor_inputs.cuda()
                trinputs = trinputs.cuda()
            optimizer.zero_grad()
            troutputs = model(cor_inputs)
            trloss = criterion(troutputs, trinputs)
            trloss.backward()
            optimizer.step()
            if i % 20 == 0:    # print status every 20 mini-batches
                print('TRAINING STATUS - Running on batch %d' %(i+20))
        for j, ttdata in enumerate(testloader, 0):
            ttinputs, ttlabels = ttdata
            cor_inputs = ttinputs
            if j%2 == 0:
                cor_inputs = corrupt_pixel(cor_inputs)
            if torch.cuda.is_available():
                cor_inputs = cor_inputs.cuda()
                ttinputs = ttinputs.cuda()
            optimizer.zero_grad()
            ttoutputs = model(cor_inputs)
            ttloss = criterion(ttoutputs, ttinputs)
            ttloss.backward()
            optimizer.step()
            if j % 20 == 0:    # print status every 20 mini-batches
                print('TRAINING STATUS - Running on batch %d' %(j+20))
        if (epoch+1) % 2 == 0:
            print("\nSaving model.")
            print('Saving..')
            state = {'model': model.state_dict(),'epoch': epoch+last_epoch}
            torch.save(state, './utils/logs/denoiser.pth')
            #ans = input("\nContinue training? (y/n): ")
            #if ans=="n":
                #return "end"
    print("Finished Training")
    return "end_f"
				
				
