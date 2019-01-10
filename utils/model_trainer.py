#import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
    

def train_model(model,eps,bsize,trset,ttset):
     
	 
    trainloader = torch.utils.data.DataLoader(trset, batch_size=bsize, shuffle=True)
    testloader = torch.utils.data.DataLoader(ttset, batch_size=bsize, shuffle=True)
    accuracy = 0
    last_epoch = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
		
    if os.path.isfile("./utils/logs/resnet_last.pth"):
        print("Loading last checkpoint ...")
        checkpoint = torch.load('./utils/logs/resnet_last.pth')
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint['epoch']
        #if last_epoch<=eps:
        #    eps = eps - last_epoch
        #else:
        #   eps = eps - last_epoch
        accuracy = checkpoint['acc']
        model.eval()
        
		
    if torch.cuda.is_available():
        model = model.cuda()
        
    for epoch in range(eps):
        print('\nTraining model for epoch %d \n' %(epoch+1))
        for i, trdata in enumerate(trainloader, 0):
            trinputs, trlabels = trdata
            if torch.cuda.is_available():
                trinputs = trinputs.cuda()
                trlabels = trlabels.cuda()
            optimizer.zero_grad()
            troutputs = model(trinputs)
            trloss = criterion(troutputs, trlabels)
            trloss.backward()
            optimizer.step()
            if i % 20 == 0:    # print status every 20 mini-batches
                print('TRAINING STATUS - Running epoch %d, on batch %d' %(epoch + 1, i+20))
        if (epoch+1) % 2 == 0:
            print('\nValidating model after %d epochs' %(epoch+1))
            correct = 0
            total = 0
            for j, ttdata in enumerate(testloader,0):
                ttinputs, ttlabels = ttdata
                if torch.cuda.is_available():
                    ttinputs = ttinputs.cuda()
                    ttlabels = ttlabels.cuda()
                ttoutputs = model(ttinputs)
                #ttloss = criterion(ttoutputs, ttlabels)
                _, predicted = ttoutputs.max(1)
                total += ttlabels.size(0)
                correct += predicted.eq(ttlabels).sum().item()
                if j % 20 == 0:    # print status every 20 mini-batches
                    print('Validating STATUS - Running on batch %d' %(j+20))
            v = 100.*correct/total
            if accuracy < v:
                print("\nSaving model.")
                accuracy = v
                print('Saving...')
                state = {'model': model.state_dict(),'acc': accuracy,'epoch': epoch+last_epoch}
                torch.save(state, './utils/logs/resnet.pth')
            print("\nCurrent accuracy is {0: .2f} . Continue training ?".format(v))
            ans = input("(y/n): ")
            if ans == "n":
                print("\nSaving model.")
                accuracy = v
                print('Saving...')
                state = {'model': model.state_dict(),'acc': accuracy,'epoch': eps+last_epoch}
                torch.save(state, './utils/logs/resnet_last.pth')
                print("\nAaccuracy is {0: .2f} .".format(v))
                return "end"
    print("Training Finished.")
    print("\nSaving model.")
    accuracy = v
    print('Saving...')
    state = {'model': model.state_dict(),'acc': accuracy,'epoch': eps+last_epoch}
    torch.save(state, './utils/logs/resnet_last.pth')
    print("\nAaccuracy is {0: .2f} .".format(v))
    return "end_f"
				
				
