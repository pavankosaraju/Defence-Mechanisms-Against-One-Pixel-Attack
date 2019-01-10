import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_data(size="all",model = True):

    if model:
        transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.3),transforms.RandomVerticalFlip(p=0.3),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))])
    else:
        transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.3),transforms.RandomVerticalFlip(p=0.3),transforms.ToTensor()])
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
	
    def subsetting(dt,m):
        # Subset the dataset 'dt' such that it has 'm' images for each class
        indc=[]
        count=0
        sum = 0
        actual_size = m*10
        # Inintialize each class has 0 images
        cls = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
        for k in dt:
            if sum <= actual_size:
                datas,lb= k
                # For every image increment label 'lb' count in 'cls'
                cls[lb]=cls.get(lb)+1
                if cls[lb]<=m:
                    # If the count of label 'lb' is is less than 'm' then append indices of the image into the list
                    indc.append(count)
                    sum = sum + 1
                count = count+1
        #print("Data subset done")
        # Subset the data 'dt' containing only indices present in list 'indc'
        fdt = torch.utils.data.Subset(dt,indc)
        return fdt
    
	
    if size == "medium":
        trainset = subsetting(trainset,2500)
        testset = subsetting(testset,500)
    elif size == "small":
        trainset = subsetting(trainset,1500)
        testset = subsetting(testset,300)
		
    return trainset,testset

