import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoEncoder,self).__init__()
        self.encoder1 = nn.Sequential(
                        nn.Conv2d(3,32,(3,3),padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.ZeroPad2d(8),
                        nn.Conv2d(32,32,(3,3),padding=1), 
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.ZeroPad2d(8)
        )
        self.encoder2 = nn.Sequential(
                        nn.Conv2d(32,64,(3,3),padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64,128,(3,3),padding=1), 
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
        )
        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(128,64,3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64,32,3,padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32,3,3,padding=1),
                        nn.BatchNorm2d(3),
                        nn.Sigmoid()
        )
        
                
    def forward(self,x):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.decoder(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        #self.denoised_layer = denoise()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = self.denoised_layer(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
		

class EnhancedResnet(nn.Module):
    def __init__(self):
        super(EnhancedResnet, self).__init__()
        self.denoised_layer = DenoisingAutoEncoder()
        self.residualnet = ResNet(Bottleneck, [3,4,6,3])

    def forward(self, x):
        #out = self.denoised_layer(x)
        out = self.denoised_layer(x)
        out = self.residualnet(out)
        return out		
