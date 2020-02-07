

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        #self.pool1 = nn.MaxPool2d(2, 2)
        #self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 48, 5, stride = 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(48, 64, 5, stride = 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(64)
        
        


        '''

        self.conv4 = nn.Conv2d(64, 80, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(80)

        self.conv5 = nn.Conv2d(256, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.bn5 = nn.BatchNorm2d(512)

        '''
        
        
        self.linear1 = nn.Linear(64 * 12 * 12 , 512)
        self.linear2 = nn.Linear(512, 256)
        #self.linear3 = nn.Linear(512 , 512)
        #self.linear3 = nn.Linear(512 , 512)
        self.linear4 = nn.Linear(256 , 136)



        
      

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.bn3(self.pool3(F.relu(self.conv3(x))))
        
        x = x.view(x.size(0) , -1)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        x = self.linear4(x)
        
        return x


      #self.conv1 = nn.Conv2d(3, 32, 5)
        
        #print(self.conv1)
       
