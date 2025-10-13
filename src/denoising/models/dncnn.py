import torch 
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, depth = 17, features = 64, c = 3):
        """
        Args:
            depth: Number of layers
            features: Number of filters
            c: Reprsents the number of image channels. (c = 1 gray, c = 3 color)
        """
        super(DnCNN, self).__init__()
        kernel = 3
        padd = 1
        layers = []

        #first layer (Conv + ReLU)
        layers.append(nn.Conv2d(c, features, kernel_size = kernel, padding = padd, bias=True))
        layers.append(nn.ReLU(inplace = True))

        #middle layers (Conv + BN + ReLU)
        for i in range(depth-2):
            layers.append(nn.Conv2d(features, features, kernel_size = kernel, padding = padd, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace = True))

        #last layer (Conv to reconstruct the output)
        layers.append(nn.Conv2d(features, c, kernel_size = kernel, padding = padd, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.dncnn(x) #predict the noise
        return residual  #output = clean estimate