import torch
import torch.nn as nn

# n1 := num_in_channels
# n2 := base_filter

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class SRCNN(torch.nn.Module):
    def __init__(self , num_channels , base_filter , upscale_factor):
        super(SRCNN , self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels , out_channels=base_filter , kernel_size=9 , padding=4 , stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=base_filter , out_channels=int(base_filter/2) , kernel_size=5 , padding=2 , stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=int(base_filter/2) , out_channels=num_channels , kernel_size=5 , padding=2 , stride=1)
        self.upscale_factor = upscale_factor
        
    def forward(self , x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        nn.PixelShuffle(self.upscale_factor)

        return x

    
    def weight_initialization(self , mean , std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    