import torch
import torch.nn as nn

class block_a(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1)
        self.BN1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.BN3 = nn.BatchNorm2d(out_channel)
        self.Drop = nn.Dropout2d(p=0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.Drop(x)
        
        return x


class block_aa(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1)
        self.BN1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, dilation=2, padding=2)
        self.BN2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, dilation=2, padding=2)
        self.BN3 = nn.BatchNorm2d(out_channel)
        self.Drop = nn.Dropout2d(p=0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.Drop(x)
        
        return x

class block_aaa(nn.Module):
    def __init__(self, in_channel, out_channel, dil):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, dilation=dil, padding=dil)
        self.BN1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, dilation=dil, padding=dil)
        self.BN2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, dilation=dil, padding=dil)
        self.BN3 = nn.BatchNorm2d(out_channel)
        self.Drop = nn.Dropout2d(p=0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.Drop(x)
        
        return x


class block_b(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.BN1 = nn.BatchNorm2d(out_channel)
        self.Tconv = nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BNT = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.BN3 = nn.BatchNorm2d(out_channel)
        self.Drop = nn.Dropout2d(p=0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        
        x = self.conv1(x)
        x = self.BN1(x)
        x = nn.ReLU()(x)
        x = self.Tconv(x)
        x = self.BNT(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.Drop(x)
        
        return x
        

class block_c(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.BN = nn.BatchNorm2d(out_channel)
        self.Tconv = nn.ConvTranspose2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BNT = nn.BatchNorm2d(out_channel)
        self.Drop = nn.Dropout2d(p=0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        
        x = self.conv(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.Tconv(x)
        x = self.BNT(x)
        x = self.Drop(x)
        
        return x


class block_d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.BN1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)
        self.BN2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax2d()
    
    def forward(self, x):
                             
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.softmax(x)
        
        return x


class NetForBertgrid(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        C = 64
        self.encoder1 = block_a(in_channel, C)
        self.encoder2 = block_a(C, 2*C)
        self.encoder3 = block_aa(2*C, 4*C)
        self.encoder4 = block_aaa(4*C, 8*C, 4)
        self.encoder5 = block_aaa(8*C, 8*C, 8)
        
        self.decoder1 = block_b(12*C, 4*C)
        self.decoder2 = block_b(6*C, 2*C)
        self.decoder3 = block_c(3*C, C)
        self.decoder4 = block_d(C, out_channel)
    
    def forward(self, x):
        out1 = self.encoder1(x)
        out2 = self.encoder2(out1)
        out3 = self.encoder3(out2)
        x = self.encoder4(out3)
        x = self.encoder5(x)
        
        x = self.decoder1(out3, x)
        x = self.decoder2(out2, x)
        x = self.decoder3(out1, x)
        x = self.decoder4(x)
        
        return x
