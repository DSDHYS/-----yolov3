import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter
import math
from collections import OrderedDict

class ConvolutionalLayer_last(nn.Module):
    def __init__(self,in_channels,out_channels,kernal_size,stride,padding):
        super(ConvolutionalLayer_last,self).__init__()
        # self.out_channels=32# Conv2d 32x32x3
        # self.conv1=nn.Conv2d(3,self.out_channels,kernel_size=3, stride=1, padding=1, bias=False)
        # #第一个3是因为inputs通道数为3;kernel_size=3是卷积核的通道数,32x32x3中的3;
        # self.BN1=nn.BatchNorm2d(self.out_channels)#参数为卷积后输入尺寸;该步进行归一化
        # self.relu1 = nn.LeakyReLU(0.1)#激活函数,一般取0.1
        # #实现inputs-Conv2D 32x3x3; 416,416,3 -> 416,416,32；第一个卷积
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernal_size,stride,padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):#前向传递
        return self.conv1(x)

# class ConvolutionalLayer(nn.Module):
#     def __init__(self,in_channels,out_channels,kernal_size,stride,padding):
#         super(ConvolutionalLayer,self).__init__()
#         # self.out_channels=32# Conv2d 32x32x3
#         # self.conv1=nn.Conv2d(3,self.out_channels,kernel_size=3, stride=1, padding=1, bias=False)
#         # #第一个3是因为inputs通道数为3;kernel_size=3是卷积核的通道数,32x32x3中的3;
#         # self.BN1=nn.BatchNorm2d(self.out_channels)#参数为卷积后输入尺寸;该步进行归一化
#         # self.relu1 = nn.LeakyReLU(0.1)#激活函数,一般取0.1
#         # #实现inputs-Conv2D 32x3x3; 416,416,3 -> 416,416,32；第一个卷积
#         self.conv1=nn.Sequential(
#             nn.Conv2d(in_channels,out_channels,kernal_size,stride,padding, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.1)
#         )
#     def forward(self,x):#前向传递
#         return self.conv1(x)

class ConvolutionalLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernal_size,stride,padding):
        super(ConvolutionalLayer,self).__init__()
        # self.out_channels=32# Conv2d 32x32x3
        # self.conv1=nn.Conv2d(3,self.out_channels,kernel_size=3, stride=1, padding=1, bias=False)
        # #第一个3是因为inputs通道数为3;kernel_size=3是卷积核的通道数,32x32x3中的3;
        # self.BN1=nn.BatchNorm2d(self.out_channels)#参数为卷积后输入尺寸;该步进行归一化
        # self.relu1 = nn.LeakyReLU(0.1)#激活函数,一般取0.1
        # #实现inputs-Conv2D 32x3x3; 416,416,3 -> 416,416,32；第一个卷积
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernal_size,stride,padding, bias=False),
            nn.BatchNorm2d(out_channels),
            Mish()
        )
    def forward(self,x):#前向传递
        return self.conv1(x)
#-------------------------------------------------#
#   MISH激活函数
#-------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


    #残差网络,由一个1x1的卷积核一个3x3的卷积构成
class ResidualLayer(nn.Module):
    def __init__(self,in_channels):
        super(ResidualLayer, self).__init__()        
        self.ReseBlock=nn.Sequential(
            ConvolutionalLayer(in_channels,in_channels//2,kernal_size=1,stride=1,padding=0),#第一次的输出变为输入的1/2，通道收缩#1X1的卷积核，所以padding=0
            ConvolutionalLayer(in_channels//2,in_channels,kernal_size=3,stride=1,padding=1)#第二次的输出变为输入的2倍，通道扩张还原，3x3的卷积核，416x416需要进行0填充
        )

    def forward(self,x):
        return x+self.ReseBlock(x)#X+两次卷积的结果，详见残差网络流程图

    #残差块的叠加,分别进行1，2，8，8，4次
# class make_layers(nn.Module):
#     def __init__(self,in_channels,count):
#         super(make_layers, self).__init__()
#         self.count= count
#         self.in_channels=in_channels
#         self.RB=ResidualLayer(self.in_channels)
#     def forward(self,x):#for循环实现叠加
#         for i in range(0,self.count):
#             x=self.RB(x)
#         return x

# #下采样层
# class DownSampleLayer(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(DownSampleLayer, self).__init__()
#         self.conv=ConvolutionalLayer(in_channels,out_channels,kernal_size=3, stride=2, padding=1)#见cfg文件
#     def forward(self,x):
#         return self.conv(x)

def make_layer(in_channels,out_channels, count):
    layers = []
    for i in range(0, count):
        layers.append(("residual_{}".format(i), ResidualLayer(out_channels)))
    return nn.Sequential(OrderedDict(layers))


#下采样层
class DownSampleLayer(nn.Module):
    def __init__(self,in_channels,out_channels,count):
        super(DownSampleLayer, self).__init__()
        self.DS=nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,kernal_size=3, stride=2, padding=1),#见cfg文件
            make_layer(in_channels,out_channels,count),
        )
    def forward(self,x):
        return self.DS(x)
#上采样层
class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()
    def forward(self,x):
        return F.interpolate(x,scale_factor=2,mode='nearest')

#搭建DarkNet53,获得三个特征层
class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()

        self.RB_56=nn.Sequential(
            ConvolutionalLayer(3,32,3,1,1),#32X32X3
            DownSampleLayer(32, 64,1),#下采样，通道扩张
            #ResidualLayer(64),#残差网络
            DownSampleLayer(64, 256,1),
            #make_layers(128, 2),#进行两次残差
            #make_layers(256, 8)
        )
        self.RB_28=nn.Sequential(
            DownSampleLayer(256,512,8),
            #make_layers(512,8)
        )
        self.RB_14=nn.Sequential(
            DownSampleLayer(512,1024,8),
            #make_layers(1024,4),
        )
        self.RB_7=nn.Sequential(
            DownSampleLayer(1024,2048,4),
            #make_layers(1024,4),
        )
        self.contact_7=nn.Sequential(
            Conv2d_Block_5L(2048,1024)
        )
        self.contact_14=nn.Sequential(
            Conv2d_Block_5L(1536,512)
        )
        self.contact_28=nn.Sequential(
            Conv2d_Block_5L(768,256)
        )
        self.contact_56=nn.Sequential(
            Conv2d_Block_5L(384,128)
        )

        self.out_7=nn.Sequential(
            ConvolutionalLayer(1024,2048,3,1,1),
            nn.Conv2d(2048,36,1,1,0)#36=(5+检测类数)*3
        )
        self.out_14=nn.Sequential(
            ConvolutionalLayer_last(512,1024,3,1,1),
            nn.Conv2d(1024,36,1,1,0)#36=(5+检测类数)*3
        )
        self.out_28=nn.Sequential(
            ConvolutionalLayer_last(256,512,3,1,1),
            nn.Conv2d(512,36,1,1,0)#36=(5+检测类数)*3
        )
        self.out_56=nn.Sequential(
            ConvolutionalLayer_last(128,256,3,1,1),
            nn.Conv2d(256,36,1,1,0)#36=(5+检测类数)*3
        )

        self.up_56=nn.Sequential(
            ConvolutionalLayer(256,128,1,1,0),
            UpSampleLayer(),
            )#上采样,
        self.up_28=nn.Sequential(
            ConvolutionalLayer(512,256,1,1,0),
            UpSampleLayer()
            )#上采样,
        self.up_14=nn.Sequential(
            ConvolutionalLayer(1024,512,1,1,0),
            UpSampleLayer(),
            )#上采样,

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self,x):
        RB_56=self.RB_56(x)
        RB_28=self.RB_28(RB_56)
        RB_14=self.RB_14(RB_28)
        RB_7=self.RB_7(RB_14)
        
        conval_7=self.contact_7(RB_7)
        out_7 = self.out_7(conval_7)

        up_14 = self.up_14(conval_7)
        route_14 = torch.cat((up_14,RB_14),dim=1)
        conval_14 = self.contact_14(route_14)
        out_14= self.out_14(conval_14)

        up_28 = self.up_28(conval_14)
        route_28 = torch.cat((up_28,RB_28),dim=1)
        conval_28 = self.contact_28(route_28)
        out_28= self.out_28(conval_28)

        up_56 = self.up_56(conval_28)
        route_56 = torch.cat((up_56,RB_56),dim=1)
        conval_56 = self.contact_56(route_56)
        out_56= self.out_56(conval_56)


        
        return  out_7,out_14,out_28,out_56



class Conv2d_Block_5L(nn.Module):#6个conv+bn+leakyReLU
    def __init__(self,in_channels,out_channels):
        super(Conv2d_Block_5L, self).__init__()
        self.Conv=nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,1,1,0),
            ConvolutionalLayer(out_channels,in_channels,3,1,1),
            ConvolutionalLayer(in_channels,out_channels,1,1,0),
            ConvolutionalLayer(out_channels,in_channels,3,1,1),
            ConvolutionalLayer(in_channels,out_channels,1,1,0)#卷积5次
        )
    def forward(self,x):
        return self.Conv(x)


# 搭建输出网络
# class yolov3(nn.Module):
#     def __init__(self):
#         super(yolov3, self).__int__()
#         self.out_1=nn.Sequential(
#             DarkNet53[0],
#             Conv2d_Block_5L(28,512),#输入28，输出512
#             ConvolutionalLayer(512,28,3,1,1),
#             ConvolutionalLayer(28,36,3,1,1)#36为（5+种类数）*3
#         )
#         self.out_2=nn.Sequential(
#             #route_14 = torch.cat((up_14,h_14),dim=1)
#             torch.cat((DarkNet53(1),DarkNet53(3)),dim=1),
#             Conv2d_Block_5L(512,256),#输入512，输出256
#             ConvolutionalLayer(256,512,3,1,1),
#             ConvolutionalLayer(512,36,3,1,1)#36为（5+种类数）*3
#         )
#         self.out_3=nn.Sequential(
#             #route_14 = torch.cat((up_14,h_14),dim=1)
#             torch.cat((DarkNet53(2),DarkNet53(4)),dim=1),
#             Conv2d_Block_5L(256,128),#输入512，输出256
#             ConvolutionalLayer(128,256,3,1,1),
#             ConvolutionalLayer(256,36,3,1,1)#36为（5+种类数）*3
#         )
#         def forward(self,x):
#             out_1=self.out_1(x)
#             out_2=self.out_2(x)
#             out_3=self.out_3(x)

#             return out_1,out_2,out_3

# X = np.empty
# X=yolov3()


#结构可视化
# input=torch.rand(32,3,28,28)
# model=DarkNet53()

# with SummaryWriter(log_dir='logs',comment='data\DarkNet53') as w:
#     w.add_graph(model,(input,))


#输出尺寸
# model = DarkNet53()

# tfms = transforms.Compose([transforms.Resize(448), transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
# img = tfms(Image.open(r'E:\document\syn\data\graduation-project\yolo3-pytorch-master\VOCdevkit\VOC2007\img_18242.jpg')).unsqueeze(0)
# print(img.shape) # torch.Size([1, 3, 416, 416])

# out_7,out_14,out_28,out_56 = model(img)
# print(out_7.shape)
# print(out_14.shape)
# print(out_28.shape)
# print(out_56.shape)
