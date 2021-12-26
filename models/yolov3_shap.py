import numpy as np
import torch
import torch.nn as nn

def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch//2, 1, 1))
            resblock_one.append(add_conv(ch//2, ch, 3, 1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x



class YOLOLayer(nn.Module):
    """
    Detection Layer
    """    
    def __init__(self, in_ch, n_anchors, n_classes):
        super(YOLOLayer, self).__init__()
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=0)

    def forward(self, x, targets=None):
        output = self.conv(x)
        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])
        
        return output




class YOLOv3SHAP(nn.Module):
    """
    YOLOv3 model module for calculating SHAP
    """
    def __init__(self, n_classes):
        """
        Initialization of YOLOv3 class.
        """
        super(YOLOv3SHAP, self).__init__()
        self.n_classes = n_classes
        self.module_list = nn.ModuleList()
        # DarkNet53
        self.module_list.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
        self.module_list.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
        self.module_list.append(resblock(ch=64))
        self.module_list.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
        self.module_list.append(resblock(ch=128, nblocks=2))
        self.module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
        self.module_list.append(resblock(ch=256, nblocks=8))    # shortcut 1 from here
        self.module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
        self.module_list.append(resblock(ch=512, nblocks=8))    # shortcut 2 from here
        self.module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
        self.module_list.append(resblock(ch=1024, nblocks=4))

        # YOLOv3
        self.module_list.append(resblock(ch=1024, nblocks=2, shortcut=False))
        self.module_list.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
        # 1st yolo branch
        self.module_list.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
        self.module_list.append(
                YOLOLayer(in_ch=1024, n_anchors=3, n_classes=self.n_classes))

        self.module_list.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
        self.module_list.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.module_list.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))
        self.module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
        self.module_list.append(resblock(ch=512, nblocks=1, shortcut=False))
        self.module_list.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
        # 2nd yolo branch
        self.module_list.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
        self.module_list.append(
            YOLOLayer(in_ch=512, n_anchors=3, n_classes=self.n_classes))
        self.module_list.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
        self.module_list.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.module_list.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))
        self.module_list.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
        self.module_list.append(resblock(ch=256, nblocks=2, shortcut=False))
        self.module_list.append(
            YOLOLayer(in_ch=256, n_anchors=3, n_classes=self.n_classes))

    def forward(self, x):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """
        output = []
        route_layers = []
        for i, module in enumerate(self.module_list):
            # yolo layers
            if i in [14, 22, 28]:
                x = module(x)
                output.append(x)
            else:
                x = module(x)

            # route layers = shortcut
            if i in [6, 8, 12, 20]:
                route_layers.append(x)
            if i == 14:
                x = route_layers[2]
            if i == 22:  # yolo 2nd
                x = route_layers[3]
            if i == 16:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 24:
                x = torch.cat((x, route_layers[0]), 1)

        return output