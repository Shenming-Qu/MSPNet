from torch import nn
import torch.nn.functional as F
import torch





class MultiScaleStripPooling(nn.Module):
    def __init__(self, in_channels, pool_size, norm_layer):
        super(MultiScaleStripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))




        inter_channels = int(in_channels / 4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))

        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     norm_layer(inter_channels))

        self.conv = nn.Sequential(nn.Conv2d(inter_channels,inter_channels,3,1,0, bias=False),norm_layer(inter_channels))

        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels * 3, in_channels, 1, bias=False),
                                   norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs = {'mode': 'bilinear', 'align_corners': True}

    def forward(self, x):
        _, _, h, w = x.size()
        #x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)

        #x2_1 = self.conv2_0(x1)
        #x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        #x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)


        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)

        x2_6 = F.interpolate(self.conv(nn.AvgPool2d((3,int(h/3)),2,1)(x2)), (h, w), **self._up_kwargs)
        x2_7 = F.interpolate(self.conv(nn.AvgPool2d(( int(w / 3),3), 2, 1)(x2)), (h, w), **self._up_kwargs)

        x2_8 = F.interpolate(self.conv(nn.AvgPool2d((7, int(h / 7)), 2, 1)(x2)), (h, w), **self._up_kwargs)
        x2_9 = F.interpolate(self.conv(nn.AvgPool2d((int(w / 7), 7), 2, 1)(x2)), (h, w), **self._up_kwargs)

        a = self.conv2_6(F.relu_(x2_6+x2_7))
        b = self.conv2_6(F.relu_(x2_8+x2_9))



        #x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))

        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([ x2, a, b], dim=1))
        return F.relu_(x + out)


