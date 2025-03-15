import torch
import torch.nn as nn
from torch.nn.modules import padding

architecture_config = [
    #tuple that describes the onv parameters (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    #M is the maxpool layer
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    #list of tuple that describes repat conv parameters 
    #[(kernel_size, num_filters, stride, padding), (kernel_size, num_filters, stride, padding), num_repeat]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) # bias is false becaus of the batchnorm
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyReLU(self.batchnorm(self.conv(x)))

class YoloV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs): 
        super(YoloV1, self).__init__()
        self.achitecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.achitecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, achitecture):
        layers = []
        in_channels = self.in_channels

        for x in achitecture:
            #adding a conv layers
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,
                        x[1],
                        kernel_size = x[0],
                        stride = x[2],
                        padding = x[3]
                    )
                ]
                in_channels = x[1]
            #adding a maxpooling layer
            elif type(x) == str:
                layers += [
                    nn.MaxPool2d(
                        kernel_size = (2,2),
                        stride = (2,2)
                    )
                ]

            #adding a sequence of conv layers
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeat = x[2]
                for _ in range(num_repeat):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size = conv1[0],
                            stride = conv1[2],
                            padding = conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3]
                        )
                    ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S,B,C = split_size, num_boxes, num_classes

        '''
        In the origianl papaer this should be 
        nn.Linear(1024*S*S, 4096)
        nn.LeakyReLU(0.1)
        nn.Linear(4096, S*S*(B*S+C))
        each image is split in S regions and each region can predict only one class
        to increase the number of objects detected we need to increase the S
        '''
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 496),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C+B*5))
        )

def test(S=7, B=2, C=20):
    model = YoloV1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2,3, 448,448))
    print(model(x).shape)

test()
