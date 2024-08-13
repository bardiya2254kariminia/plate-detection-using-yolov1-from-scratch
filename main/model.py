import torch
import torch.nn as nn

"""
yolov1 implementation with added batchnorm 
"""
"""
architecture_config  = (kernel_size, num_filters(output_channels) ,  stride , padding , )

we use 2 anchor box for the predictions for each slice so we have 20 class label
and 5 for each anchor box
(c0 , --- , c19 , pc , x ,  y , w,  h ,pc , x , y , h , w)

"""
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
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
    def __init__(self ,in_channels , out_channels, kernel_size , stride , padding):
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels , 
                              out_channels=out_channels , 
                              kernel_size=kernel_size , 
                              stride=stride ,
                              padding = padding)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        out = self.conv(x)
        out = self.batchnorm(out)
        return self.leakyrelu(out)


class Yolov1(nn.Module):
    def __init__(self, split_size, num_boxes , num_classes , in_channels=3):
        super(Yolov1 , self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture_config
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(split_size, num_boxes , num_classes)

    def forward(self , x:torch.tensor):
        out = self.darknet(x)
        # become (batch_num , S * S * 30)
        out = torch.flatten(out , start_dim=1)
        return self.fcs(out)
    
    def _create_conv_layers(self, architecture):
        layers = []
        # in_channels for every for turn
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers.append(
                    CNNBlock(
                        in_channels= in_channels,
                        out_channels = x[1],
                        kernel_size= x[0],
                        stride = x[2],
                        padding =  x[3] 
                    )
                )
                in_channels = x[1]
            elif type(x) == str:
                layers.append(
                    nn.MaxPool2d(
                        kernel_size = 2 ,
                        stride = 2
                    )
                )
            else : 
                conv1 , conv2 , num_repeats = x[0] , x[1] , x[2]

                for _ in range(num_repeats):
                    layers.append(
                        CNNBlock(
                            in_channels=in_channels ,
                            out_channels= conv1[1],
                            kernel_size= conv1[0],
                            stride=conv1[2],
                            padding=conv1[3] 
                        )
                    )
                    layers.append(
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride= conv2[2],
                            padding=conv2[3]
                        )
                    )
                in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes , num_classes):
        S , B , C = split_size , num_boxes , num_classes
        """
        in original paper we have a 4096 layers output somewhere but though 
        to the resource we only use 496
        """

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S , 496), # it is in the original paper
            nn.Dropout(p=0.0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(496 , S*S*(C + B *5)) 
        )