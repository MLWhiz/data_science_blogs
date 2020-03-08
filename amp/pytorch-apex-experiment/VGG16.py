import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v

        layers += [nn.Conv2d(512, 4096, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(4096), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(4096), nn.ReLU(inplace=True)]
        layers += {nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0, bias=False)}

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        return x.squeeze()

