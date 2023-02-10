from utils import torch, nn, crop



class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_in=False, use_bn=True, use_dropout=False, use_maxpool3=False):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.activation = nn.LeakyReLU(0.2)
        self.use_in = use_in
        self.use_bn = use_bn
        self.use_maxpool3 = use_maxpool3
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels*2)
        if use_in:
            self.instancenorm = nn.InstanceNorm2d(input_channels*2)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout()
            

    def forward(self, x):
        x = self.conv1(x)
        if self.use_in:
            x = self.instancenorm(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout: # Check if layer1 and layer2 shutdowns are same
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_in:
            x = self.instancenorm(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        if self.use_maxpool3:
            x = self.maxpool3(x)
        else:
            x = self.maxpool(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, use_in=False, use_bn=True, use_dropout=False, scale_factor=2):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=2, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.use_in = use_in
        self.use_bn = use_bn
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels//2)
        if use_in:
            self.instancenorm = nn.InstanceNorm2d(input_channels//2)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout()

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = self.conv1(x)
        x_skip_cropped = crop(x_skip, x.shape)
        x = torch.cat([x, x_skip_cropped], dim=1)
        x = self.conv2(x)
        if self.use_in:
            x = self.instancenorm(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_in:
            x = self.instancenorm(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=16):
        super(UNet, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=False)
        self.contract2 = ContractingBlock(hidden_channels*2, use_dropout=False)
        self.contract3 = ContractingBlock(hidden_channels*4, use_dropout=False)
        self.contract4 = ContractingBlock(hidden_channels*8, use_maxpool3=True)
        self.contract5 = ContractingBlock(hidden_channels*16, use_maxpool3=True)
        self.expand1 = ExpandingBlock(hidden_channels*32, scale_factor=3)
        self.expand2 = ExpandingBlock(hidden_channels*16, scale_factor=3)
        self.expand3 = ExpandingBlock(hidden_channels*8)
        self.expand4 = ExpandingBlock(hidden_channels*4)
        self.expand5 = ExpandingBlock(hidden_channels*2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.expand1(x5, x4)
        x7 = self.expand2(x6, x3)
        x8 = self.expand3(x7, x2)
        x9 = self.expand4(x8, x1)
        x10 = self.expand5(x9, self.dropout(x0))
        x11 = self.downfeature(x10)
        return self.leaky_relu(x11)


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            FeatureMapBlock(input_channels, hidden_channels),
            ContractingBlock(hidden_channels, use_in=True, use_bn=False),
            ContractingBlock(hidden_channels*2, use_in=True, use_bn=False),
            ContractingBlock(hidden_channels*4, use_in=True, use_bn=False),
            FeatureMapBlock(hidden_channels*8, 1)
        )

    def forward(self, x):
        return self.disc(x)
