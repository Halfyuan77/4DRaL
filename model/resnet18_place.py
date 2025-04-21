import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.unet import UNet_tiny

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet18FeatureExtractor, self).__init__()
        # 使用 pretrained ResNet18
        resnet18 = models.resnet18(weights='DEFAULT')
        
        # 去掉全连接层和平均池化层，只保留到最后的卷积层
        self.features = nn.Sequential(*list(resnet18.children())[:-3])

        # 选项：使用非就地的 ReLU
        for layer in self.features:
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

    def forward(self, x):
        x = self.features(x)
        return x

class resnet_place(nn.Module):
    def __init__(self, require_init=True):
        super(resnet_place, self).__init__()
        # self.opt = opt
        self.require_init = require_init
        # self.encoder = CustomNet()
        
        self.encoder = ResNet18FeatureExtractor()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.g2r = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),           # Conv1
            nn.ReLU())
        
    def forward(self, input):                                                  # ([32, 3, 3, 200, 200])
        
        batch_size, input_c, input_h, input_w = input.shape           # (32, 3, 3, 200, 200)
        input = input.view(batch_size, input_c, input_h, input_w)    # ([32*3, 3, 200, 200])
        
        if input_c < 3:
            input = self.g2r(input)
        feature = self.encoder(input)                                          # ([96, 256, 12, 12])
        
        feature = self.pool(feature)                                           # ([96, 256, 6, 6])
        feature = feature.view(batch_size, -1)                                                 # ([32, 9216])
        feature = F.normalize(feature, p=2, dim=1)                                             # ([32, 9216])
        
        return feature

class resnet_place_stu(nn.Module):
    def __init__(self, require_init=True):
        super(resnet_place_stu, self).__init__()
        # self.opt = opt
        self.require_init = require_init
        
        # self.enhance_model = UNet(1,1)
        self.enhance_model = UNet_tiny(1,1)
        self.encoder = ResNet18FeatureExtractor()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.g2r = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),           # Conv1
            nn.ReLU())
        
    def forward(self, input):                                                  # ([32, 3, 3, 200, 200])
        
        input = self.enhance_model(input)
        stage1 = input
        
        batch_size, input_c, input_h, input_w = input.shape           # (32, 3, 3, 200, 200)
        input = input.view(batch_size, input_c, input_h, input_w)    # ([32*3, 3, 200, 200])
        
        if input_c < 3:
            input = self.g2r(input)
        feature = self.encoder(input)                                          # ([96, 256, 12, 12])
        
        feature = self.pool(feature)                                           # ([96, 256, 6, 6])
        feature = feature.view(batch_size, -1)                                                 # ([32, 9216])
        feature = F.normalize(feature, p=2, dim=1)                                             # ([32, 9216])
        
        return stage1, feature

class resnet_place_teacher_v2(nn.Module):
    def __init__(self, require_init=True):
        super(resnet_place_teacher_v2, self).__init__()
        # self.opt = opt
        self.require_init = require_init
        # self.encoder = CustomNet()
        
        self.encoder = ResNet18FeatureExtractor()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.g2r = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),           # Conv1
            nn.ReLU(inplace=False))
        
        
    def forward(self, input):
        
        batch_size, input_c, input_h, input_w = input.shape
        input = input.view(batch_size, input_c, input_h, input_w)
        
        if input_c < 3:
            input = self.g2r(input)
        feature = self.encoder(input)
        
        feature = self.pool(feature)
        encorder_out = feature
        feature = feature.view(batch_size, -1)
        feature = F.normalize(feature, p=2, dim=1)
        
        return encorder_out, feature

class resnet_place_v2(nn.Module):
    def __init__(self, require_init=True):
        super(resnet_place_v2, self).__init__()

        self.require_init = require_init

        self.enhance_model = UNet_tiny(1, 1)
        self.encoder = ResNet18FeatureExtractor()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.g2r = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=False)  # Ensure no in-place operation
        )
        
        self.pie = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),  # Avoid in-place operation
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        
        input = self.enhance_model(input)
        stage1 = input  # Save intermediate output

        batch_size, input_c, input_h, input_w = input.shape

        if input_c < 3:
            input = self.g2r(input)
        
        feature = self.encoder(input)
        feature = self.pool(feature)

        stage2 = self.pie(feature)
        stage2_out = stage2
        feature = torch.add(feature, stage2_out)
        

        feature = feature.contiguous().view(batch_size, -1)
        feature = F.normalize(feature, p=2, dim=1)

        return stage1, stage2, feature

class resnet_place_v2_r2l(nn.Module):

    def __init__(self, require_init=True):
        super(resnet_place_v2_r2l, self).__init__()

        self.require_init = require_init
        
        # self.enhance_model = UNet(1, 1)
        self.enhance_model = UNet_tiny(1, 1)
        self.encoder = ResNet18FeatureExtractor()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.g2r = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(inplace=False)  # Ensure no in-place operation
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        
        input = self.enhance_model(input)
        stage1 = input  # Save intermediate output

        batch_size, input_c, input_h, input_w = input.shape

        if input_c < 3:
            input = self.g2r(input)
        
        feature = self.encoder(input)
        feature = self.pool(feature)

        feature = feature.contiguous().view(batch_size, -1)
        
        # Normalize the feature
        feature = F.normalize(feature, p=2, dim=1)
        
        return stage1,  feature