import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 

class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result
        
class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))
        
        # 使用提供的ASPP模块替代原有定义
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )        

        self.branch1_0 = nn.Sequential(
            nn.Conv2d(24, 4, (3, 1), 1, (1, 0)), 
            nn.Conv2d(4, 4, (1, 3), 1, (0, 1))) 
        self.branch1_1 = nn.Sequential(
            nn.Conv2d(24, 4, (1, 3), 1, (0, 1)), 
            nn.Conv2d(4, 4, (3, 1), 1, (1, 0))) 
        
        self.branch2_0 = nn.Sequential(
            nn.Conv2d(24, 4, (7, 1), 1, (3, 0)),  
            nn.Conv2d(4, 4, (1, 7), 1, (0, 3)))  
        self.branch2_1 = nn.Sequential(
            nn.Conv2d(24, 4, (1, 7), 1, (0, 3)),  
            nn.Conv2d(4, 4, (7, 1), 1, (3, 0)))  
        
        self.branch3_0 = nn.Sequential(
            nn.Conv2d(24, 4, (11, 1), 1, (5, 0)),  
            nn.Conv2d(4, 4, (1, 11), 1, (0, 5)))  
        self.branch3_1 = nn.Sequential(
            nn.Conv2d(24, 4, (1, 11), 1, (0, 5)),  
            nn.Conv2d(4, 4, (11, 1), 1, (5, 0)))  
        
        self.branch4_0 = nn.Sequential(
            nn.Conv2d(24, 4, (15, 1), 1, (7, 0)),  
            nn.Conv2d(4, 4, (1, 15), 1, (0, 7)))  
        self.branch4_1 = nn.Sequential(
            nn.Conv2d(24, 4, (1, 15), 1, (0, 7)), 
            nn.Conv2d(4, 4, (15, 1), 1, (7, 0)))  
        
        self.br = nn.Sequential(
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1))
        
        self.cat_conv = nn.Sequential(
            nn.Conv2d(296, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
 
            nn.Conv2d(128, 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        self.cls_conv = nn.Conv2d(4, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        x0 = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)),
                           mode='bilinear', align_corners=True)
        
        branch1_0 = self.branch1_0(low_level_features)
        branch1_1 = self.branch1_1(low_level_features)
        branch1 = branch1_0 + branch1_1
        br1 = self.br(branch1)
        result1 = br1 + branch1
        
        branch2_0 = self.branch2_0(low_level_features)
        branch2_1 = self.branch2_1(low_level_features)
        branch2 = branch2_0 + branch2_1
        br2 = self.br(branch2)
        result2 = br2 + branch2
        
        branch3_0 = self.branch3_0(low_level_features)
        branch3_1 = self.branch3_1(low_level_features)
        branch3 = branch3_0 + branch3_1
        br3 = self.br(branch3)
        result3 = br3 + branch3
        
        branch4_0 = self.branch4_0(low_level_features)
        branch4_1 = self.branch4_1(low_level_features)
        branch4 = branch4_0 + branch4_1
        br4 = self.br(branch4)
        result4 = br4 + branch4
        
        x1 = torch.cat((result1, result2, result3, result4), dim=1)
        x2 = torch.cat((x1, low_level_features, x0), dim=1)
        x3 = self.cat_conv(x2)
        x4 = F.interpolate(x3, size=(H, W), mode='bilinear', align_corners=True)
        
        return self.cls_conv(x4)
