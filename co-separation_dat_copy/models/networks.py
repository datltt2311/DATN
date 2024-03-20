import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from collections import OrderedDict

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type='maxpool', input_channel=3, with_fc=False, fc_in=512, fc_out=256):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        #customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) #features before pooling
        #print self.feature_extraction

        if pool_type == 'conv1x1':
            self.conv1x1 = create_conv(512, 128, 1, 0)
            self.conv1x1.apply(weights_init)

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        x = self.feature_extraction(x)

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
        elif self.pool_type == 'conv1x1':
            x = self.conv1x1(x)
        else:
            return x #no pooling and conv1x1, directly return the feature map

        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            if self.pool_type == 'conv1x1':
                x = x.view(x.size(0), -1, 1, 1) #expand dimension if using conv1x1 + fc to reduce dimension
            return x
        else:
            return x.view(x.size(0), -1, 1, 1)
    def forward_multiframe(self, x, pool=True):
        (B, T, C, H, W) = x.size()
        x = x.contiguous()
        x = x.view(B * T, C, H, W)
        x = self.feature_extraction(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x 

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
        
        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.view(x.size(0), -1, 1, 1)
        else:
            return x.view(x.size(0), -1, 1, 1)
        return x
    
class AudioVisual7layerUNet_Audio(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual7layerUNet_Audio, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask
        self.conv1(68, 1, kernel_size=1)
        self.conv2(20, 1, kernel_size=1)
    def forward(self, x, visual_feat):
        # print("x size: ",x.size())
        # print("visual si ze: ", visual_feat.size())
        audio_conv1feature = self.audionet_convlayer1(x)
        # print("conv1",audio_conv1feature.size())
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)

        audioVisual_feature = torch.cat((x, audio_conv7feature), dim=1)
        # print("audiovisual", audioVisual_feature.size())
        # audioVisual_feature = torch.cat((audio_conv7feature, audio_conv7feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        # print("upconv1",audio_upconv1feature.size())
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))
        # print("upconv2",audio_upconv2feature.size())
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))
        # print("upconv3",audio_upconv3feature.size())
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))
        # print("upconv4",audio_upconv4feature.size())
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))
        # print("upconv5",audio_upconv5feature.size())
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))
        # print("upconv6",audio_upconv6feature.size())
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))
        # print("mask",mask_prediction.size())
        return mask_prediction
    

class AudioVisual5layerUNet_Audio(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual5layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(ngf * 24, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x, visual_embaddings, new_audio_embaddings):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_embadding_feat = visual_embaddings.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
        audio_embadding_feat = new_audio_embaddings.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
        audioVisual_feature = torch.cat((visual_embadding_feat, audio_embadding_feat, audio_conv7feature), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        return mask_prediction

class AudioVisuaPose7layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisuaPose7layerUNet, self).__init__()
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

        self.conv1 = nn.Conv2d(20, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.Maxpool2d(kernel_size=2, stride=2, )
        self.fc1 = nn.Linear(128*128*2, 512)

    def forward(self, pose_feat):
        # print("x size: ",x.size())
        # print("visual size: ", visual_feat.size())

        audio_conv1feature = self.audionet_convlayer1(x)
        # print("conv1",audio_conv1feature.size())
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)

        x = pose_feat
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)

        x = x.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])

        # visual_feat = visual_feat.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
        audioPose_feature = torch.cat((x, audio_conv7feature), dim=1)
        # print("audiovisual",audioVisual_feature.size())
        # audioVisual_feature = torch.cat((audio_conv7feature, audio_conv7feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        # print("upconv1",audio_upconv1feature.size())
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))
        # print("upconv2",audio_upconv2feature.size())
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))
        # print("upconv3",audio_upconv3feature.size())
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))
        # print("upconv4",audio_upconv4feature.size())
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))
        # print("upconv5",audio_upconv5feature.size())
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))
        # print("upconv6",audio_upconv6feature.size())
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))
        # print("mask",mask_prediction.size())
        return mask_prediction

class AudioVisual5layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual5layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[2], audio_conv5feature.shape[3])
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        return mask_prediction

class AudioPose7layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioPose7layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

        self.conv1 = nn.Conv2d(20, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 128 * 34, 512)

    def forward(self, x, visual_feature, pose_feat):

        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)

        visual_first_size = visual_feature.size(0)
        x = pose_feat.view(visual_first_size, 20, 68, 256)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print("x size: ", x.size())
        x = self.fc1(x)
        x = x.view(x.size(0), x.size(1), 1, 1).repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
        # print("x size after: ", x.size())
        # print("audio_conv7feature size: ", audio_conv7feature.size())

        # visual_feat = visual_feat.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
        audioVisual_feature = torch.cat((x, audio_conv7feature), dim=1)
        # print("audiovisual",audioVisual_feature.size())
        # audioVisual_feature = torch.cat((audio_conv7feature, audio_conv7feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        # print("upconv1",audio_upconv1feature.size())
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))
        # print("upconv2",audio_upconv2feature.size())
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))
        # print("upconv3",audio_upconv3feature.size())
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))
        # print("upconv4",audio_upconv4feature.size())
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))
        # print("upconv5",audio_upconv5feature.size())
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))
        # print("upconv6",audio_upconv6feature.size())
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))
        # print("mask",mask_prediction.size())
        return mask_prediction





