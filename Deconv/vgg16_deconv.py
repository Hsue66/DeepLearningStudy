import torch
import torch.nn as nn
import torchvision.models as models

class VGG16_Deconv(nn.Module):
    def __init__(self):
        super(VGG16_Deconv, self).__init__()

        self.features = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512,256,kernel_size=3,stride=1,padding=1),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=1,padding=1),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=1,padding=1),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,3,kernel_size=3,stride=1,padding=1),
        )

        self.index_mapping = {0:30, 2: 28, 5:25, 7:23, 10:20, 12:18, 14:16, 17:13, 19:11, 21:9, 24:6, 26:4, 28:2}
        self.pool_mapping = {26:4, 21:9, 14:16, 7:23, 0:30}#{4:26, 9:21, 16:14, 23:7, 30:0}

        self.init_weight()

    def init_weight(self):
        vgg16 = models.vgg16(pretrained=True)
        pre_features = vgg16.features
        for i,layer in enumerate(pre_features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.index_mapping[i]].weight.data = layer.weight.data


    def forward(self, x, layer_idx, pool_switch):
        deconv_idx = self.index_mapping[layer_idx]
        print(deconv_idx)

        for i in range(deconv_idx, len(self.features)):
            #print(self.features[i])
            if isinstance(self.features[i], nn.MaxUnpool2d):
                x = self.features[i](x, pool_switch[self.pool_mapping[i]])
            else:
                x = self.features[i](x)

        return x


if __name__ == '__main__':
    vgg16_deconv = VGG16_Deconv()
    vgg16_deconv(0)
