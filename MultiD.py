import torch
from torch import nn
from torch.nn import functional as F


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4 # 2048//4 = 512
        # input shape B 2048 12 12
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)

class Discriminator_with_multi_tasks(nn.Module):
    def __init__(self, num_classes=20) -> None:
        super(Discriminator_with_multi_tasks, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 256 x 256 /   (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 128 x 128 / (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 64 x 64 / (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 32 x 32 /  (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 16 x 16 / (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier_real_or_fake = nn.Sequential(
            # state size. (512) x 16 x 16 /  (512) x 8 x 8
            # nn.Flatten(),
            # nn.Linear(512 * 16 * 16, 1024),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(1024, 1),

            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, kernel_size=16),  # 全卷积代替全连接  shape num_classes C 1 1
        )

        self.classifier_multi_labels = nn.Sequential(
            # # state size. (512) x 16 x 16 / (512) x 8 x 8
            # nn.Flatten(),
            # nn.Linear(512 * 16 * 16, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            #
            # nn.Linear(1024, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            #
            # nn.Linear(256, num_classes + 1)

            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            nn.Conv2d(256, num_classes+1, kernel_size=16)  # 全卷积代替全连接  shape num_classes C 1 1

        )

        self.semseg_pred_mask = nn.Sequential(
            # state size. (512) x 16 x 16 -> (21) x 16x 16

            # state size. (512) x 8 x 8 -> (21) x 8 x 8
            # VGG 16的 三个全连接换为全卷积
            nn.Conv2d(512, 2048, 3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(2048, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            FCNHead(in_channels=1024, channels=num_classes + 1)
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # get input shape H W
        input_shape = x.shape[-2:]
        # shared backbone extract features
        out = self.features(x)  # B 512 12 12

        # trunk for adjuest real or fake
        real_or_fake_logits = self.classifier_real_or_fake(out)
        real_or_fake_logits = real_or_fake_logits.view(x.shape[0],-1)

        # trunk for pred labels
        pred_label_logits = self.classifier_multi_labels(out)
        pred_label_logits = pred_label_logits.view(x.shape[0],-1)

        # trunk for semseg pred mask
        pred_semseg_logits = self.semseg_pred_mask(out)
        pred_semseg_logits = F.interpolate(pred_semseg_logits, size=input_shape, mode='bilinear', align_corners=False)

        return real_or_fake_logits, pred_label_logits, pred_semseg_logits



class Discriminator_with_semseg(nn.Module):
    def __init__(self, num_classes=20) -> None:
        super(Discriminator_with_semseg, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 256 x 256 /   (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 128 x 128 / (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 64 x 64 / (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 32 x 32 /  (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 16 x 16 / (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier_real_or_fake = nn.Sequential(

            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False),  # 全卷积代替全连接  shape num_classes C 1 1
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )


        self.semseg_pred_mask = nn.Sequential(
            # state size. (512) x 16 x 16 -> (21) x 16x 16
            # state size. (512) x 8 x 8 -> (21) x 8 x 8
            # VGG 16的 三个全连接换为全卷积
            nn.Conv2d(512, 2048, 3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(2048, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            FCNHead(in_channels=1024, channels=num_classes + 1)
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # get input shape H W
        input_shape = x.shape[-2:]
        # shared backbone extract features
        out = self.features(x)  # B 512 12 12

        # trunk for adjuest real or fake
        real_or_fake_logits = self.classifier_real_or_fake(out)
        real_or_fake_logits = real_or_fake_logits.view(x.shape[0],-1)

        # trunk for semseg pred mask
        pred_semseg_logits = self.semseg_pred_mask(out)
        pred_semseg_logits = F.interpolate(pred_semseg_logits, size=input_shape, mode='bilinear', align_corners=False)

        return real_or_fake_logits, pred_semseg_logits


class Discriminator_cat_mask_with_semseg(nn.Module):
    def __init__(self, num_classes=20) -> None:
        super(Discriminator_cat_mask_with_semseg, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 256 x 256 /   (3) x 128 x 128
            nn.Conv2d(3+num_classes+1, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 128 x 128 / (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 64 x 64 / (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 32 x 32 /  (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 16 x 16 / (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier_real_or_fake = nn.Sequential(

            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False),  # 全卷积代替全连接  shape num_classes C 1 1
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )


        self.semseg_pred_mask = nn.Sequential(
            # state size. (512) x 16 x 16 -> (21) x 16x 16
            # state size. (512) x 8 x 8 -> (21) x 8 x 8
            # VGG 16的 三个全连接换为全卷积
            nn.Conv2d(512, 2048, 3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(2048, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            FCNHead(in_channels=1024, channels=num_classes + 1)
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # get input shape H W
        input_shape = x.shape[-2:]
        # shared backbone extract features
        out = self.features(x)  # B 512 12 12

        # trunk for adjuest real or fake
        real_or_fake_logits = self.classifier_real_or_fake(out)
        real_or_fake_logits = real_or_fake_logits.view(x.shape[0],-1)

        # trunk for semseg pred mask
        pred_semseg_logits = self.semseg_pred_mask(out)
        pred_semseg_logits = F.interpolate(pred_semseg_logits, size=input_shape, mode='bilinear', align_corners=False)

        return real_or_fake_logits, pred_semseg_logits

class Discriminator_with_classify(nn.Module):
    def __init__(self, num_classes=20) -> None:
        super(Discriminator_with_classify, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 256 x 256 /   (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 128 x 128 / (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 64 x 64 / (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 32 x 32 /  (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 16 x 16 / (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier_real_or_fake = nn.Sequential(

            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, kernel_size=1,stride=1,bias=False),  # 全卷积代替全连接  shape num_classes C 1 1
            nn.AdaptiveAvgPool3d((1,1,1))
        )

        self.classifier_multi_labels = nn.Sequential(

            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            # no bg class
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=False),  # 全卷积代替全连接  shape num_classes 16 16
            nn.AdaptiveAvgPool3d((num_classes, 1, 1))

            # nn.Conv2d(256, num_classes+1, kernel_size=1, stride=1, bias=False),  # 全卷积代替全连接  shape num_classes 16 16
            # nn.AdaptiveAvgPool3d((num_classes+1,1,1))


        )


    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # get input shape H W
        input_shape = x.shape[-2:]
        # shared backbone extract features
        out = self.features(x)  # B 512 12 12

        # trunk for adjuest real or fake
        real_or_fake_logits = self.classifier_real_or_fake(out)
        real_or_fake_logits = real_or_fake_logits.view(x.shape[0],-1)

        # trunk for pred labels
        pred_label_logits = self.classifier_multi_labels(out)
        pred_label_logits = pred_label_logits.view(x.shape[0],-1)


        return real_or_fake_logits, pred_label_logits

class Discriminator_cat_label_with_classify(nn.Module):
    def __init__(self, num_classes=20, concat=False) -> None:
        super(Discriminator_cat_label_with_classify, self).__init__()

        # 是否需要拼接label到img上
        self.concat = concat

        if self.concat:
            # numclasses = 21
            # 拼接后的channel数量
            # self.inchannel = 3 + num_classes + 1
            # num_classes = 20
            self.inchannel = 3 + num_classes

        else:
            self.inchannel = 3

        self.features = nn.Sequential(
            # input size. (3) x 256 x 256 /   (3) x 128 x 128
            nn.Conv2d(self.inchannel, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 128 x 128 / (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 64 x 64 / (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 32 x 32 /  (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 16 x 16 / (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier_real_or_fake = nn.Sequential(

            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, kernel_size=1,stride=1,bias=False),  # 全卷积代替全连接  shape num_classes C 1 1
            nn.AdaptiveAvgPool3d((1,1,1))
        )

        self.classifier_multi_labels = nn.Sequential(

            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),  # shape B C 8 8
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            # 如果是num——class = 20

            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=False),  # 全卷积代替全连接  shape num_classes 16 16
            nn.AdaptiveAvgPool3d((num_classes + 1, 1, 1))

            # nn.Conv2d(256, num_classes+1, kernel_size=1, stride=1, bias=False),  # 全卷积代替全连接  shape num_classes 16 16
            # nn.AdaptiveAvgPool3d((num_classes+1,1,1))


        )


    def forward(self, x: torch.Tensor, label: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # get input shape H W
        input_shape = x.shape[-2:]
        # mul label in image
        if self.concat:
            label = label.unsqueeze(2).unsqueeze(3) # 扩展为四个维度 [B,C,1,1]
            label = label.repeat(1, 1, x.size(2), x.size(3)) # [B,C,x_h,x_w]
            x = torch.cat((x, label), dim=1) # 按照channel 维度进行拼接
        else:
            x = x

        # shared backbone extract features
        out = self.features(x)  # B 512 12 12

        # trunk for adjuest real or fake
        real_or_fake_logits = self.classifier_real_or_fake(out)
        real_or_fake_logits = real_or_fake_logits.view(x.shape[0],-1)

        # trunk for pred labels
        pred_label_logits = self.classifier_multi_labels(out)
        pred_label_logits = pred_label_logits.view(x.shape[0],-1)


        return real_or_fake_logits, pred_label_logits

if __name__ == '__main__':
    input = torch.rand([2,3,256,256])
    label = torch.ones([2,21])
    # model = Discriminator_with_semseg(num_classes=20)
    model = Discriminator_cat_label_with_classify(num_classes=20,concat=True)

    real_or_fake_logits, pred_label_logits = model(input, label)

    print(real_or_fake_logits.shape)
    print(pred_label_logits.shape)
